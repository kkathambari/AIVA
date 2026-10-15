import os
import json
import re
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from rank_bm25 import BM25Okapi

def technical_tokenize(text: str) -> List[str]:
    """
    Standardized technical tokenizer for sparse retrieval in engineering/dissertation texts:
    - Normalizes text to lowercase.
    - Preserves alphanumeric tokens, hyphens, and underscores within technical terms
      (e.g., 'docker-compose', 'fastapi', 'resnet-50', 'big-o', 'time_complexity').
    - Discards standalone punctuation and solitary single-character tokens.
    """
    if not text:
        return []
    raw_tokens = re.findall(r'[a-zA-Z0-9_\-\.]+', text.lower())
    cleaned = []
    for token in raw_tokens:
        tok = token.strip(".-_")
        if len(tok) > 1 or tok.isalnum():
            cleaned.append(tok)
    return cleaned

class RAGEngine:
    def __init__(self, db_path="chroma_db", api_key=None):
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="aiva_docs",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Load BM25 index corpus if available
        self.chunks_cache_file = os.path.join(db_path, "bm25_chunks.json")
        self.bm25 = None
        self.corpus = []
        self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        if os.path.exists(self.chunks_cache_file):
            try:
                with open(self.chunks_cache_file, "r", encoding="utf-8") as f:
                    self.corpus = json.load(f)
                if self.corpus:
                    tokenized_corpus = [technical_tokenize(doc) for doc in self.corpus]
                    self.bm25 = BM25Okapi(tokenized_corpus)
            except Exception as e:
                print(f"Error loading BM25 corpus: {e}")

    def clear(self):
        """Clears all indexed chunks from ChromaDB and BM25 memory to ensure a fresh session."""
        try:
            # Delete collection in ChromaDB if it exists
            self.chroma_client.delete_collection("aiva_docs")
        except Exception as e:
            print(f"Info/Error deleting ChromaDB collection: {e}")
            
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name="aiva_docs",
                metadata={"hnsw:space": "cosine"}
            )
            # Ensure any lingering items are deleted
            all_data = self.collection.get()
            if all_data and all_data.get("ids"):
                self.collection.delete(ids=all_data["ids"])
        except Exception as e:
            print(f"Error resetting ChromaDB collection: {e}")
            
        # Clear BM25 memory
        self.corpus = []
        self.bm25 = None
        if os.path.exists(self.chunks_cache_file):
            try:
                os.remove(self.chunks_cache_file)
            except Exception as e:
                print(f"Error removing chunks cache: {e}")

    def index_document(self, text: str, document_id: str) -> int:
        chunks = self.text_splitter.split_text(text)
        
        if not chunks:
            return 0
            
        import time

        embeddings_list = []
        batch_size = 20  # Batch size of 20 to prevent overloading free tier quota
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            retries = 5
            delay = 5.0
            
            while retries > 0:
                try:
                    batch_embeddings = self.embeddings.embed_documents(batch)
                    embeddings_list.extend(batch_embeddings)
                    break
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        retries -= 1
                        if retries == 0:
                            raise e
                        match = re.search(r"retry in (\d+\.?\d*)s", error_msg)
                        sleep_time = float(match.group(1)) if match else delay
                        print(f"Rate limit (429) hit during embedding batch {i // batch_size + 1}. Sleeping for {sleep_time:.2f} seconds before retry...")
                        time.sleep(sleep_time + 1.0)
                        delay *= 1.5
                    else:
                        raise e
        
        ids = [f"{session_id or 'default'}_{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"document_id": document_id, "session_id": session_id or "default"} for _ in chunks]
        
        self.collection.add(
            embeddings=embeddings_list,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        # Save chunks for BM25 (scoped strictly to current document)
        self.corpus = list(chunks)
        os.makedirs(os.path.dirname(self.chunks_cache_file), exist_ok=True)
        with open(self.chunks_cache_file, "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, ensure_ascii=False, indent=2)
            
        tokenized_corpus = [technical_tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        return len(chunks)

    def query_dense(self, query_text: str, n_results: int = 3, session_id: Optional[str] = None) -> list:
        """Dense semantic retrieval via ChromaDB embeddings with optional session scoping."""
        query_embedding = self.embeddings.embed_query(query_text)
        where_filter = {"session_id": session_id} if session_id else None
        
        try:
            chroma_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(n_results, 1),
                where=where_filter
            )
            if chroma_results['documents'] and chroma_results['documents'][0]:
                return chroma_results['documents'][0][:n_results]
        except Exception as e:
            # Fallback without session filter if no docs found or filter error
            pass
            
        chroma_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(n_results, 1)
        )
        if chroma_results['documents'] and chroma_results['documents'][0]:
            return chroma_results['documents'][0][:n_results]
        return []

    def query_sparse_bm25(self, query_text: str, n_results: int = 3) -> list:
        """Sparse lexical retrieval via BM25Okapi using technical tokenization."""
        if not self.bm25 or not self.corpus:
            return []
        tokenized_query = technical_tokenize(query_text)
        if not tokenized_query:
            return []
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n_results]
        return [self.corpus[i] for i in top_indices if doc_scores[i] > 0]

    def query_hybrid(self, query_text: str, n_results: int = 3, k: int = 60, session_id: Optional[str] = None) -> list:
        """
        Hybrid retrieval combining Dense Semantic search and Sparse BM25 via Reciprocal Rank Fusion (RRF).
        RRF Score: sum(1 / (k + rank_i)) across retrieved candidate sets.
        """
        candidates_to_retrieve = max(n_results * 3, 10)
        
        # 1. Dense search
        dense_hits = []
        try:
            dense_hits = self.query_dense(query_text, n_results=candidates_to_retrieve, session_id=session_id)
        except Exception as e:
            print(f"RAGEngine dense search fallback error: {e}")

        # 2. Sparse search
        sparse_hits = []
        if self.bm25 and self.corpus:
            tokenized_query = technical_tokenize(query_text)
            if tokenized_query:
                doc_scores = self.bm25.get_scores(tokenized_query)
                top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:candidates_to_retrieve]
                sparse_hits = [self.corpus[i] for i in top_indices if doc_scores[i] > 0]

        # 3. Reciprocal Rank Fusion
        all_hits = list(dict.fromkeys(dense_hits + sparse_hits))
        if not all_hits:
            return []

        rrf_scores = {}
        for doc in all_hits:
            score = 0.0
            if doc in dense_hits:
                rank_dense = dense_hits.index(doc) + 1
                score += 1.0 / (k + rank_dense)
            if doc in sparse_hits:
                rank_sparse = sparse_hits.index(doc) + 1
                score += 1.0 / (k + rank_sparse)
            rrf_scores[doc] = score
            
        sorted_docs = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
        return sorted_docs[:n_results]

    def query(self, query_text: str, n_results: int = 3, session_id: Optional[str] = None) -> list:
        """Default query method using Hybrid RRF with optional session scoping."""
        return self.query_hybrid(query_text, n_results=n_results, k=60, session_id=session_id)

