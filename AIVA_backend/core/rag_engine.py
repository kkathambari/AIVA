import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from rank_bm25 import BM25Okapi

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
                    tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]
                    self.bm25 = BM25Okapi(tokenized_corpus)
            except Exception as e:
                print(f"Error loading BM25 corpus: {e}")

    def index_document(self, text: str, document_id: str) -> int:
        chunks = self.text_splitter.split_text(text)
        
        if not chunks:
            return 0
            
        import time
        import re

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
                        # Try to parse retry delay from the error message (e.g. 'retry in 13s'), fallback to delay
                        match = re.search(r"retry in (\d+\.?\d*)s", error_msg)
                        sleep_time = float(match.group(1)) if match else delay
                        print(f"Rate limit (429) hit during embedding batch {i // batch_size + 1}. Sleeping for {sleep_time:.2f} seconds before retry...")
                        time.sleep(sleep_time + 1.0) # add a 1 second buffer
                        delay *= 1.5
                    else:
                        raise e
        
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"document_id": document_id} for _ in chunks]
        
        self.collection.add(
            embeddings=embeddings_list,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        # Save chunks for BM25
        self.corpus.extend(chunks)
        os.makedirs(os.path.dirname(self.chunks_cache_file), exist_ok=True)
        with open(self.chunks_cache_file, "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, ensure_ascii=False, indent=2)
            
        tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        return len(chunks)

    def query(self, query_text: str, n_results: int = 3) -> list:
        # 1. Dense search using ChromaDB
        query_embedding = self.embeddings.embed_query(query_text)
        
        # Request slightly more candidates for fusion
        candidates_to_retrieve = max(n_results * 3, 10)
        
        chroma_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidates_to_retrieve
        )
        
        dense_hits = []
        if chroma_results['documents'] and chroma_results['documents'][0]:
            dense_hits = chroma_results['documents'][0]
            
        # 2. Sparse search using BM25
        sparse_hits = []
        if self.bm25 and self.corpus:
            tokenized_query = query_text.lower().split(" ")
            doc_scores = self.bm25.get_scores(tokenized_query)
            # Get top indices
            top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:candidates_to_retrieve]
            sparse_hits = [self.corpus[i] for i in top_indices if doc_scores[i] > 0]
            
        # 3. Reciprocal Rank Fusion (RRF)
        # Combine dense and sparse hits
        all_hits = set(dense_hits + sparse_hits)
        rrf_scores = {}
        k = 60
        
        for doc in all_hits:
            score = 0.0
            if doc in dense_hits:
                rank_dense = dense_hits.index(doc) + 1
                score += 1.0 / (k + rank_dense)
            if doc in sparse_hits:
                rank_sparse = sparse_hits.index(doc) + 1
                score += 1.0 / (k + rank_sparse)
            rrf_scores[doc] = score
            
        # Sort by RRF score descending
        sorted_docs = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
        
        # If no hits, return empty or fallback
        if not sorted_docs:
            return []
            
        return sorted_docs[:n_results]
