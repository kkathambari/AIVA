"""
AIVA Research Benchmark: Hybrid RAG vs Gemini Dense vs BM25 vs TF-IDF Retrieval Evaluation
-----------------------------------------------------------------------------------------
Evaluates four distinct retrieval strategies across an expanded, multi-chapter engineering
dissertation corpus (25 passages, 15 independently labelled queries):

1. TF-IDF Baseline (tfidf_baseline): Lexical vectorizer with cosine similarity.
2. BM25 Lexical Retrieval (bm25_sparse): BM25Okapi with technical tokenization preserving compound tokens.
3. Gemini Dense Semantic Retrieval (gemini_dense): Google Generative AI Embeddings (models/gemini-embedding-2, 3072-dim) with cosine distance.
4. Hybrid Retrieval (hybrid_rrf): Reciprocal Rank Fusion (RRF, k=60) combining Gemini Dense and BM25 Sparse ranks.

Metrics Evaluated:
- Recall@1, Recall@3, Recall@5
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@3, nDCG@5)
"""

import os
import sys
import json
import math
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.rag_engine import technical_tokenize
from rank_bm25 import BM25Okapi
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Expanded 25-passage dissertation benchmark corpus across 5 technical chapters
BENCHMARK_CORPUS = [
    {
        "id": "chunk_0",
        "chapter": "Ch 1: API & Microservices",
        "text": "The AIVA architecture implements FastAPI as an asynchronous microservices gateway, handling incoming HTTP and WebSocket connections with sub-50ms latency using Uvicorn worker loops."
    },
    {
        "id": "chunk_1",
        "chapter": "Ch 1: API & Microservices",
        "text": "User authentication relies on JSON Web Tokens (JWT) signed with HMAC-SHA256, verifying cryptographically signed claims on every protected endpoint without database lookups."
    },
    {
        "id": "chunk_2",
        "chapter": "Ch 1: API & Microservices",
        "text": "Cross-Origin Resource Sharing (CORS) middleware is configured to restrict unauthorized origin domains while allowing bidirectional credentialed WebSocket handshakes."
    },
    {
        "id": "chunk_3",
        "chapter": "Ch 2: Cloud & Deployment",
        "text": "Deployment is containerized using Docker and orchestrated across Kubernetes clusters with Horizontal Pod Autoscaler (HPA) scaling replica pods when CPU exceeds 75 percent."
    },
    {
        "id": "chunk_4",
        "chapter": "Ch 2: Cloud & Deployment",
        "text": "Kubernetes readiness and liveness probes continuously monitor container health, preventing traffic from reaching uninitialized or hanging backend pods."
    },
    {
        "id": "chunk_5",
        "chapter": "Ch 2: Cloud & Deployment",
        "text": "The CI/CD pipeline runs GitHub Actions to automate linting, pytest unit suites, Docker image building, and container registry publishing upon pull requests."
    },
    {
        "id": "chunk_6",
        "chapter": "Ch 3: Machine Learning & Modeling",
        "text": "To prevent neural network overfitting during training, we applied Dropout regularization (p=0.3) along with early stopping based on validation cross-entropy loss."
    },
    {
        "id": "chunk_7",
        "chapter": "Ch 3: Machine Learning & Modeling",
        "text": "Weakness classification is performed by an XGBoost ensemble classifier trained on student turn score, question difficulty, concept coverage, and response length."
    },
    {
        "id": "chunk_8",
        "chapter": "Ch 3: Machine Learning & Modeling",
        "text": "Model hyperparameters were tuned using 5-fold cross-validation with grid search optimizing for Area Under the Receiver Operating Characteristic Curve (ROC-AUC)."
    },
    {
        "id": "chunk_9",
        "chapter": "Ch 4: Reinforcement Learning & Control",
        "text": "The Deep Q-Network (DQN) difficulty engine optimizes question difficulty transitions using an epsilon-greedy exploration schedule with gamma=0.99 and target network update interval of 500 steps."
    },
    {
        "id": "chunk_10",
        "chapter": "Ch 4: Reinforcement Learning & Control",
        "text": "The RL state vector encodes the last five student answer scores, current difficulty level, cumulative viva turn count, and normalized response latency."
    },
    {
        "id": "chunk_11",
        "chapter": "Ch 4: Reinforcement Learning & Control",
        "text": "A prioritized experience replay buffer stores state transition tuples (s, a, r, s') to break temporal correlation during stochastic gradient descent updates."
    },
    {
        "id": "chunk_12",
        "chapter": "Ch 5: Information Retrieval & Knowledge Graphs",
        "text": "For sparse textual retrieval, the system utilizes BM25Okapi scoring combined with an inverted index, tokenizing compound technical identifiers like docker-compose and time_complexity."
    },
    {
        "id": "chunk_13",
        "chapter": "Ch 5: Information Retrieval & Knowledge Graphs",
        "text": "Dense semantic retrieval uses Google Generative AI Embeddings (models/gemini-embedding-2) in ChromaDB, capturing semantic similarity across paraphrased natural language questions."
    },
    {
        "id": "chunk_14",
        "chapter": "Ch 5: Information Retrieval & Knowledge Graphs",
        "text": "Reciprocal Rank Fusion (RRF) with constant k=60 merges ranked candidate lists from dense and sparse retrieval engines to produce a robust consensus ranking."
    },
    {
        "id": "chunk_15",
        "chapter": "Ch 5: Information Retrieval & Knowledge Graphs",
        "text": "The knowledge graph is constructed from dissertation text by extracting technical entities and typed relationships using Gemini 2.5 Flash and NetworkX graph analysis."
    },
    {
        "id": "chunk_16",
        "chapter": "Ch 6: Database & Storage",
        "text": "Database queries are indexed with B-Tree composite indices in PostgreSQL on (student_id, session_id, created_at) to maintain O(log N) lookup time for user dialogue transcripts."
    },
    {
        "id": "chunk_17",
        "chapter": "Ch 6: Database & Storage",
        "text": "Redis is implemented as an in-memory caching layer for active examination session states and temporary token blacklists with a 15-minute time-to-live (TTL)."
    },
    {
        "id": "chunk_18",
        "chapter": "Ch 6: Database & Storage",
        "text": "Database schema migrations are tracked and executed deterministically using Alembic migration scripts to ensure continuous schema synchronization across staging and production."
    },
    {
        "id": "chunk_19",
        "chapter": "Ch 7: Speech & Multimodal",
        "text": "Speech-to-text utilizes Google Speech Recognition with browser MediaRecorder fallback, ensuring reliable multimodal audio transcription across web client browsers."
    },
    {
        "id": "chunk_20",
        "chapter": "Ch 7: Speech & Multimodal",
        "text": "Audio is captured from the browser microphone at 16kHz sampling rate in Opus format inside WebM containers before transmission over WebSocket streams."
    },
    {
        "id": "chunk_21",
        "chapter": "Ch 8: Multi-Agent Architecture",
        "text": "The viva examination panel employs LangGraph to coordinate four specialized examiner personas: Professor (theory), Industry Expert (production), Critic (trade-offs), and Examiner (general)."
    },
    {
        "id": "chunk_22",
        "chapter": "Ch 8: Multi-Agent Architecture",
        "text": "A traceable LangGraph router node inspects dialogue history and student responses to dynamically assign questions while recording routing rationale in audit trails."
    },
    {
        "id": "chunk_23",
        "chapter": "Ch 9: Monitoring & Fault Tolerance",
        "text": "Prometheus metrics scrapers collect request throughput, 95th percentile latency, and error rates, visualized on real-time Grafana dashboards."
    },
    {
        "id": "chunk_24",
        "chapter": "Ch 9: Monitoring & Fault Tolerance",
        "text": "A circuit breaker pattern wraps external third-party API dependencies, automatically opening to fast-fail and prevent cascade failures when upstream timeout exceeds 3 seconds."
    }
]

# 15 Independently Labelled Technical Queries:
# Mixed across exact keyword matches, paraphrased conceptual searches, and composite queries
BENCHMARK_QUERIES = [
    {
        "query": "How is FastAPI configured as an asynchronous microservices gateway?",
        "type": "exact_keyword",
        "relevant_ids": ["chunk_0"]
    },
    {
        "query": "How does the system ensure fast authentication without hitting the database on each request?",
        "type": "semantic_paraphrase",
        "relevant_ids": ["chunk_1"]
    },
    {
        "query": "Explain the container orchestration and autoscaling strategy when CPU spikes.",
        "type": "composite",
        "relevant_ids": ["chunk_3"]
    },
    {
        "query": "What health check probes stop traffic from being sent to dead or unready containers?",
        "type": "semantic_paraphrase",
        "relevant_ids": ["chunk_4"]
    },
    {
        "query": "What techniques were used during neural net training to stop the model from memorizing training data?",
        "type": "semantic_paraphrase",
        "relevant_ids": ["chunk_6"]
    },
    {
        "query": "How does the XGBoost classifier detect weaknesses from student performance features?",
        "type": "exact_keyword",
        "relevant_ids": ["chunk_7"]
    },
    {
        "query": "What are the reinforcement learning hyperparameters and exploration strategy for the DQN difficulty controller?",
        "type": "composite",
        "relevant_ids": ["chunk_9"]
    },
    {
        "query": "How does the experience replay memory break temporal autocorrelation during reinforcement learning?",
        "type": "exact_keyword",
        "relevant_ids": ["chunk_11"]
    },
    {
        "query": "How does BM25Okapi sparse lexical retrieval handle compound technical identifiers like docker-compose?",
        "type": "exact_keyword",
        "relevant_ids": ["chunk_12"]
    },
    {
        "query": "Why use Reciprocal Rank Fusion to combine dense semantic and sparse lexical search rankings?",
        "type": "composite",
        "relevant_ids": ["chunk_14"]
    },
    {
        "query": "How are database indexing and lookup times optimized for student transcript history in PostgreSQL?",
        "type": "exact_keyword",
        "relevant_ids": ["chunk_16"]
    },
    {
        "query": "What is the fast in-memory key-value cache used for ephemeral session data?",
        "type": "semantic_paraphrase",
        "relevant_ids": ["chunk_17"]
    },
    {
        "query": "What happens if the online speech-to-text recognition service fails during a viva?",
        "type": "semantic_paraphrase",
        "relevant_ids": ["chunk_19"]
    },
    {
        "query": "How does the LangGraph multi-agent panel assign questions among Professor, Critic, and Industry Expert?",
        "type": "exact_keyword",
        "relevant_ids": ["chunk_21", "chunk_22"]
    },
    {
        "query": "What mechanism protects the system from cascading service failures when an external API is down?",
        "type": "semantic_paraphrase",
        "relevant_ids": ["chunk_24"]
    }
]

def calculate_ndcg(retrieved_ids, relevant_ids, k=3):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)
        
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval_system():
    print("=" * 80)
    print("AIVA EMPIRICAL BENCHMARK: Retrieval Evaluation (TF-IDF vs BM25 vs Dense vs Hybrid)")
    print("=" * 80)
    print(f"Corpus Size:  {len(BENCHMARK_CORPUS)} technical dissertation passages")
    print(f"Query Count:  {len(BENCHMARK_QUERIES)} independently labelled queries")
    
    corpus_texts = [item["text"] for item in BENCHMARK_CORPUS]
    corpus_ids = [item["id"] for item in BENCHMARK_CORPUS]
    
    # 1. TF-IDF Baseline Setup
    print("\n[1/4] Fitting TF-IDF baseline vectorizer...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tfidf_matrix = tfidf.fit_transform(corpus_texts)
    
    # 2. BM25 Sparse Setup (with technical tokenizer)
    print("[2/4] Indexing BM25Okapi sparse corpus with technical tokenizer...")
    tokenized_corpus = [technical_tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 3. Dense Embeddings Setup (Gemini models/gemini-embedding-2)
    print("[3/4] Computing Google Gemini Dense Embeddings (models/gemini-embedding-2)...")
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    # Generate corpus embeddings in batches of 10 to respect quota
    corpus_embeddings = []
    for i in range(0, len(corpus_texts), 10):
        batch = corpus_texts[i:i + 10]
        emb_batch = embeddings_model.embed_documents(batch)
        corpus_embeddings.extend(emb_batch)
        time.sleep(1.0)
    corpus_emb_matrix = np.array(corpus_embeddings)
    print(f"Corpus embedding matrix built: {corpus_emb_matrix.shape}")
    
    # Generate query embeddings
    print("Generating query embeddings...")
    query_embeddings = []
    for q in BENCHMARK_QUERIES:
        q_emb = embeddings_model.embed_query(q["query"])
        query_embeddings.append(q_emb)
        time.sleep(0.5)
    query_emb_matrix = np.array(query_embeddings)
    
    # 4. Evaluation Loop
    methods = ["tfidf_baseline", "bm25_sparse", "gemini_dense", "hybrid_rrf"]
    results = {}
    
    print("\n[4/4] Executing benchmark retrieval evaluation across all methods...")
    
    for method in methods:
        recall_at_1 = []
        recall_at_3 = []
        recall_at_5 = []
        reciprocal_ranks = []
        ndcg_at_3 = []
        ndcg_at_5 = []
        
        for q_idx, q in enumerate(BENCHMARK_QUERIES):
            query_str = q["query"]
            rel_ids = q["relevant_ids"]
            
            # TF-IDF Scores
            q_tfidf_vec = tfidf.transform([query_str])
            tfidf_scores = cosine_similarity(q_tfidf_vec, tfidf_matrix)[0]
            tfidf_ranks = sorted(range(len(corpus_texts)), key=lambda i: tfidf_scores[i], reverse=True)
            
            # BM25 Scores
            tok_q = technical_tokenize(query_str)
            bm25_scores = bm25.get_scores(tok_q) if tok_q else np.zeros(len(corpus_texts))
            bm25_ranks = sorted(range(len(corpus_texts)), key=lambda i: bm25_scores[i], reverse=True)
            
            # Gemini Dense Scores (Cosine Similarity)
            q_vec = query_emb_matrix[q_idx].reshape(1, -1)
            dense_scores = cosine_similarity(q_vec, corpus_emb_matrix)[0]
            dense_ranks = sorted(range(len(corpus_texts)), key=lambda i: dense_scores[i], reverse=True)
            
            # Select Ranking based on method
            if method == "tfidf_baseline":
                top_indices = tfidf_ranks[:5]
            elif method == "bm25_sparse":
                top_indices = bm25_ranks[:5]
            elif method == "gemini_dense":
                top_indices = dense_ranks[:5]
            else: # Hybrid RRF (k=60)
                k_rrf = 60
                combined_scores = {}
                for idx in range(len(corpus_texts)):
                    r_dense = dense_ranks.index(idx) + 1
                    r_bm25 = bm25_ranks.index(idx) + 1
                    combined_scores[idx] = (1.0 / (k_rrf + r_dense)) + (1.0 / (k_rrf + r_bm25))
                top_indices = sorted(combined_scores.keys(), key=lambda i: combined_scores[i], reverse=True)[:5]
                
            retrieved_chunk_ids = [corpus_ids[i] for i in top_indices]
            
            # Metric Computations
            hit_at_1 = any(doc in rel_ids for doc in retrieved_chunk_ids[:1])
            hit_at_3 = any(doc in rel_ids for doc in retrieved_chunk_ids[:3])
            hit_at_5 = any(doc in rel_ids for doc in retrieved_chunk_ids[:5])
            
            recall_at_1.append(1.0 if hit_at_1 else 0.0)
            recall_at_3.append(1.0 if hit_at_3 else 0.0)
            recall_at_5.append(1.0 if hit_at_5 else 0.0)
            
            # MRR
            rr = 0.0
            for rank_idx, doc_id in enumerate(retrieved_chunk_ids):
                if doc_id in rel_ids:
                    rr = 1.0 / (rank_idx + 1)
                    break
            reciprocal_ranks.append(rr)
            
            # nDCG
            ndcg_at_3.append(calculate_ndcg(retrieved_chunk_ids, rel_ids, k=3))
            ndcg_at_5.append(calculate_ndcg(retrieved_chunk_ids, rel_ids, k=5))
            
        results[method] = {
            "Recall@1": round(float(np.mean(recall_at_1)), 4),
            "Recall@3": round(float(np.mean(recall_at_3)), 4),
            "Recall@5": round(float(np.mean(recall_at_5)), 4),
            "MRR": round(float(np.mean(reciprocal_ranks)), 4),
            "nDCG@3": round(float(np.mean(ndcg_at_3)), 4),
            "nDCG@5": round(float(np.mean(ndcg_at_5)), 4),
            "query_count": len(BENCHMARK_QUERIES)
        }
        
        print(f"\nMethod: {method.upper()}")
        print(f"  Recall@1: {results[method]['Recall@1']:.4f} | Recall@3: {results[method]['Recall@3']:.4f} | Recall@5: {results[method]['Recall@5']:.4f}")
        print(f"  MRR:      {results[method]['MRR']:.4f}")
        print(f"  nDCG@3:   {results[method]['nDCG@3']:.4f} | nDCG@5:   {results[method]['nDCG@5']:.4f}")

    # Output JSON file
    os.makedirs("experiments", exist_ok=True)
    out_path = os.path.join("experiments", "results_rag_retrieval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nRAG benchmark completed! Results saved to: {out_path}")
    return results

if __name__ == "__main__":
    evaluate_retrieval_system()
