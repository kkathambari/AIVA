# AIVA: Adaptive Multi-Agent Viva Assessment Framework

AIVA is an intelligent, research-oriented oral examination (viva) platform that evaluates students based on their project reports. Instead of relying on static question pools, AIVA dynamically parses documents, extracts interactive Knowledge Graphs, indexes text for Hybrid Retrieval (RAG), and uses a rule-based Multi-Agent AI panel alongside Reinforcement Learning to conduct a highly adaptive conversational examination.

---

## 🌟 Core Architecture & Features

### 1. Hybrid Retrieval Engine (RAG)
*   **Dense Search:** Uses `ChromaDB` with `models/gemini-embedding-2` for semantic document chunk retrieval.
*   **Sparse Search:** Uses `BM25` (`rank-bm25`) to catch exact technical keywords.
*   **Reciprocal Rank Fusion (RRF):** Fuses dense and sparse rankings with a rank constant $k = 60$ to deliver highly context-aware search results.

### 2. Entity-Relation Knowledge Graph
*   **Extraction:** Leverages Google's Gemini to identify project concepts, tools, and relations from uploaded PDF/DOCX files.
*   **Visualization:** Renders an interactive 3D physics graph using `NetworkX` and `PyVis` outputting directly to the UI.
*   **Dissertation Metrics:** Records graph density, node count, edge count, and extraction speed in `graph_metadata.json` for analysis.

### 3. Adaptive Difficulty Control (Reinforcement Learning)
*   **Gymnasium Environment:** Implements a custom Gymnasium RL environment where the examiner adjusts difficulty (Levels 0 to 4) based on student response scores.
*   **DQN Model:** Pre-trained using a Deep Q-Network (DQN) via `stable-baselines3` for 15,000 steps to make instant, live difficulty adjustments.

### 4. Explainable Multi-Agent Routing (LangGraph)
Routes topics dynamically to specialized examiner personas based on student answer keywords:
*   **Professor:** Theoretical concepts, algorithms, and complexity.
*   **Industry Expert:** Deployment, docker, scaling, databases, and CI/CD.
*   **Critic:** Drawbacks, bottlenecks, and system limitations.
*   **Examiner:** General project architecture.
*   *Balances turns automatically when no keywords match.*

### 5. Analytics & Weakness Detection
*   **Concept Coverage:** Uses Scikit-Learn's `TfidfVectorizer` and Cosine Similarity to measure the semantic coverage of the student's answers against each extracted Knowledge Graph concept node.
*   **XGBoost Weakness Classifier:** Flags specific concept gaps where performance features predict a high likelihood of weakness.
*   **Logistic Regression Pass Predictor:** Calculates the overall passing probability based on answer length, scores, and coverage.

### 6. Multimodal Voice Fallback
*   **STT & TTS:** Speech recognition and audio synthesis.
*   **Gemini Fallback:** If browser audio formats (WebM/OGG) cause native parser errors, the backend automatically redirects audio blobs to the Gemini Multimodal Audio API for transcription.

---

## 📁 Repository Structure

```
AIVA/
├── AIVA_backend/               # FastAPI Microservice (Python)
│   ├── core/                   # Core engines (RAG, KG, RL, Multi-Agent, Voice, Analytics)
│   ├── models/                 # Pre-trained ML models (.zip, .json, .joblib)
│   ├── main.py                 # API router and session manager (Binds to Port 8001)
│   ├── requirements.txt        # Python dependency list
│   └── verify_backend.py       # Automated test suite
│
├── AIVA_frontend/              # Next.js Dashboard (React 19 & Next.js 16)
│   ├── src/app/                # App Router files (Tabs: Learn, Viva, Analytics, Graph)
│   ├── src/components/         # Glassmorphism panels, SVG charts, upload cards
│   ├── src/lib/api.ts          # Axios API client calling port 8001
│   ├── package.json            # Node dependencies and scripts
│   └── tsconfig.json           # TypeScript configuration
│
└── .gitignore                  # Clean repository filters
```

---

## 🛠️ Installation & Setup

### 1. Prerequisite: API Key Configuration
Create a `.env` file inside `AIVA_backend/` and configure your API key:
```env
GEMINI_API_KEY="your-google-gemini-api-key"
```

---

### 2. Backend Setup (FastAPI)
Navigate to the backend directory, initialize your virtual environment, install dependencies, and run:
```bash
cd AIVA_backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Run the FastAPI server (Runs on http://127.0.0.1:8001)
python -m uvicorn main:app --port 8001 --reload
```

---

### 3. Frontend Setup (Next.js)
Navigate to the frontend directory, install npm packages, and run the development server:
```bash
cd AIVA_frontend

# Install node modules
npm install

# Run dev server (Runs on http://localhost:3000)
npm run dev
```

---

## 🧪 Automated Testing & Empirical Research Benchmarks

### 1. Deterministic Unit & Integration Test Suites (Offline, Zero-Quota)
Runs the offline test suite with mocked external dependencies to verify session state isolation, HTTP viva lifecycle, and UCS edge cases:
```bash
cd AIVA_backend

# 1. Deterministic End-to-End HTTP Integration & Multi-Session Isolation Test
python tests/test_deterministic_end_to_end.py

# 2. UCS Knowledge-Gap Engine Unit Tests & Topological Edge Cases
python tests/test_ucs_selection.py

# 3. Multi-Agent Persona Sequential Alternation Test
python tests/test_routing_sequence.py

# 4. Automated Research Artifact & Manuscript Consistency Validation Gate
python experiments/validate_experiment_artifacts.py
```

### 2. Backend Integration Verification
Runs the live model smoke test checking RAG, RL, analytics, and multi-agent routing:
```bash
cd AIVA_backend
python verify_backend.py
```

### 3. Research Benchmark Experiments
To reproduce all empirical tables and baseline comparisons published in the research manuscript:
```bash
cd AIVA_backend

# 1. Primary Research Benchmark (E1): Question Selection Strategy Comparison (UCS vs Fixed vs Random vs Difficulty-Only vs Greedy Gap)
python experiments/eval_question_selection.py

# 2. Adaptive Difficulty Benchmark (E4): (DQN vs Rule-Based vs Fixed across 3 skill profiles)
python experiments/eval_adaptive_engine.py

# 3. Multi-Agent Routing Benchmark (E3): (Keyword-guided vs Round-Robin baseline)
python experiments/eval_multi_agent_routing.py

# 4. Hybrid RAG Benchmark (E5): (Recall@k, MRR, nDCG comparing BM25, Dense, and RRF)
python experiments/eval_rag_retrieval.py

# 5. Knowledge Graph Factuality Benchmark (E2): (Precision, Recall, F1 against ground truth)
python experiments/eval_knowledge_graph.py

# 6. Double-Blind Faculty Rubric Calibration (E6): (Inter-rater agreement & AIVA calibration)
python experiments/eval_rubric_agreement.py

# 7. Component Ablation Studies: (Grounding fidelity, Persona TTR diversity, KG paths)
python experiments/eval_ablations.py
```

---

## 📚 Academic Publication & Documentation

- **Full Research Paper Manuscript:** [`paper/AIVA_Research_Paper_Manuscript.md`](paper/AIVA_Research_Paper_Manuscript.md)
- **Empirical Methodology & Evaluation Protocol:** [`docs/RESEARCH_METHODOLOGY_AND_EVALUATION.md`](docs/RESEARCH_METHODOLOGY_AND_EVALUATION.md)
- **Standardized Evaluation Protocol Specification:** [`benchmark_data/evaluation_protocol.json`](benchmark_data/evaluation_protocol.json)
- **Master Project Report (All Milestones):** [`AIVA_MASTER_PROJECT_REPORT_ALL_PHASES.md`](AIVA_MASTER_PROJECT_REPORT_ALL_PHASES.md)

