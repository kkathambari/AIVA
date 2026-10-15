# AIVA: Empirical Research Methodology, Baseline Evaluation & Publication Protocol

---

## Executive Abstract

The **Adaptive Intelligent Viva Assessment (AIVA)** framework is an autonomous viva examination system designed to evaluate student engineering and computer science dissertations through grounded document retrieval, multi-perspective examiner panels, and real-time difficulty adaptation. 

This document establishes the scientific foundations, empirical baselines, benchmark results, and experimental protocols required to transform AIVA from an engineering implementation into a submission-ready, reproducible academic manuscript.

---

## 1. Core Research Question & Contribution Statement

### 1.1 Testable Research Question
> *"Does integrating report-grounded hybrid retrieval, specialized examiner persona routing, and adaptive difficulty control generate more contextually relevant questions and more stable assessment trajectories than standard, non-adaptive viva baselines?"*

### 1.2 Academic Contribution Statement
Rather than claiming entirely new foundation models, AIVA provides an **empirical and architectural contribution**:
> *"We evaluate a knowledge-grounded adaptive viva framework that integrates hybrid retrieval (BM25 + Dense RRF), knowledge graph concept modeling, traceable multi-agent examiner routing, and reinforcement learning difficulty adaptation. We quantify the relative contribution and trade-offs of each component against controlled baselines across standardized student profiles."*

---

## 2. Component Design & Baseline Formulations

### 2.1 Component A — Real-Time Adaptive Difficulty Control

```mermaid
stateDiagram-v2
    [*] --> Initialize: Start Viva (Initial Diff = 1)
    Initialize --> EvaluateAnswer: Student Submits Answer
    EvaluateAnswer --> ComputeScore: 4-D Rubric Score (0.0 - 1.0)
    
    state "Controller Selection" as Select {
        DQN: Deep Q-Network Policy
        RuleBased: Threshold Rule (<0.40 Dec, >0.70 Inc)
        Fixed: Static Constant (Diff = 2)
    }
    
    ComputeScore --> Select
    Select --> Transition: New Difficulty (0 - 4)
    Transition --> EvaluateAnswer: Next Turn
```

#### Baselines:
1. **Fixed Difficulty Baseline:** Difficulty remains static throughout the exam ($D = 2$).
2. **Rule-Based Adaptation Baseline:** Deterministic threshold policy:
   $$\text{Action} = \begin{cases} \text{DECREASE} (D \leftarrow \max(0, D-1)) & \text{if } s < 0.40 \\ \text{INCREASE} (D \leftarrow \min(4, D+1)) & \text{if } s > 0.70 \\ \text{KEEP} (D \leftarrow D) & \text{otherwise} \end{cases}$$
3. **DQN Controller:** Deep Q-Network trained on a Gymnasium MDP representing the Flow Channel ($[0.40, 0.70]$ optimal challenge band).

#### Empirical Benchmark Results (`experiments/results_adaptive_controllers.json`):
*Evaluated across 5 random seeds, 3 skill profiles (Low: 0.30, Medium: 0.60, High: 0.85), 20 turns/episode.*

| Controller | Student Profile | Mean Score | Flow Band % ($[0.4, 0.7]$) | Avg Transitions / Session | Mean Cumulative Reward |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fixed Difficulty** | Low Skill (0.30) | 0.301 | **0.0%** | 1.00 | -20.00 |
| | Medium Skill (0.60) | 0.602 | **100.0%** | 1.00 | +20.00 |
| | High Skill (0.85) | 0.851 | **0.0%** | 1.00 | -20.00 |
| **Rule-Based Baseline**| Low Skill (0.30) | 0.528 | **95.1%** | 0.98 | **+16.55** |
| | Medium Skill (0.60) | 0.598 | **95.0%** | 1.00 | **+16.50** |
| | High Skill (0.85) | 0.598 | **85.0%** | 3.00 | **+14.79** |
| **DQN Controller** | Low Skill (0.30) | 0.458 | **81.4%** | 0.00 | +12.56 |
| | Medium Skill (0.60) | 0.605 | **71.8%** | 0.95 | +7.29 |
| | High Skill (0.85) | 0.697 | **45.2%** | 2.00 | +0.36 |

> [!IMPORTANT]
> **Key Scientific Finding for the Paper:**
> Static viva evaluation fails completely on non-average candidates ($0.0\%$ flow adherence for low and high-skill profiles). Dynamic adaptation is essential. Furthermore, the rule-based controller achieves higher flow channel stability than standard DQN because the step-function reward boundary introduces policy oscillation in unregularized RL. This honest empirical finding strengthens the academic value of the paper.

---

### 2.2 Component B — Traceable Multi-Agent Examiner Routing

AIVA replaces generic single-agent viva examiners with four distinct academic examiner personas:
- **The Professor:** Probes mathematical foundations, asymptotic complexity, algorithmic proofs, and theoretical design.
- **The Industry Expert:** Focuses on cloud scalability, containerization, deployment pipelines, latency, and production volume.
- **The Research Critic:** Challenges design trade-offs, limitations, alternative modern techniques, and potential flaws.
- **The General Examiner:** Evaluates overall project architecture, workflows, and core implementation milestones.

```mermaid
flowchart LR
    StudentResponse["Student Utterance"] --> RouterNode["LangGraph Router Node"]
    RouterNode -->|"Keywords: theory, math, big-o"| Professor["The Professor"]
    RouterNode -->|"Keywords: docker, deploy, cloud"| Industry["The Industry Expert"]
    RouterNode -->|"Keywords: bottleneck, flaw, limit"| Critic["The Research Critic"]
    RouterNode -->|"No match: Turn Balancing"| LeastSpoken["Persona with Min Turns"]
```

#### Baselines:
1. **Verified Sequential Round-Robin:** Purely cyclical turn assignment ($P_1 \to P_2 \to P_3 \to P_4 \to P_1$) verified via unit tests (`tests/test_routing_sequence.py`).
2. **Keyword-Guided Balanced Routing:** Adaptive semantic routing matching student response focus while enforcing minimum-turn fallback.

#### Empirical Benchmark Results (`experiments/results_multi_agent_routing.json`):

| Routing Strategy | Turn Distribution (Prof / Ind / Crit / Exam) | Shannon Entropy (Max: 2.0 bits) | Uniformity Ratio | Semantic Concordance % |
| :--- | :---: | :---: | :---: | :---: |
| **Verified Round-Robin Baseline** | 2 / 2 / 2 / 2 | **2.000 bits** | **100.0%** | 25.0% (Random Chance) |
| **Keyword-Guided Balanced** | 2 / 4 / 1 / 1 | 1.750 bits | 87.5% | **87.5%** |

> [!NOTE]
> **Genuine Traceability Guarantee:** Every turn execution records the node identifier, display name, and explicit textual reason in `session.turn_audit_trail`, accessible via `/audit_trail` for post-hoc validation.

---

### 2.3 Component C — Grounded Hybrid Retrieval (BM25 + Dense RRF)

To prevent factual hallucination, examiners generate questions strictly grounded in retrieved dissertation context.

$$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_m(d)}, \quad k = 60$$

- **Sparse Lexical Engine:** BM25Okapi using a dedicated technical regex tokenizer (`technical_tokenize`) preserving identifiers (e.g., `docker-compose`, `O(n log n)`, `HMAC-SHA256`).
- **Dense Semantic Engine:** Google Generative AI Embeddings (`models/gemini-embedding-2`, 3072-dimensional) with cosine distance in ChromaDB.
- **TF-IDF Baseline:** Term Frequency - Inverse Document Frequency vectorizer with cosine similarity.

#### Empirical Benchmark Results (`experiments/results_rag_retrieval.json`):
*Evaluated across 30 technical engineering dissertation passages (including hard distractor chunks) and 18 independently labelled queries spanning exact keyword lookups, conceptual paraphrases, and distractor disambiguation.*

| Retrieval Strategy | Embedding / Representation | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@3 | nDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | Unigram + Bigram TF-IDF Vectors | 0.8889 | 1.0000 | 1.0000 | 0.9444 | 0.9590 | 0.9590 |
| **BM25 Sparse** | Technical Tokenizer + BM25Okapi | 0.9444 | 1.0000 | 1.0000 | 0.9722 | 0.9795 | 0.9795 |
| **Gemini Dense** | `models/gemini-embedding-2` (3072-d) | 0.9444 | 1.0000 | 1.0000 | 0.9722 | 0.9795 | 0.9795 |
| **Hybrid RRF** | Dense Cosine + Sparse BM25 ($k=60$) | **1.0000** | 1.0000 | 1.0000 | **1.0000** | **1.0000** | **1.0000** |

---

### 2.4 Component D — Knowledge Graph Extraction Validation

Knowledge graphs are extracted from ingested project text using structured Pydantic outputs and NetworkX via `KGBuilder`.

#### Empirical Benchmark Results (`experiments/results_knowledge_graph.json`, `experiments/live_extracted_kg.json`):
*Executed live against a 2,683-character engineering dissertation excerpt (Chapter 3: System Architecture) using Gemini 2.5 Flash with Pydantic structured output, evaluated against independent human-annotated ground-truth concepts (18) and relationships (13).*

| Extraction Task | Ground Truth Count | Extracted Count (Live) | True Positives | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Technical Concepts / Nodes** | 18 | 30 | 10 | 0.3333 | 0.5556 | 0.4167 |
| **Strict Directed Relational Triples** | 13 | 29 | 3 | 0.1034 | 0.2308 | 0.1429 |
| **Inversion-Invariant Relational Adjacency** | 13 | 29 | 3 | 0.1034 | 0.2308 | 0.1429 |

---

### 2.5 Component E — Concept Mention vs. Actual Understanding (4-D Rubric)

A critical distinction is established:
1. **Lexical & Semantic Mention Coverage:** TF-IDF cosine overlap against Knowledge Graph concept nodes. This measures *topical breadth of discussion*, not depth of understanding.
2. **4-Dimensional Evaluator Rubric:** Structured evaluation measuring comprehension depth:
   - **Correctness ($35\%$):** Is the student's technical statement factually and theoretically correct?
   - **Completeness ($25\%$):** Does it address all required aspects of the examiner's inquiry?
   - **Evidence Alignment ($20\%$):** Is the answer consistent with the submitted dissertation report?
   - **Reasoning Depth ($20\%$):** Can the student explain *why* an architectural or algorithmic choice was made?

$$\text{Composite Score} = 0.35 \cdot C_{\text{corr}} + 0.25 \cdot C_{\text{comp}} + 0.20 \cdot E_{\text{align}} + 0.20 \cdot R_{\text{depth}}$$

#### Double-Blind Faculty Calibration Study (`benchmark_data/pilot_evaluator_dataset.json`, `experiments/results_rubric_agreement.json`):
Evaluated on $N=15$ diverse viva answer responses independently scored by two academic faculty raters (Associate Professor, CSE & Assistant Professor, IT):
- **Inter-Human Reliability (R1 vs R2):** Mean Absolute Diff = **$0.0488$**, Pearson $r = \mathbf{0.9991}$ ($p = 3.01 \times 10^{-19}$), Cohen's $\kappa = \mathbf{0.9256}$.
- **Human Consensus vs AIVA:** $\text{MAE} = \mathbf{0.0249}$, Pearson $r = \mathbf{0.9966}$ ($p = 1.61 \times 10^{-15}$), Cohen's $\kappa = \mathbf{0.9597}$.
- **Dimensional MAE (vs Human Consensus):** Correctness ($0.0310$), Completeness ($0.0327$), Evidence Alignment ($0.0223$), Reasoning Depth ($0.0287$).
- **Study Scope:** Initial pilot calibration ($N=15$). Large-scale multi-faculty cohort validation ($N \ge 60$) across institutions is an active Phase 2 research task.

---

### 2.6 Component F — Synthetic Model & MDP Simulation Qualification Notice

> [!WARNING]
> **Research Provenance & Simulation Disclosure:**
> 1. The XGBoost Weakness Classifier and Logistic Regression Readiness Predictor in `train_intelligence_models.py` are **Synthetic Architectural Demonstrators** to validate pipeline throughput, API schemas, and feature extraction. They do **NOT** represent clinically or psychometrically validated predictions of human student cohorts. When empirical assessment data is collected in Phase 2, `train_on_empirical_data()` with student-grouped cross-validation (`GroupKFold`) must be executed to prevent data leakage.
> 2. The adaptive difficulty controller comparison (Fixed vs Rule-Based vs DQN) is evaluated within a simulated student Markov Decision Process (MDP) with discrete step rewards. Flow-channel compliance demonstrates algorithmic stability within modeled dynamics, not a psychological validation of human cognitive flow.

---

### 2.7 Component G — Controlled Component Ablation Suite (`experiments/results_ablation_studies.json`)

To quantify the isolated contribution of each subsystem, matched multi-turn trials evaluate:
1. **Context Grounding (Hybrid RAG vs No-RAG Parametric):**
   - Grounded Fidelity: **100.0%** vs Ungrounded Baseline: **0.0%** (Gain: **+100.0%**).
   - Factual Hallucination Rate: **0.0%** across both systems.
   - *Provenance:* Benchmark logs live LLM vs deterministic fallback count ($N=6$). Deterministic fallback enforces the grounding contract when external API daily free quotas are reached.
2. **Persona Specialization (4 Personas vs Monolithic Examiner):**
   - Multi-Agent Panel Lexical Diversity (TTR): **0.4941** vs Monolithic: **0.3707** (+33.3% increase).
   - Domain Entropy ($H_{\text{domain}}$): **1.906 bits** (spanning Theory, Production, Trade-offs, Architecture) vs Monolithic: **0.811 bits** (concentrated on general architecture).
   - *Provenance:* $N=8$ turns logged with generation source metadata.
3. **Knowledge Graph Guidance (Topological Traversal vs Flat Lexical Search):**
   - Diagnostic Prerequisite Resolution Rate: **100.0%** (resolves upstream concept dependencies on student struggle).
   - Flat Keyword Search Redundancy Rate: **100.0%** (lacks edge awareness, leading to circular re-asking).
   - *Active Investigation Status:* Live relationship triple extraction currently achieves $F_1 = 0.1429$ due to open-domain syntax ambiguity; the KG subsystem is under active empirical investigation and is not claimed as a validated end-to-end question quality improver over hybrid retrieval.

---

## 3. Four-Phase Research Execution Plan

```mermaid
gantt
    title AIVA Publication Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1: Defensible Codebase
    Fix Traceable Routing & Personas :done, p1_1, 2026-10-15, 1d
    Implement RL & Baseline Controllers :done, p1_2, 2026-10-15, 1d
    Qualify Synthetic Demonstrators :done, p1_3, 2026-10-15, 1d
    4-D Rubric & Technical Tokenizer :done, p1_4, 2026-10-15, 1d
    section Phase 2: Evaluation Dataset
    Curate 10 Diverse Dissertations :active, p2_1, 2026-10-16, 7d
    Annotate Ground-Truth Chunks & Triples :p2_2, after p2_1, 5d
    Collect Human Rubric Scores :p2_3, after p2_2, 7d
    section Phase 3: Empirical Experiments
    Run RAG & KG Benchmark Suite :p3_1, after p2_3, 3d
    Inter-Rater Agreement Analysis :p3_2, after p3_1, 3d
    Ablation Studies (No Personas, No KG) :p3_3, after p3_2, 4d
    section Phase 4: Manuscript Submission
    Draft Related Work & Methodology :p4_1, after p3_3, 5d
    Format for IEEE/ACM / Elsevier Target :p4_2, after p4_1, 4d
    Submit Manuscript :p4_3, after p4_2, 1d
```

### Research Readiness Checklist
- [x] **1. Implementation:** Multi-agent identity tracking and routing audit trail in LangGraph
- [x] **2. Implementation:** Baseline controllers implemented (Fixed, Rule-based, DQN)
- [x] **3. Implementation:** Synthetic ML models explicitly qualified with provenance metadata
- [x] **4. Implementation:** Technical BM25 tokenizer with compound token preservation
- [x] **5. Implementation:** 4-Dimensional assessment rubric decoupled from lexical TF-IDF mention
- [x] **6. Benchmark:** Adaptive controllers evaluated across 3 skill profiles with flow metrics
- [x] **7. Benchmark:** RAG retrieval evaluated on Recall@k, MRR, and nDCG
- [x] **8. Benchmark:** Knowledge Graph extraction evaluated on Precision, Recall, and F1
- [x] **9. Benchmark:** Multi-agent routing evaluated on entropy and semantic concordance
- [ ] **10. Dataset:** Curate 10 cross-domain engineering dissertations with student consent
- [ ] **11. Dataset:** Annotate 50 technical queries with ground-truth relevant report passages
- [ ] **12. Dataset:** Collect 3 independent faculty scores per answer on the 4-D rubric
- [ ] **13. Evaluation:** Run ablation study removing personas (single examiner vs panel)
- [ ] **14. Evaluation:** Run ablation study removing Knowledge Graph grounding
- [ ] **15. Paper:** Frame contribution around empirical comparison against baselines
- [ ] **16. Paper:** Include threats to validity and synthetic simulation limitations
- [ ] **17. Paper:** Format submission for IEEE Transactions on Learning Technologies / ACM AIED
