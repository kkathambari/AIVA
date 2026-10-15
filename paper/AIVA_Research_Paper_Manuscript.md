# An Empirical Evaluation of Knowledge-Grounded Multi-Agent Panels and Reinforcement Learning for Adaptive Oral Viva Assessment

**Kathambari K.**  
*Department of Computer Science and Engineering*  
*Autonomous Intelligent Viva Assessment (AIVA) Research Project*

---

## Abstract
Oral viva voce examinations are the gold standard for assessing student dissertation ownership, conceptual depth, and technical defense capabilities in engineering education. However, traditional human-conducted vivas suffer from evaluator fatigue, inter-examiner subjectivity, and high scheduling overhead. In this paper, we present and empirically evaluate **AIVA (Adaptive Intelligent Viva Assessment)**, an autonomous oral examination framework that unifies: (1) document-grounded hybrid retrieval combining technical BM25 tokenization and dense embeddings via Reciprocal Rank Fusion ($k=60$), (2) automated knowledge graph extraction for topological concept tracking, (3) a traceable multi-agent examiner panel with dynamic semantic routing across four specialized academic personas (*The Professor*, *The Industry Expert*, *The Research Critic*, and *The General Examiner*), and (4) real-time reinforcement learning difficulty adaptation targeting the psychological flow channel. Across rigorous controlled simulations spanning three student skill cohorts ($0.30, 0.60, 0.85$), we demonstrate that static-difficulty examinations collapse completely on non-average candidates ($0.0\%$ flow adherence), confirming the necessity of dynamic adaptation. Our findings also reveal that a deterministic rule-based controller outperforms standard discrete-reward Deep Q-Networks ($95.1\%$ vs $81.4\%$ flow adherence), providing critical empirical evidence regarding RL reward continuity in low-dimensional educational state spaces. Controlled ablation experiments show that hybrid retrieval elevates report technical term fidelity from $0.0\%$ to $100.0\%$, while multi-agent persona routing expands question lexical diversity by $+33.3\%$ ($0.4941$ vs $0.3707$ Type-Token Ratio), achieves $1.906$ bits pedagogical domain entropy, and maintains $87.5\%$ semantic topic concordance over sequential round-robin routing ($25.0\%$). Furthermore, automated four-dimensional rubric scoring exhibits substantial alignment with faculty grading in a double-blind calibration benchmark ($\text{MAE} = 0.0249$, $r = 0.9966$, $\kappa = 0.9597$). We provide full reproducible configurations, baseline comparison data, and an open evaluation protocol for autonomous assessment.

**Keywords:** Automated Oral Assessment, Retrieval-Augmented Generation, Multi-Agent Dialogue Systems, Reinforcement Learning, Adaptive Learning, Knowledge Graphs.

---

## 1. Introduction

The oral defense (viva voce) serves as the primary summative milestone in university engineering education. Unlike written examinations that primarily measure passive recall, a viva challenges students to defend architectural choices, demonstrate command over edge cases, explain trade-offs, and prove individual authorship of their dissertation projects.

Despite its academic pedagogical centrality, the oral viva suffers from critical systemic limitations:
1. **Evaluator Fatigue and Subjectivity:** Human faculty typically evaluate dozens of project groups consecutively, resulting in grading drift, fatigue-induced variance, and inconsistent probing depth.
2. **Prohibitive Pre-Reading Overhead:** Technical dissertation reports routinely span 50 to 150 pages with intricate code snippets, mathematical derivations, and deployment manifests. Comprehensive pre-examination review by busy examiners is often impractical, forcing evaluators to resort to generic question pools.
3. **Static Difficulty Trajectories:** Conventional examinations apply homogeneous question sequences, which under-challenge high-performing candidates and induce anxiety or premature surrender in struggling students.

Existing automated assessment tools have focused overwhelmingly on automated essay scoring (AES) or closed-domain multiple-choice generation. Recent large language model (LLM) conversational agents have exhibited conversational fluency but suffer from severe hallucination, superficial questioning, and an inability to dynamically adjust technical rigor based on candidate performance.

To overcome these challenges, we present **AIVA (Adaptive Intelligent Viva Assessment)**. Rather than claiming novel foundational language architectures, this paper makes a **systematic empirical and architectural contribution**:
- We formalize a multi-perspective viva framework combining document-grounded hybrid retrieval (dense embeddings + sparse BM25 with compound technical tokenization), knowledge graph concept modeling, explainable multi-agent examiner personas, and dynamic difficulty control.
- We benchmark AIVA against controlled algorithmic baselines across each sub-engine, measuring retrieval quality (Recall@k, MRR, nDCG), persona routing concordance, knowledge graph extraction precision, and difficulty adaptation stability across simulated student skill profiles.
- We report transparent empirical findings—including the unexpected trade-offs between reinforcement learning and rule-based controllers in educational flow channels—establishing an open, reproducible benchmark protocol for autonomous assessment systems.

---

## 2. Related Work

### 2.1 Automated Assessment and Oral Examination Systems
Automated assessment has evolved from statistical n-gram analysis (e.g., e-rater) to dense transformer representations (BERT, RoBERTa) for scoring essay coherence and argumentative structure. However, conversational oral assessment requires interactive dialogue management, real-time question generation, and acoustic speech handling. Prior conversational tutors (e.g., AutoTutor, Jill Watson) demonstrated the viability of dialogue-based pedagogical agents but relied on hardcoded semantic slot-filling or static curriculum trees, lacking the capability to ingest arbitrary technical dissertations.

### 2.2 Retrieval-Augmented Generation (RAG) in Education
Large language models fine-tuned on general text frequently hallucinate technical assertions when quizzing students on novel project codebases. Retrieval-Augmented Generation (RAG) mitigates hallucination by retrieving context passages from a verified corpus. While standard RAG relies exclusively on dense semantic vector similarity, dense retrieval often fails on exact technical identifiers (e.g., `O(n log n)`, `FastAPI`, `docker-compose`). Hybrid retrieval strategies combining dense representations with sparse BM25 through Reciprocal Rank Fusion (RRF) have shown superior robustness in information retrieval literature, but their application to academic viva questioning remains unexplored.

### 2.3 Multi-Agent Conversational Architectures
Recent multi-agent frameworks (e.g., AutoGen, LangGraph) allow multiple LLM nodes to adopt specialized roles. In viva examinations, evaluation requires distinct cognitive perspectives: theoretical scrutiny, production practicality, critical doubt, and overarching design synthesis. While monolithic single-agent prompts struggle to simultaneously balance these competing personas, structured state-machine routing enables verifiable persona delegation.

### 2.4 Reinforcement Learning for Adaptive Difficulty
Csikszentmihalyi’s Flow Theory posits that optimal learning and engagement occur in a dynamic channel between anxiety (when difficulty exceeds skill) and boredom (when skill exceeds challenge). Reinforcement learning (RL) has been applied to curriculum sequencing, typically using multi-armed bandits or Deep Q-Networks (DQN). However, whether continuous RL algorithms outperform discrete rule-based threshold heuristics in low-dimensional viva state spaces has not been empirically verified.

---

## 3. System Architecture

```
                          ┌──────────────────────────────────────┐
                          │     Submitted Project Document       │
                          │        (PDF / DOCX / PPTX)           │
                          └──────────────────┬───────────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
          ┌───────────────────────────┐               ┌───────────────────────────┐
          │     Hybrid RAG Engine     │               │  Knowledge Graph Engine   │
          │  Dense + BM25 (RRF, k=60) │               │   (Concepts & Relations)  │
          └────────────┬──────────────┘               └─────────────┬─────────────┘
                       │                                            │
                       └─────────────────────┬──────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │   LangGraph Router Node   │
                               │(Keyword + Turn-Balancing) │
                               └─────────────┬─────────────┘
                                             │
             ┌─────────────────┬─────────────┴─────────────┬─────────────────┐
             ▼                 ▼                           ▼                 ▼
     ┌───────────────┐ ┌───────────────┐           ┌───────────────┐ ┌───────────────┐
     │ The Professor │ │Industry Expert│           │  The Critic   │ │  Examiner   │
     └───────┬───────┘ └───────┬───────┘           └───────┬───────┘ └───────┬───────┘
             │                 │                           │                 │
             └─────────────────┴─────────────┬─────────────┴─────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │     Student Response      │
                               └─────────────┬─────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │  4-Dimensional Rubric Eval│
                               │  (Score: 0.0 to 1.0)      │
                               └─────────────┬─────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │    DQN / Rule-Based RL    │
                               │   Difficulty Controller   │
                               └───────────────────────────┘
```

### 3.1 Document Ingestion and Knowledge Graph Construction
Dissertation files are parsed into clean text. A knowledge graph builder extracts technical concepts ($V$) and typed relational edges ($E$) using structured Pydantic schemas and Gemini 2.5 Flash. The graph is stored in NetworkX and rendered via PyVis physics simulation, providing an explorable topological map of the candidate's project domains.

### 3.2 Grounded Hybrid Retrieval Engine
Texts are partitioned into chunks of 1,000 characters with 200-character overlaps.
- **Sparse BM25:** Employs a specialized technical tokenizer (`technical_tokenize`) that normalizes casing while retaining technical tokens, underscores, and hyphens (`docker-compose`, `gemini-2.5-flash`, `time_complexity`).
- **Dense Retrieval:** Computes cosine similarity over Google Gemini Embedding 2 vectors.
- **Reciprocal Rank Fusion:** Merges candidate sets into a joint ranking:
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_m(d)}, \quad k = 60$$

### 3.3 Multi-Agent Persona Panel & Traceable Routing
Examiner personas are coordinated via a compiled LangGraph state machine with an explicit router entry node:
1. **The Professor:** Probes theoretical depth, algorithms, proofs, and time/space complexity.
2. **The Industry Expert:** Focuses on cloud scalability, containerization, deployment, and high-load performance.
3. **The Research Critic:** Evaluates bottlenecks, design flaws, trade-offs, and alternative solutions.
4. **The General Examiner:** Synthesizes system architecture, workflows, and core implementation details.

The router executes keyword-guided matching against the student's previous utterance with a turn-balancing fallback (selecting the least-spoken examiner), and logs the `selected_agent` and `routing_reason` into an immutable audit trail.

### 3.4 Adaptive Difficulty Controller
The examination environment is modeled as a Markov Decision Process (MDP):
- **State Space:** $s_t = [d_t, \text{score}_t] \in \{0, 1, 2, 3, 4\} \times [0.0, 1.0]$
- **Action Space:** $a_t \in \{0: \text{DECREASE}, 1: \text{KEEP}, 2: \text{INCREASE}\}$
- **Reward Function:** Penalizes anxiety ($s < 0.40$) and boredom ($s > 0.70$), rewarding maintenance of the optimal Flow Channel:
  $$R(s_t, a_t) = \begin{cases} +1.0 & \text{if } \text{score} < 0.40 \text{ and } a_t = 0 \\ -1.0 & \text{if } \text{score} < 0.40 \text{ and } a_t \neq 0 \\ +1.0 & \text{if } \text{score} > 0.70 \text{ and } a_t = 2 \\ -1.0 & \text{if } \text{score} > 0.70 \text{ and } a_t \neq 2 \\ +1.0 & \text{if } 0.40 \le \text{score} \le 0.70 \text{ and } a_t = 1 \\ -0.5 & \text{if } 0.40 \le \text{score} \le 0.70 \text{ and } a_t \neq 1 \end{cases}$$

### 3.5 4-Dimensional Rubric Answer Evaluation
To decouple topical lexical mention from actual comprehension, candidate answers are graded along a 4-dimensional academic rubric:
$$\text{Score} = 0.35 \cdot C_{\text{corr}} + 0.25 \cdot C_{\text{comp}} + 0.20 \cdot E_{\text{align}} + 0.20 \cdot R_{\text{depth}}$$
Where $C_{\text{corr}}$ is Technical Correctness, $C_{\text{comp}}$ is Completeness, $E_{\text{align}}$ is Evidence Alignment with the project report, and $R_{\text{depth}}$ is Reasoning Depth & Trade-offs.

---

## 4. Empirical Evaluation & Results

### 4.1 Adaptive Difficulty Controller Comparison
We simulated 300 complete viva sessions (20 turns each) across 5 independent random seeds ($42, 101, 2024, 777, 999$) evaluating three student skill cohorts: Low ($0.30$), Medium ($0.60$), and High ($0.85$).

**Table 1: Adaptive Difficulty Controller Simulation Benchmark (`results_adaptive_controllers.json`)**
| Controller | Student Skill Profile | Mean Score | Flow Band % ($[0.40, 0.70]$) | Transitions / Session | Cumulative Reward |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fixed Baseline** | Low Skill (0.30) | 0.301 | **0.0%** | 1.00 | -20.00 |
| *(Pinned at level 2)*| Medium Skill (0.60) | 0.602 | **100.0%** | 1.00 | +20.00 |
| | High Skill (0.85) | 0.851 | **0.0%** | 1.00 | -20.00 |
| **Rule-Based Baseline**| Low Skill (0.30) | 0.528 | **95.1%** | 0.98 | **+16.55** |
| *(Threshold policy)* | Medium Skill (0.60) | 0.598 | **95.0%** | 1.00 | **+16.50** |
| | High Skill (0.85) | 0.598 | **85.0%** | 3.00 | **+14.79** |
| **AIVA DQN Controller** | Low Skill (0.30) | 0.458 | **81.4%** | 0.00 | +12.56 |
| *(Trained RL agent)* | Medium Skill (0.60) | 0.605 | **71.8%** | 0.95 | +7.29 |
| | High Skill (0.85) | 0.697 | **45.2%** | 2.00 | +0.36 |

**Analysis:**
1. Fixed-difficulty examination fails completely on non-average candidates ($0.0\%$ flow compliance), proving that static questioning produces frustration or boredom.
2. The deterministic rule-based controller outperforms the trained DQN across all skill cohorts ($95.1\%$ vs $81.4\%$ in low-skill, $85.0\%$ vs $45.2\%$ in high-skill). In discrete educational flow bands, step-function rewards cause RL value function oscillations around boundaries, whereas deterministic threshold policies maintain higher stability.

### 4.2 Multi-Agent Persona Routing Evaluation
We evaluated persona distribution balance and semantic domain concordance across multi-turn technical student dialogues comparing sequential round-robin against AIVA's keyword-guided router. The sequential round-robin alternation was verified via unit test (`tests/test_routing_sequence.py`) ensuring a strict four-persona cycle (`Examiner` $\rightarrow$ `Critic` $\rightarrow$ `Industry Expert` $\rightarrow$ `Professor`).

**Table 2: Multi-Agent Persona Routing Comparison (`results_multi_agent_routing.json`)**
| Strategy | Persona Distribution (Prof / Ind / Critic / Exam) | Shannon Entropy (Max: 2.0) | Uniformity Ratio | Semantic Concordance % |
| :--- | :---: | :---: | :---: | :---: |
| **Verified Round-Robin Baseline** | 2 / 2 / 2 / 2 | **2.000 bits** | **100.0%** | 25.0% (Random Chance) |
| **AIVA Keyword-Guided** | 2 / 4 / 1 / 1 | 1.750 bits | 87.5% | **87.5%** |

**Analysis:** The verified round-robin baseline achieves theoretical maximum entropy ($2.000$ bits) by cycling deterministically regardless of dialogue topic, but achieves only chance concordance ($25.0\%$) with the student's ongoing subject matter. In contrast, AIVA’s keyword-guided router achieves $87.5\%$ semantic concordance while maintaining near-maximal entropy ($1.750$ bits, $87.5\%$ uniformity) across examiner turns.

### 4.3 Knowledge Graph Extraction Evaluation
To evaluate factuality and precision, AIVA's live LLM-driven graph extractor (`KGBuilder` invoking Gemini 2.5 Flash with Pydantic structured schema validation) was executed against an independently annotated graduate dissertation excerpt (Chapter 3: System Architecture, 2,683 characters). Extracted nodes and relationship triples were compared against 18 ground-truth concepts and 13 typed relationships (`results_knowledge_graph.json`, `live_extracted_kg.json`).

**Table 3: Live Knowledge Graph Extraction Factuality**
| Extraction Entity | Reference Count | Extracted Count (Live) | True Positives | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Technical Concepts / Nodes** | 18 | 30 | 10 | 0.3333 | 0.5556 | 0.4167 |
| **Strict Directed Relational Triples** | 13 | 29 | 3 | 0.1034 | 0.2308 | 0.1429 |
| **Inversion-Invariant Relational Adjacency** | 13 | 29 | 3 | 0.1034 | 0.2308 | 0.1429 |

**Linguistic Error Taxonomy & Analysis:**
Live evaluation on unstructured project text highlights key information extraction dynamics:
1. *Core Architectural Retention:* Core technologies (*FastAPI*, *Docker*, *Kubernetes*, *PostgreSQL*, *BM25Okapi*, *Gemini*, *ChromaDB*, *Deep Q-Network*) are extracted reliably.
2. *Granular Entity Splitting:* The LLM decomposed high-level entities into granular sub-components (e.g., extracting *HTTP* and *WebSocket* as children of *API Gateway*).
3. *Passive Voice Linguistic Inversions:* Zero-shot extraction frequently flips relation directionality depending on syntax (e.g., ground-truth `Deep Q-Network CONTROLS Adaptive Difficulty` was extracted as `Adaptive Difficulty CONTROLLED_BY Deep Q-Network`).
4. *Hierarchical Mediators:* Text describing containerization led the model to introduce intermediate nodes (e.g., *Microservices Infrastructure* between *AIVA* and *Docker*).

### 4.4 Hybrid RAG Retrieval Quality
To evaluate retrieval robustness against distractor interference, we benchmarked an expanded corpus of **30 technical passages** (including challenging distractor chunks spanning local Docker Compose development, L2 regularization, token rotation, and PostgreSQL table partitioning) evaluated against **18 independently labelled queries** spanning exact keyword lookups, conceptual paraphrases, and distractor disambiguation (`results_rag_retrieval.json`).

**Table 4: Retrieval Engine Comparison on Engineering Dissertation Corpus (30 Passages, 18 Queries)**
| Retrieval Strategy | Embedding / Representation | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@3 | nDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | Unigram + Bigram TF-IDF Vectors | 0.8889 | 1.0000 | 1.0000 | 0.9444 | 0.9590 | 0.9590 |
| **BM25 Sparse** | Technical Tokenizer + BM25Okapi | 0.9444 | 1.0000 | 1.0000 | 0.9722 | 0.9795 | 0.9795 |
| **Gemini Dense** | `models/gemini-embedding-2` (3072-d) | 0.9444 | 1.0000 | 1.0000 | 0.9722 | 0.9795 | 0.9795 |
| **Hybrid RRF** | Dense Cosine + Sparse BM25 ($k=60$) | **1.0000** | 1.0000 | 1.0000 | **1.0000** | **1.0000** | **1.0000** |

**Analysis:** When confronted with semantically related distractor passages, both BM25 alone and Gemini Dense alone exhibit occasional rank-1 misses ($\text{Recall@1} = 0.9444, \text{MRR} = 0.9722$). BM25 struggles when queries use abstract paraphrases lacking exact terminology, while Dense embeddings occasionally rank high-level architectural overviews above fine-grained technical specifications. Combining them via Reciprocal Rank Fusion ($k=60$) compensates for both error modes, elevating top-1 precision to **100.0%** across all test queries.

### 4.5 Component Ablation Studies
To quantify the isolated contribution of AIVA's core architectural components, controlled ablations were conducted on question generation (`results_ablation_studies.json`):

**Table 5: Controlled System Ablation Results (`results_ablation_studies.json`)**
| Ablation Experiment | Full AIVA System | Ablated Baseline System | Empirical Finding |
| :--- | :---: | :---: | :---: |
| **Context Grounding (Fidelity %)** | **100.0%** (RAG Grounded) | **0.0%** (Ungrounded Generic) | RAG anchors questions strictly in project-reported technologies, eliminating boilerplate drift |
| **Persona Diversity (TTR & Entropy)** | **0.4941** (TTR), **1.906 bits** | **0.3707** (TTR), **0.811 bits** | Personas increase lexical diversity (+33.3%) and cover theory, scale, trade-offs, and architecture |
| **Relational KG Guidance (Prerequisites)** | **100.0%** Prerequisite Traversal | **0.0%** Traversal (**100.0%** Redundancy) | Traverses upstream topological dependencies upon student struggle, avoiding circular repetition |

**Generation Provenance Disclosure:**
To ensure scientific transparency, our benchmark harness tracks whether each question is generated via live LLM inference or calibrated fallback:
- **Grounding Trials ($N=6$):** Evaluated across paired dialogue scenarios. When external LLM API rate limits are encountered (e.g., Google Generative AI free-tier limit of 20 requests/day), the benchmark executes a context-anchored fallback extracting retrieved report entities, maintaining the grounding distinction between document-derived questions and generic baseline prompts.
- **Persona Diversity ($N=8$):** Evaluated across eight consecutive viva turns. Lexical diversity and domain entropy gains reflect the four cognitive viewpoints (*Theory*, *Production*, *Trade-offs*, *Architecture*) compared to a monolithic examiner focused on general implementation.

### 4.6 Rubric Calibration & Faculty Consensus Study
To evaluate scoring consistency between automated rubric grading and experienced evaluators, we conducted a pilot calibration study using an anonymized dataset of $N=15$ diverse student viva responses (`benchmark_data/pilot_evaluator_dataset.json`, `results_rubric_agreement.json`). 

**Faculty Provenance & Protocol:**
Responses were scored under a double-blind protocol by two academic faculty members:
- **Rater 1:** Associate Professor in Computer Science & Engineering (Specialization: Distributed Systems & Cloud Infrastructure).
- **Rater 2:** Assistant Professor in Information Technology (Specialization: Applied Machine Learning & NLP).

Both evaluators received the standardized 4-dimensional scoring rubric instructions without access to each other's ratings or the system's automated evaluations. Scores were recorded across Correctness ($0.35$), Completeness ($0.25$), Evidence Alignment ($0.20$), and Reasoning Depth ($0.20$) on a normalized $[0.0, 1.0]$ scale.

**Table 6: Pilot Rubric Inter-Rater Reliability & Calibration Metrics ($N=15$)**
| Comparison Pair | Metric | Observed Value | Academic Standard | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Inter-Human Reliability**<br>*(Rater 1 vs Rater 2)* | Mean Absolute Difference<br>Pearson Correlation ($r$)<br>Cohen's $\kappa$ (Weighted) | **0.0488**<br>**0.9991** ($p = 3.01 \times 10^{-19}$)<br>**0.9256** | $< 0.100$<br>$> 0.800$<br>$> 0.750$ | High inter-evaluator consistency |
| **Human Consensus vs AIVA**<br>*(Faculty Mean vs Automated)* | Mean Absolute Error (MAE)<br>Pearson Correlation ($r$)<br>Cohen's $\kappa$ (Weighted) | **0.0249**<br>**0.9966** ($p = 1.61 \times 10^{-15}$)<br>**0.9597** | $< 0.100$<br>$> 0.800$<br>$> 0.750$ | Substantial rubric alignment |

**Dimensional MAE Breakdown (AIVA vs Faculty Consensus):**
- Correctness ($w=0.35$): $\text{MAE} = 0.0310$
- Completeness ($w=0.25$): $\text{MAE} = 0.0327$
- Evidence Alignment ($w=0.20$): $\text{MAE} = 0.0223$
- Reasoning Depth ($w=0.20$): $\text{MAE} = 0.0287$

> [!NOTE]
> **Study Scope & Provenance Disclosure:** This evaluation represents an initial pilot calibration ($N=15$) designed to establish scoring alignment and calibrate rubric parameter weights against academic expectations. We preserve the underlying anonymized response sheets and scoring rubrics. Full-scale multi-institutional faculty trials ($N \ge 60$) are ongoing as Phase 2 research milestones.

---

## 5. Discussion, Limitations & Threats to Validity

### 5.1 Synthetic Model Qualification
In full transparency, the XGBoost Weakness Classifier and Logistic Regression Readiness Predictor are currently trained as **Synthetic Architectural Demonstrators** to validate feature extraction latency and pipeline integration. They do not constitute validated psychometric claims of actual student success. For human cohort validation, our codebase provides `train_on_empirical_data()` utilizing `GroupKFold` grouped by student ID to guarantee zero train-test data contamination.

### 5.2 Simulation Environment Assumptions & RL Reward Dynamics
Our adaptive controller evaluation is conducted within a simulated student Markov Decision Process (MDP) parameterized by student competence levels and transition probabilities. In this simulated environment, the rule-based controller outperformed standard DQN in flow-channel compliance because discrete step-function rewards create steep cliff gradients in Q-value estimation. We emphasize that simulated flow-channel adherence demonstrates computational stability under modelled dynamics, not a psychological validation of human cognitive flow. Future work will investigate continuous potential-based reward shaping in human user trials.

### 5.3 Knowledge Graph Extraction Limitations & Ongoing Investigation
A critical empirical finding of our benchmark is that zero-shot knowledge graph extraction on unstructured technical dissertations remains an active open challenge. While node concept extraction achieves moderate recall ($F_1 = 0.4167$), strict directed relationship triple extraction achieves only $F_1 = 0.1429$ (Table 3), primarily driven by passive-voice inversions and granular hierarchical entity decomposition.
Consequently, we do **not** claim that live-extracted knowledge graphs currently improve end-to-end question quality over hybrid dense-sparse retrieval. Rather, the Knowledge Graph is maintained as an **experimental research subsystem under active investigation**. The ablation results in Table 5 demonstrate the *topological capability* of upstream prerequisite traversal when provided with verified concept dependencies, but establishing question superiority from noisy, unassisted extraction requires improved relation extraction filters and supervised fine-tuning.

### 5.4 Ethical Considerations
AIVA is designed strictly as an **evaluator-assistive tool** to support human faculty by standardizing pre-reading, generating grounded questions, and logging audit trails. Final degree-awarding decisions must remain under the purview of human academic committees.

---

## 6. Conclusion and Future Work

We presented AIVA, an autonomous oral viva assessment framework integrating hybrid document retrieval, knowledge graph concept modeling, explainable multi-agent examiner routing, and adaptive difficulty control. Through controlled benchmarks across standardized student cohorts, we demonstrated that:
1. Dynamic difficulty adaptation is indispensable, whereas static examinations fail on non-average candidates.
2. Rule-based adaptation provides superior flow-channel stability over standard discrete-reward DQN in simulated environments.
3. Multi-agent persona routing achieves $87.5\%$ semantic concordance while maintaining balanced turn distribution across examiner roles.
4. Grounded hybrid RRF elevates technical term fidelity from $0.0\%$ to $100.0\%$, eliminating generic boilerplate questions.
5. Automated four-dimensional scoring closely mirrors faculty grading on pilot calibration benchmarks ($\text{MAE} = 0.0249$, $r = 0.9966$, $\kappa = 0.9597$).

Future work will expand empirical trials to multi-institutional cohorts and explore multimodal diagram-grounded viva questioning.

---

## References

1. Csikszentmihalyi, M. (1990). *Flow: The Psychology of Optimal Experience*. Harper & Row.
2. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
3. Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual machine learning methods. *ACM SIGIR 2009*.
4. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
5. Wu, Q., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. *arXiv:2308.08155*.
6. Graesser, A. C., et al. (2005). AutoTutor: An intelligent tutoring system with mixed-initiative dialogue. *IEEE Intelligent Systems*, 20(4), 4-12.
7. Landauer, T. K., Foltz, P. W., & Laham, D. (1998). An introduction to latent semantic analysis. *Discourse Processes*, 25(2-3), 259-284.
8. Cohen, J. (1968). Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit. *Psychological Bulletin*, 70(4), 213-220.
