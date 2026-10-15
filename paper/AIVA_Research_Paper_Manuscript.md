# An Empirical Evaluation of Knowledge-Grounded Multi-Agent Panels and Reinforcement Learning for Adaptive Oral Viva Assessment

**Kathambari K.**  
*Department of Computer Science and Engineering*  
*Autonomous Intelligent Viva Assessment (AIVA) Research Project*

---

## Abstract
Oral viva voce examinations are the gold standard for assessing student dissertation ownership, conceptual depth, and technical defense capabilities in engineering education. However, traditional human-conducted vivas suffer from evaluator fatigue, inter-examiner subjectivity, and high scheduling overhead. In this paper, we present and empirically evaluate **AIVA (Adaptive Intelligent Viva Assessment)**, an autonomous oral examination framework that unifies: (1) document-grounded hybrid retrieval combining technical BM25 tokenization and dense embeddings via Reciprocal Rank Fusion ($k=60$), (2) automated knowledge graph extraction for topological concept tracking, (3) a traceable multi-agent examiner panel with dynamic semantic routing across four specialized academic personas (*The Professor*, *The Industry Expert*, *The Research Critic*, and *The General Examiner*), and (4) real-time reinforcement learning difficulty adaptation targeting the psychological flow channel. Across rigorous controlled simulations spanning three student skill cohorts ($0.30, 0.60, 0.85$), we demonstrate that static-difficulty examinations collapse completely on non-average candidates ($0.0\%$ flow adherence), confirming the necessity of dynamic adaptation. Our findings also reveal that a deterministic rule-based controller outperforms standard discrete-reward Deep Q-Networks ($95.1\%$ vs $81.4\%$ flow adherence), providing critical empirical evidence regarding RL reward continuity in low-dimensional educational state spaces. Ablation experiments show that hybrid retrieval improves report technical term fidelity from $0.0\%$ to $75.0\%$, while multi-agent persona routing elevates question vocabulary diversity by $+29.4\%$ ($0.9028$ vs $0.6977$ Type-Token Ratio) and achieves $87.5\%$ semantic concordance over sequential round-robin routing ($37.5\%$). We provide full reproducible configurations, baseline comparison data, and an open evaluation protocol for autonomous assessment.

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
We evaluated persona distribution balance and semantic domain concordance across multi-turn technical student dialogues comparing sequential round-robin against AIVA's keyword-guided router.

**Table 2: Multi-Agent Persona Routing Comparison (`results_multi_agent_routing.json`)**
| Strategy | Persona Distribution (Prof / Ind / Critic / Exam) | Shannon Entropy (Max: 2.0) | Uniformity Ratio | Semantic Concordance % |
| :--- | :---: | :---: | :---: | :---: |
| **Round-Robin Baseline** | 7 / 0 / 0 / 1 | 0.544 bits | 27.2% | **37.5%** |
| **AIVA Keyword-Guided** | 2 / 4 / 1 / 1 | **1.750 bits** | **87.5%** | **87.5%** |

**Analysis:** Round-robin rotation blindly cycles without regard to student utterance topics, yielding low semantic concordance ($37.5\%$). In contrast, AIVA’s router achieves $87.5\%$ semantic concordance while maintaining near-maximal entropy ($1.750$ bits, $87.5\%$ uniformity) across examiner turns.

### 4.3 Knowledge Graph Extraction Evaluation
Extracted concepts and relationships were benchmarked against expert human-annotated reference entities and typed triples.

**Table 3: Knowledge Graph Extraction Factuality (`results_knowledge_graph.json`)**
| Extraction Entity | Reference Count | Extracted Count | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Technical Concepts** | 10 | 11 | **0.9091** | **1.0000** | **0.9524** |
| **Typed Relations** | 7 | 7 | **0.8571** | **0.8571** | **0.8571** |

### 4.4 Hybrid RAG Retrieval Quality
Sparse BM25 with technical tokenization was compared with dense cosine retrieval and Hybrid RRF ($k=60$) across 10 engineering dissertation queries:
- **Recall@1, Recall@3, Recall@5:** **$1.0000$**
- **Mean Reciprocal Rank (MRR):** **$1.0000$**
- **nDCG@3, nDCG@5:** **$1.0000$**

### 4.5 Component Ablation Studies
To quantify the isolated value of each subsystem, we performed controlled ablations:

**Table 4: Component Ablation Results (`results_ablation_studies.json`)**
| Ablation Experiment | Full AIVA System | Ablated Baseline System | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Context Grounding (Fidelity %)** | **75.0%** (RAG Grounded) | **0.0%** (Ungrounded Generic) | **+75.0%** report term fidelity |
| **Persona Specialization (TTR)** | **0.9028** (4 Personas) | **0.6977** (Monolithic Examiner) | **+29.4%** lexical diversity |
| **Relational KG Guidance (Paths)** | **4 Paths** Tracked | **0 Paths** (Flat Keyword Search) | Full dependency reasoning |

### 4.6 Rubric Inter-Rater Reliability
Evaluator scoring across the 4-dimensional rubric achieved high alignment with human faculty benchmark scores:
- **Mean Absolute Error (MAE):** $0.0333$
- **Pearson Correlation ($r$):** $0.9799$
- **Quadratic Weighted Cohen's Kappa:** $0.7273$ (substantial inter-rater reliability)

---

## 5. Discussion, Limitations & Threats to Validity

### 5.1 Synthetic Model Qualification
In full transparency, the XGBoost Weakness Classifier and Logistic Regression Readiness Predictor are currently trained as **Synthetic Architectural Demonstrators** to validate feature extraction latency and pipeline integration. They do not constitute validated psychometric claims of actual student success. For human cohort validation, our codebase provides `train_on_empirical_data()` utilizing `GroupKFold` grouped by student ID to guarantee zero train-test data contamination.

### 5.2 Discontinuous RL Reward Dynamics
Our experimental comparison revealed that deterministic rule-based controllers achieved higher flow-channel compliance than standard DQN. This occurs because discrete step-function rewards create steep cliff gradients in Q-value estimation. Future iterations will investigate continuous potential-based reward shaping to allow RL to capture long-horizon examination pacing.

### 5.3 Ethical Considerations
AIVA is designed strictly as an **evaluator-assistive tool** to support human faculty by standardizing pre-reading, generating grounded questions, and logging audit trails. Final degree-awarding decisions must remain under the purview of human academic committees.

---

## 6. Conclusion and Future Work

We presented AIVA, an autonomous oral viva assessment framework integrating hybrid document retrieval, knowledge graph concept modeling, explainable multi-agent examiner routing, and adaptive difficulty control. Through controlled benchmarks across standardized student cohorts, we demonstrated that:
1. Dynamic difficulty adaptation is indispensable, whereas static examinations fail on non-average candidates.
2. Rule-based adaptation provides superior flow-channel stability over standard discrete-reward DQN.
3. Multi-agent persona routing achieves $87.5\%$ semantic concordance and $+29.4\%$ question diversity over monolithic examiners.
4. Grounded hybrid RRF elevates technical term fidelity from $0.0\%$ to $75.0\%$.

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
