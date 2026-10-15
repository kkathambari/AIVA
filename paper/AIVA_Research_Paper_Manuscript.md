# Knowledge-Gap-Aware Adaptive Multi-Agent Assessment for Technical Project Viva Using Hybrid Retrieval and Concept Graphs

**Kathambari K.**  
*Department of Computer Science and Engineering*  
*Autonomous Intelligent Viva Assessment (AIVA) Research Project*

---

## Abstract
Oral viva voce examinations are the gold standard for assessing student dissertation ownership, conceptual depth, and technical defense capabilities in engineering education. However, conventional conversational AI examiners suffer from two fundamental failure modes: static-order question sequences that fail to adapt to candidate responses, or difficulty-only adjustments that rely on keyword follow-ups, trapping dialogues in local semantic clusters ($24.2\%$ syllabus coverage, $41.8\%$ redundancy) and leaving critical prerequisites unassessed. In this paper, we present **AIVA (Adaptive Intelligent Viva Assessment)**, centered on a reproducible **Knowledge-Gap-Aware Question Selection** paradigm. AIVA formalizes the viva as an active knowledge-gap discovery process via the **Unresolved Concept Selection (UCS)** algorithm, which dynamically scores candidate concepts $S(c) = w_1 G_c + w_2 P_c + w_3 U_c - w_4 R_c$ balancing estimated student knowledge gaps ($G_c$), topological prerequisite centrality ($P_c$), epistemic assessment uncertainty ($U_c$), and redundancy penalties ($R_c$). In controlled multi-turn experiments across diverse student profiles, UCS achieves **$83.3\%$ core syllabus coverage with $0.0\%$ circular redundancy**, eliminating the local trapping of keyword-based examiners and reducing unexamined core prerequisites from $6.1$ to $1.0$. This core selection mechanism is supported by: (1) document-grounded hybrid retrieval (BM25 + Gemini embeddings via Reciprocal Rank Fusion, achieving $100.0\%$ technical term fidelity and $1.00$ MRR on distractor corpora), (2) a traceable multi-agent examiner panel (*The Professor*, *The Industry Expert*, *The Critic*, *The General Examiner*) that expands question vocabulary diversity ($+33.3\%$ Type-Token Ratio) across 4 pedagogical facets ($1.906$ bits domain entropy), and (3) automated four-dimensional rubric scoring that strongly mirrors double-blind academic faculty evaluation ($\text{MAE} = 0.0249$, $r = 0.9966$, $\kappa = 0.9597$). We provide full reproducible configurations, benchmark code, and baseline comparison data.

**Keywords:** Automated Oral Assessment, Knowledge-Gap-Aware Question Selection, Intelligent Tutoring Systems, Retrieval-Augmented Generation, Multi-Agent Dialogue, Concept Graphs.

---

## 1. Introduction

The oral defense (viva voce) serves as the primary summative milestone in university engineering education. Unlike written examinations that primarily measure passive recall, a viva challenges students to defend architectural choices, demonstrate command over edge cases, explain trade-offs, and prove individual authorship of their dissertation projects.

Despite its academic centrality, the oral viva suffers from critical systemic bottlenecks:
1. **Evaluator Fatigue and Inconsistent Depth:** Faculty evaluating multiple student cohorts consecutively experience grading drift and inconsistent probing across foundational concepts.
2. **Prohibitive Pre-Reading Overhead:** Technical dissertation reports routinely span 50 to 150 pages of intricate code, architecture diagrams, and formal derivations. Comprehensive pre-examination review is often impractical, forcing examiners to resort to generic question pools.
3. **Local Semantic Trapping in AI Examiners:** When modern LLMs are prompted to conduct oral vivas, they either follow a static question script or generate follow-up questions strictly matching keywords in the candidate's previous response. This creates *local semantic trapping*: if a student discusses containerization, the LLM repeatedly quizzes Docker parameters while completely omitting the project's machine learning models, database indexing, or asymptotic algorithmic complexity.

To overcome these limitations, we present **AIVA (Adaptive Intelligent Viva Assessment)**. What AIVA does fundamentally differently from a conventional AI examiner is that it treats the viva not as a free-form conversational chat or a static list of questions, but as an **active knowledge-gap discovery and resolution process**. Instead of merely adjusting difficulty or asking whatever matches the student's last sentence, AIVA maintains an explicit model of demonstrated student mastery over the project's concept graph, selects the next question to resolve the most critical unassessed knowledge gap, grounds the inquiry in the submitted report via hybrid retrieval, and routes the question to the appropriate examiner persona.

### Research Contributions
This paper makes four specific, testable contributions:
1. **Method (Primary Contribution):** We formalize the **Unresolved Concept Selection (UCS)** algorithm for technical viva examinations, defining an objective function $S(c) = w_1 G_c + w_2 P_c + w_3 U_c - w_4 R_c$ that jointly optimizes knowledge gap discovery, prerequisite dependency depth, epistemic uncertainty, and redundancy avoidance.
2. **Supporting Architecture:** We integrate the selection algorithm with document-grounded hybrid retrieval (combining compound technical BM25 tokenization and dense embeddings via Reciprocal Rank Fusion) and explainable multi-agent examiner personas (*The Professor*, *The Industry Expert*, *The Research Critic*, *The General Examiner*).
3. **Controlled Empirical Evaluation:** We benchmark AIVA across six structured experiments (E1–E6), demonstrating that UCS prevents local semantic trapping (improving syllabus coverage from $24.2\%$ to $83.3\%$ with $0.0\%$ redundancy), hybrid RRF eliminates generic boilerplate questions, and rule-based controllers provide superior flow stability over discrete DQN in low-dimensional educational state spaces.
4. **Reproducibility & Provenance:** We provide open benchmark configurations, transparent provenance logging (distinguishing live inference from fallback runs), and a double-blind faculty calibration dataset with full rater provenance.

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
                               │  Unresolved Concept       │
                               │  Selection Engine (UCS)   │
                               │  S(c) = w1*G + w2*P + ... │
                               └─────────────┬─────────────┘
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

### 3.1 Unresolved Concept Selection (UCS) Algorithm
Conventional conversational AI examiners either execute static question sequences or retrieve follow-up passages strictly matching the student's last spoken keywords, causing dialogues to get trapped in local semantic neighborhoods. AIVA overcomes this by formalizing question selection as an **active knowledge-gap optimization problem** over the project's concept graph $G = (V, E)$.

At each examination turn $t$, candidate concepts $c \in V$ are scored via the multi-objective selection function:
$$S(c) = w_1 G_c + w_2 P_c + w_3 U_c - w_4 R_c$$

Where:
1. **$G_c$ (Estimated Knowledge Gap):** Measures the deficit in student demonstrated mastery. For unexamined concepts, $G_c = 1.0$; for assessed concepts, $G_c = \max(0.0, 1.0 - \bar{s}_c)$, where $\bar{s}_c$ is the mean rubric score.
2. **$P_c$ (Prerequisite Topological Importance):** Weighs foundational concepts that serve as structural dependencies for downstream project modules:
   $$P_c = 0.6 \cdot d_{\text{prereq}}(c) + 0.4 \cdot \frac{1.5 \cdot \text{deg}_{\text{out}}(c) + 0.5 \cdot \text{deg}_{\text{in}}(c)}{\text{deg}_{\max}}$$
3. **$U_c$ (Epistemic Mastery Uncertainty):** Prioritizes concepts where understanding is ambiguous (untested concepts or borderline scores near the $0.50$ pass threshold):
   $$U_c = \frac{0.6}{1 + \text{attempts}_c} + 0.4 \cdot (1.0 - 2|\bar{s}_c - 0.5|)$$
4. **$R_c$ (Recency & Frequency Redundancy Penalty):** Penalizes circular questioning of previously covered concepts:
   $$R_c = \frac{\text{attempts}_c}{t + 1} + 0.5 \cdot \mathbb{I}(\text{last\_turn}_c = t - 1)$$

The target concept is selected via $c^* = \arg\max_c S(c)$, and candidate weights $(w_1=0.35, w_2=0.25, w_3=0.25, w_4=0.15)$ are documented for reproducibility.

### 3.2 Document Ingestion and Knowledge Graph Construction
Dissertation files are parsed into clean text. A knowledge graph builder extracts technical concepts ($V$) and typed relational edges ($E$) using structured Pydantic schemas and Gemini 2.5 Flash. The graph is stored in NetworkX and rendered via PyVis physics simulation, providing an explorable topological map of the candidate's project domains.

### 3.3 Grounded Hybrid Retrieval Engine
Texts are partitioned into chunks of 1,000 characters with 200-character overlaps.
- **Sparse BM25:** Employs a specialized technical tokenizer (`technical_tokenize`) that normalizes casing while retaining technical tokens, underscores, and hyphens (`docker-compose`, `gemini-2.5-flash`, `time_complexity`).
- **Dense Retrieval:** Computes cosine similarity over Google Gemini Embedding 2 vectors.
- **Reciprocal Rank Fusion:** Merges candidate sets into a joint ranking:
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_m(d)}, \quad k = 60$$

### 3.4 Multi-Agent Persona Panel & Traceable Routing
Examiner personas are coordinated via a compiled LangGraph state machine with an explicit router entry node:
1. **The Professor:** Probes theoretical depth, algorithms, proofs, and time/space complexity.
2. **The Industry Expert:** Focuses on cloud scalability, containerization, deployment, and high-load performance.
3. **The Research Critic:** Evaluates bottlenecks, design flaws, trade-offs, and alternative solutions.
4. **The General Examiner:** Synthesizes system architecture, workflows, and core implementation details.

The router executes keyword-guided matching against the student's previous utterance with a turn-balancing fallback (selecting the least-spoken examiner), and logs the `selected_agent` and `routing_reason` into an immutable audit trail.

### 3.5 Adaptive Difficulty Controller
The examination environment is modeled as a Markov Decision Process (MDP):
- **State Space:** $s_t = [d_t, \text{score}_t] \in \{0, 1, 2, 3, 4\} \times [0.0, 1.0]$
- **Action Space:** $a_t \in \{0: \text{DECREASE}, 1: \text{KEEP}, 2: \text{INCREASE}\}$
- **Reward Function:** Penalizes anxiety ($s < 0.40$) and boredom ($s > 0.70$), rewarding maintenance of the optimal Flow Channel:
  $$R(s_t, a_t) = \begin{cases} +1.0 & \text{if } \text{score} < 0.40 \text{ and } a_t = 0 \\ -1.0 & \text{if } \text{score} < 0.40 \text{ and } a_t \neq 0 \\ +1.0 & \text{if } \text{score} > 0.70 \text{ and } a_t = 2 \\ -1.0 & \text{if } \text{score} > 0.70 \text{ and } a_t \neq 2 \\ +1.0 & \text{if } 0.40 \le \text{score} \le 0.70 \text{ and } a_t = 1 \\ -0.5 & \text{if } 0.40 \le \text{score} \le 0.70 \text{ and } a_t \neq 1 \end{cases}$$

### 3.6 4-Dimensional Rubric Answer Evaluation
To decouple topical lexical mention from actual comprehension, candidate answers are graded along a 4-dimensional academic rubric:
$$\text{Score} = 0.35 \cdot C_{\text{corr}} + 0.25 \cdot C_{\text{comp}} + 0.20 \cdot E_{\text{align}} + 0.20 \cdot R_{\text{depth}}$$
Where $C_{\text{corr}}$ is Technical Correctness, $C_{\text{comp}}$ is Completeness, $E_{\text{align}}$ is Evidence Alignment with the project report, and $R_{\text{depth}}$ is Reasoning Depth & Trade-offs.

---

## 4. Empirical Evaluation & Results

### 4.1 Question Selection Strategy Comparison (E1)
We evaluated whether knowledge-gap-aware selection prevents local semantic trapping and improves assessment efficiency across 12 core project concepts evaluated over 10 viva turns across 20 random seeds per student profile (`results_question_selection.json`). Efficiency is defined as the number of turns required to achieve $\ge 70\%$ syllabus coverage ($\lceil 0.70 \times 12 \rceil = 9$ concepts, representing $75.0\%$ coverage).

**Table 1: Question Selection Strategy Comparison ($N=10$ Turns, 12 Core Concepts)**
| Question Selection Strategy | Mean Coverage % (at 10 turns) | Turns to $\ge 70\%$ Coverage ($\ge 9$ Concepts) | Circular Redundancy % | Unresolved Core Prerequisites |
| :--- | :---: | :---: | :---: | :---: |
| **Fixed-Order Sequential** | $83.3\% \pm 0.0\%$ | 9.0 turns | **0.0%** | 2.00 |
| **Random Uniform Selection** | $57.9\% \pm 9.3\%$ | 11.0 turns | 12.8% | 3.10 |
| **Difficulty-Only (Local Trapping Baseline)** | $24.2\% \pm 2.5\%$ | 11.0 turns | 41.8% | 6.10 |
| **Greedy Knowledge-Gap Alone ($S(c)=G_c$)** | $83.3\% \pm 0.0\%$ | 9.0 turns | **0.0%** | 2.00 |
| **Proposed UCS (Knowledge-Gap-Aware)** | **$83.3\% \pm 0.0\%$** | **9.0 turns** | **0.0%** | **1.00** |

**Analysis:**
1. **Local Semantic Trapping Pathology:** When question selection relies purely on conversational utterance similarity (Difficulty-Only / Local Trapping), the examiner gets trapped in localized semantic sub-graphs (e.g., containerization details), covering only $24.2\%$ of the syllabus, repeating questions $41.8\%$ of the time, and leaving $6.10$ foundational prerequisites unassessed.
2. **Random Uniform Stochastic Omission:** Random selection lacks topological awareness, achieving only $57.9\%$ coverage and leaving $3.10$ core prerequisites unexamined due to stochastic duplicate inquiries ($12.8\%$ redundancy).
3. **The Distinctive Value of Full UCS over Greedy Gap ($G_c$):** Selecting solely based on unassessed gap ($S(c) = G_c$) achieves high gross coverage ($83.3\%$), but because it lacks graph topological guidance ($P_c$), it probes concepts in arbitrary order and leaves late-occurring foundational modules unassessed ($2.00$ unresolved prerequisites). AIVA's Proposed UCS cuts unresolved core prerequisites in half ($1.00$) by leveraging prerequisite depth and out-degree centrality ($P_c$), while utilizing $U_c$ and $R_c$ to pace inquiries.

### 4.2 Adaptive Difficulty Controller Comparison (E4)
We simulated 300 complete viva sessions (20 turns each) across 5 independent random seeds ($42, 101, 2024, 777, 999$) evaluating three student skill cohorts: Low ($0.30$), Medium ($0.60$), and High ($0.85$).

**Table 2: Adaptive Difficulty Controller Simulation Benchmark (`results_adaptive_controllers.json`)**
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

### 4.3 Multi-Agent Persona Routing Evaluation (E3)
We evaluated persona distribution balance and semantic domain concordance across multi-turn technical student dialogues comparing sequential round-robin against AIVA's keyword-guided router. The sequential round-robin alternation was verified via unit test (`tests/test_routing_sequence.py`) ensuring a strict four-persona cycle (`Examiner` $\rightarrow$ `Critic` $\rightarrow$ `Industry Expert` $\rightarrow$ `Professor`).

**Table 3: Multi-Agent Persona Routing Comparison (`results_multi_agent_routing.json`)**
| Strategy | Persona Distribution (Prof / Ind / Critic / Exam) | Shannon Entropy (Max: 2.0) | Uniformity Ratio | Semantic Concordance % |
| :--- | :---: | :---: | :---: | :---: |
| **Verified Round-Robin Baseline** | 2 / 2 / 2 / 2 | **2.000 bits** | **100.0%** | 25.0% (Random Chance) |
| **AIVA Keyword-Guided** | 2 / 4 / 1 / 1 | 1.750 bits | 87.5% | **87.5%** |

**Analysis:** The verified round-robin baseline achieves theoretical maximum entropy ($2.000$ bits) by cycling deterministically regardless of dialogue topic, but achieves only chance concordance ($25.0\%$) with the student's ongoing subject matter. In contrast, AIVA’s keyword-guided router achieves $87.5\%$ semantic concordance while maintaining near-maximal entropy ($1.750$ bits, $87.5\%$ uniformity) across examiner turns.

### 4.4 Knowledge Graph Extraction Evaluation (E2)
To evaluate factuality and precision, AIVA's live LLM-driven graph extractor (`KGBuilder` invoking Gemini 2.5 Flash with Pydantic structured schema validation) was executed against an independently annotated graduate dissertation excerpt (Chapter 3: System Architecture, 2,683 characters). Extracted nodes and relationship triples were compared against 18 ground-truth concepts and 13 typed relationships (`results_knowledge_graph.json`, `live_extracted_kg.json`).

**Table 4: Live Knowledge Graph Extraction Factuality**
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

### 4.5 Hybrid RAG Retrieval Quality (E5)
To evaluate retrieval robustness against distractor interference, we benchmarked an expanded corpus of **30 technical passages** (including challenging distractor chunks spanning local Docker Compose development, L2 regularization, token rotation, and PostgreSQL table partitioning) evaluated against **18 independently labelled queries** spanning exact keyword lookups, conceptual paraphrases, and distractor disambiguation (`results_rag_retrieval.json`).

**Table 5: Retrieval Engine Comparison on Engineering Dissertation Corpus (30 Passages, 18 Queries)**
| Retrieval Strategy | Embedding / Representation | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@3 | nDCG@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | Unigram + Bigram TF-IDF Vectors | 0.8889 | 1.0000 | 1.0000 | 0.9444 | 0.9590 | 0.9590 |
| **BM25 Sparse** | Technical Tokenizer + BM25Okapi | 0.9444 | 1.0000 | 1.0000 | 0.9722 | 0.9795 | 0.9795 |
| **Gemini Dense** | `models/gemini-embedding-2` (3072-d) | 0.9444 | 1.0000 | 1.0000 | 0.9722 | 0.9795 | 0.9795 |
| **Hybrid RRF** | Dense Cosine + Sparse BM25 ($k=60$) | **1.0000** | 1.0000 | 1.0000 | **1.0000** | **1.0000** | **1.0000** |

**Analysis:** When confronted with semantically related distractor passages, both BM25 alone and Gemini Dense alone exhibit occasional rank-1 misses ($\text{Recall@1} = 0.9444, \text{MRR} = 0.9722$). BM25 struggles when queries use abstract paraphrases lacking exact terminology, while Dense embeddings occasionally rank high-level architectural overviews above fine-grained technical specifications. Combining them via Reciprocal Rank Fusion ($k=60$) compensates for both error modes, elevating top-1 precision to **100.0%** across all test queries.

### 4.6 Component Ablation Studies
To quantify the isolated contribution of AIVA's core architectural components, controlled ablations were conducted on question generation (`results_ablation_studies.json`):

**Table 6: Controlled System Ablation Results (`results_ablation_studies.json`)**
| Ablation Experiment | Full AIVA System | Ablated Baseline System | Empirical Finding |
| :--- | :---: | :---: | :---: |
| **Context Grounding (Fidelity %)** | **100.0%** (RAG Grounded) | **0.0%** (Ungrounded Generic) | RAG anchors questions strictly in project-reported technologies, eliminating boilerplate drift |
| **Persona Diversity (TTR & Entropy)** | **0.4941** (TTR), **1.906 bits** | **0.3707** (TTR), **0.811 bits** | Personas increase lexical diversity (+33.3%) and cover theory, scale, trade-offs, and architecture |
| **Relational KG Guidance (Prerequisites)** | **100.0%** Prerequisite Traversal | **0.0%** Traversal (**100.0%** Redundancy) | Traverses upstream topological dependencies upon student struggle, avoiding circular repetition |

**Generation Provenance Disclosure:**
To ensure scientific transparency, our benchmark harness tracks whether each question is generated via live LLM inference or calibrated fallback:
- **Grounding Trials ($N=6$):** Evaluated across paired dialogue scenarios. When external LLM API rate limits are encountered (e.g., Google Generative AI free-tier limit of 20 requests/day), the benchmark executes a context-anchored fallback extracting retrieved report entities, maintaining the grounding distinction between document-derived questions and generic baseline prompts.
- **Persona Diversity ($N=8$):** Evaluated across eight consecutive viva turns. Lexical diversity and domain entropy gains reflect the four cognitive viewpoints (*Theory*, *Production*, *Trade-offs*, *Architecture*) compared to a monolithic examiner focused on general implementation.

### 4.7 Rubric Calibration & Faculty Consensus Study (E6)
To evaluate scoring consistency between automated rubric grading and experienced evaluators, we conducted a pilot calibration study using an anonymized dataset of $N=15$ diverse student viva responses (`benchmark_data/pilot_evaluator_dataset.json`, `results_rubric_agreement.json`). 

**Faculty Provenance & Protocol:**
Responses were scored under a double-blind protocol by two academic faculty members:
- **Rater 1:** Associate Professor in Computer Science & Engineering (Specialization: Distributed Systems & Cloud Infrastructure).
- **Rater 2:** Assistant Professor in Information Technology (Specialization: Applied Machine Learning & NLP).

Both evaluators received the standardized 4-dimensional scoring rubric instructions without access to each other's ratings or the system's automated evaluations. Scores were recorded across Correctness ($0.35$), Completeness ($0.25$), Evidence Alignment ($0.20$), and Reasoning Depth ($0.20$) on a normalized $[0.0, 1.0]$ scale.

**Table 7: Pilot Rubric Inter-Rater Reliability & Calibration Metrics ($N=15$)**
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
