# AIVA Gold-Standard Knowledge Graph Annotation Guidelines
**Version:** 1.2.0  
**Authors:** AIVA Research Engineering Team  
**Scope:** Specification for annotating technical engineering dissertations to validate Knowledge Graph extraction for adaptive concept selection (UCS).

---

## 1. Objective and Pedagogical Rationale
The primary purpose of extracting a Knowledge Graph in AIVA is to guide **Unresolved Concept Selection (UCS)** during viva voce examinations. UCS assumes that:
1. Some concepts are foundational prerequisites for higher-level architectures.
2. Inquiring into foundational gaps ($A$) before assessing compound applications ($B$) accurately diagnoses student mastery.

If prerequisite edges are directionally inverted (e.g., assessing a complex downstream metric before its foundational mathematics), the examiner probes out of pedagogical order. Therefore, high relationship precision and strict directional correctness on the **prerequisite subgraph** are critical requirements.

---

## 2. Concept Annotation Protocol

### 2.1 What Qualifies as a Canonical Concept?
A canonical concept must be a distinct, technical entity that can be individually probed or evaluated in a viva examination:
- **Mathematical Foundations:** Calculus, Linear Algebra, Probability, Information Theory concepts (e.g., *Cosine Similarity*, *Cross-Entropy Loss*, *Singular Value Decomposition*).
- **Algorithms & Methods:** Concrete algorithmic procedures (e.g., *Reciprocal Rank Fusion*, *Backpropagation*, *Q-Learning*, *BM25Okapi*).
- **Model Architectures:** Specific network or system designs (e.g., *Convolutional Neural Network*, *Transformer*, *Deep Q-Network*, *XGBoost*).
- **System Components & Middleware:** Architectural building blocks (e.g., *API Gateway*, *Reverse Proxy*, *Vector Database*, *Kubernetes Ingress*).
- **Frameworks & Tools:** Concrete production libraries/runtimes (e.g., *FastAPI*, *PyTorch*, *Docker*, *PostgreSQL*).
- **Protocols & Standards:** Communication or authentication standards (e.g., *WebSockets*, *JSON Web Tokens*, *gRPC*, *HTTP/2*).
- **Evaluation Metrics & Methodologies:** Empirical evaluation procedures (e.g., *Normalized Discounted Cumulative Gain*, *BLEU*, *F1 Score*, *ROC-AUC*).

### 2.2 Aliases and Lexical Normalization
Each concept must define a canonical `id` and an exhaustive list of `aliases`:
- **Acronym expansion:** Canonical: `Deep Q-Network`, Aliases: `["DQN", "Deep Q Network", "Deep Q-Learning"]`
- **Grammatical & morphological variants:** Canonical: `Convolution`, Aliases: `["2D Convolution", "Spatial Convolution", "Convolutional Operation"]`
- **Punctuation & capitalization:** Canonical: `PostgreSQL`, Aliases: `["Postgres", "PostgreSQL Database", "Postgre"]`

---

## 3. Relationship Taxonomy & Strict Typing

Annotators must classify relationships into exactly one of the five mutually exclusive categories:

| Relationship Type | Formal Definition | Subgraph Constraint | Example Triple |
| :--- | :--- | :--- | :--- |
| **`prerequisite`** | Understanding concept $A$ is strictly necessary before understanding concept $B$ in a computer science curriculum. | **Strict DAG ($G_{\text{prereq}}$): Directed Acyclic Graph. Cycles strictly prohibited.** | `(Convolution, prerequisite, Feature Extraction)` |
| **`dependency`** | Operational/execution dependency at runtime or software build time ($A$ invokes, requires, or consumes $B$). | General Directed Graph (Cycles allowed in mutual dependencies). | `(FastAPI, dependency, PostgreSQL)` |
| **`part_of`** | Architectural composition / structural containment ($A$ is a subsystem, layer, or module of $B$). | General Directed Graph (Hierarchical tree/DAG preferred). | `(Horizontal Pod Autoscaler, part_of, Kubernetes)` |
| **`implements`** | A concrete component or software realization of an abstract protocol, model, or algorithm. | General Directed Graph. | `(ChromaDB, implements, Vector Database)` |
| **`evaluates`** | A metric, benchmark, or loss function used to measure or assess a model or system. | General Directed Graph. | `(Normalized Discounted Cumulative Gain, evaluates, Hybrid RAG)` |

---

## 4. Prerequisite Direction Convention & Rule-Based Checks

### 4.1 Direction Convention
For any relationship of type `prerequisite`:
$$\text{Source} \xrightarrow{\text{prerequisite}} \text{Target} \iff \text{Source is a prerequisite of Target}$$
- **Source:** The foundational concept.
- **Target:** The advanced/dependent concept requiring knowledge of the source.
- `source_is_prerequisite` must be set to `true`.

### 4.2 Counter-Example (Common Failure Mode):
- **Incorrect:** `(Adaptive Difficulty, prerequisite, Deep Q-Network)` with rationale "Adaptive difficulty uses DQN".
- **Correction:** The relationship is `(Deep Q-Network, dependency, Adaptive Difficulty)` or `(Markov Decision Process, prerequisite, Deep Q-Network)`.

### 4.3 Rule-Based Validity Constraints
1. **Anti-Symmetry of Prerequisites:** If $(A, \text{prerequisite}, B) \in G$, then $(B, \text{prerequisite}, A) \notin G$.
2. **Subsystem Isolation for Cycles:**
   - Cycles in $G_{\text{dependency}}$ (e.g., service A calls service B, and service B notifies service A) are semantically valid and **must not be deleted**.
   - Cycles in $G_{\text{prereq}}$ are epistemically invalid. Any directed cycle in $G_{\text{prereq}}$ indicates an annotation error or extraction failure.
3. **Prerequisite Transitivity:** If $A \to B$ and $B \to C$ are prerequisites, $A \to C$ is implicitly true. Explicit direct edges $A \to C$ are allowed but will be pruned during transitive reduction for evaluation clarity.

---

## 5. Dual-Annotation and Consensus Workflow
1. **Independent Annotation:** Annotator 1 and Annotator 2 independently annotate the document text without cross-communication.
2. **Inter-Rater Agreement (Cohen's Kappa):**
   - Concept boundary & identification agreement: $\kappa_{\text{concept}}$.
   - Prerequisite edge classification and direction agreement: $\kappa_{\text{prereq}}$.
   - Acceptance threshold: $\kappa \ge 0.75$.
3. **Consensus Reconciliation:** Any discrepancies where `consensus: false` are reviewed by a lead annotator to produce a reconciled gold label and document `reconciliation_notes`.

---

## 6. Dataset Split Policy (Pre-Registration)
To prevent overfitting during extraction prompt tuning and pipeline refinement:
- **Pilot & Validation Split (`pilot_val`):** 5–10 documents used for development, error taxonomy identification, and prompt tuning.
- **Frozen Test Split (`frozen_test`):** 25–40 documents completely frozen. No prompt modifications, alias additions, or pipeline heuristics may be developed against this split prior to final evaluation reporting.
