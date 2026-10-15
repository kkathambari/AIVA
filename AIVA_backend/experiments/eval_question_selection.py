"""
AIVA Research Benchmark: Question Selection Strategy Evaluation (E1)
--------------------------------------------------------------------
Empirically benchmarks the proposed Knowledge-Gap-Aware Unresolved Concept Selection (UCS)
against standard and competitive baseline strategies on a standardized 12-concept engineering project graph:

1. Fixed-Order Sequential: Static syllabus / report appearance order
2. Random Uniform Selection: Uniform stochastic question sampling across concepts
3. Difficulty-Only (Local Trapping Baseline): Adjusts difficulty, but retrieves chunks purely from last answer keywords
4. Greedy Knowledge-Gap Alone (S(c) = G_c): Evaluates gap G_c without prerequisite topology, uncertainty, or redundancy penalties
5. Proposed UCS (Full Formulation): S(c) = w1*G_c + w2*P_c + w3*U_c - w4*R_c

Evaluation Metrics:
- Concept Coverage (%) at 10 turns
- Assessment Efficiency: Turns required to reach >= 70% syllabus coverage (Target: ceil(0.70 * N_concepts))
- Question Redundancy Rate (%): Turns spent re-asking already resolved/mastered concepts (score >= 0.70)
- Unresolved Core Prerequisites: Number of high-prerequisite foundational concepts (depth >= 0.8) left unassessed
"""

import os
import sys
import json
import random
import math
import datetime
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ucs_engine import UCSEngine

# Standardized 12-Concept Engineering Dissertation Concept Graph with Prerequisite Topology
BENCHMARK_CONCEPTS = [
    {"id": "FastAPI Gateway", "type": "Architecture", "prereq_depth": 0.3},
    {"id": "Asynchronous Event Loops", "type": "Theory", "prereq_depth": 0.8},
    {"id": "Docker Containerization", "type": "Production", "prereq_depth": 0.7},
    {"id": "Kubernetes Pod Orchestration", "type": "Production", "prereq_depth": 0.8},
    {"id": "Horizontal Pod Autoscaler", "type": "Production", "prereq_depth": 0.9},
    {"id": "PostgreSQL Composite Indexing", "type": "Production", "prereq_depth": 0.6},
    {"id": "BM25Okapi Lexical Retrieval", "type": "Theory", "prereq_depth": 0.7},
    {"id": "Gemini Dense Embeddings", "type": "Theory", "prereq_depth": 0.7},
    {"id": "Reciprocal Rank Fusion", "type": "Theory", "prereq_depth": 0.9},
    {"id": "Markov Decision Process", "type": "Theory", "prereq_depth": 0.9},
    {"id": "Deep Q-Network (DQN)", "type": "Theory", "prereq_depth": 0.9},
    {"id": "XGBoost Weakness Ensemble", "type": "Theory", "prereq_depth": 0.8}
]

BENCHMARK_RELATIONSHIPS = [
    {"source": "Docker Containerization", "target": "Kubernetes Pod Orchestration", "type": "DEPENDS_ON"},
    {"source": "Kubernetes Pod Orchestration", "target": "Horizontal Pod Autoscaler", "type": "SCALES_WITH"},
    {"source": "Asynchronous Event Loops", "target": "FastAPI Gateway", "type": "POWERS"},
    {"source": "BM25Okapi Lexical Retrieval", "target": "Reciprocal Rank Fusion", "type": "COMBINED_IN"},
    {"source": "Gemini Dense Embeddings", "target": "Reciprocal Rank Fusion", "type": "COMBINED_IN"},
    {"source": "Markov Decision Process", "target": "Deep Q-Network (DQN)", "type": "FORMULATES"},
    {"source": "PostgreSQL Composite Indexing", "target": "FastAPI Gateway", "type": "PERSISTS"}
]

# Student Cognitive Profiles across the 12 concepts (Ground-truth mastery level [0.0 - 1.0])
STUDENT_PROFILES = {
    "Competent_Student": {
        "FastAPI Gateway": 0.90, "Asynchronous Event Loops": 0.85, "Docker Containerization": 0.90,
        "Kubernetes Pod Orchestration": 0.80, "Horizontal Pod Autoscaler": 0.75, "PostgreSQL Composite Indexing": 0.85,
        "BM25Okapi Lexical Retrieval": 0.80, "Gemini Dense Embeddings": 0.85, "Reciprocal Rank Fusion": 0.75,
        "Markov Decision Process": 0.70, "Deep Q-Network (DQN)": 0.70, "XGBoost Weakness Ensemble": 0.85
    },
    "Struggling_Student": {
        "FastAPI Gateway": 0.50, "Asynchronous Event Loops": 0.30, "Docker Containerization": 0.45,
        "Kubernetes Pod Orchestration": 0.35, "Horizontal Pod Autoscaler": 0.20, "PostgreSQL Composite Indexing": 0.40,
        "BM25Okapi Lexical Retrieval": 0.35, "Gemini Dense Embeddings": 0.30, "Reciprocal Rank Fusion": 0.25,
        "Markov Decision Process": 0.20, "Deep Q-Network (DQN)": 0.20, "XGBoost Weakness Ensemble": 0.35
    },
    "Uneven_Student": {
        "FastAPI Gateway": 0.90, "Asynchronous Event Loops": 0.40, "Docker Containerization": 0.85,
        "Kubernetes Pod Orchestration": 0.80, "Horizontal Pod Autoscaler": 0.75, "PostgreSQL Composite Indexing": 0.85,
        "BM25Okapi Lexical Retrieval": 0.35, "Gemini Dense Embeddings": 0.40, "Reciprocal Rank Fusion": 0.30,
        "Markov Decision Process": 0.25, "Deep Q-Network (DQN)": 0.30, "XGBoost Weakness Ensemble": 0.40
    }
}

TOTAL_TURNS = 10
NUM_SEEDS = 20

def get_target_70() -> int:
    """Returns strictly ceil(70% * N_concepts). For 12 concepts, ceil(8.4) = 9 concepts (75.0%)."""
    return math.ceil(0.70 * len(BENCHMARK_CONCEPTS))

def simulate_student_response(true_mastery: float, seed: int) -> float:
    """Simulates an observed rubric evaluation score with slight response noise."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, 0.05)
    return float(np.clip(true_mastery + noise, 0.0, 1.0))

# --- Strategy 1: Fixed-Order Sequential ---
def run_fixed_order(profile: dict, seed: int):
    assessed = {}
    turns_to_70 = None
    target_70 = get_target_70()
    redundancy_count = 0
    
    for turn in range(TOTAL_TURNS):
        cid = BENCHMARK_CONCEPTS[turn % len(BENCHMARK_CONCEPTS)]["id"]
        if cid in assessed and assessed[cid]["mean_score"] >= 0.70:
            redundancy_count += 1
            
        score = simulate_student_response(profile[cid], seed * 100 + turn)
        if cid not in assessed:
            assessed[cid] = {"attempts": 1, "scores": [score], "mean_score": score}
        else:
            assessed[cid]["attempts"] += 1
            assessed[cid]["scores"].append(score)
            assessed[cid]["mean_score"] = sum(assessed[cid]["scores"]) / len(assessed[cid]["scores"])
            
        if len(assessed) >= target_70 and turns_to_70 is None:
            turns_to_70 = turn + 1
            
    coverage_pct = (len(assessed) / len(BENCHMARK_CONCEPTS)) * 100.0
    redundancy_rate = (redundancy_count / TOTAL_TURNS) * 100.0
    unresolved_prereqs = sum(1 for c in BENCHMARK_CONCEPTS if c["prereq_depth"] >= 0.8 and c["id"] not in assessed)
    
    return {
        "coverage_pct": coverage_pct,
        "turns_to_70": turns_to_70 if turns_to_70 else TOTAL_TURNS + 1,
        "redundancy_rate": redundancy_rate,
        "unresolved_prereqs": unresolved_prereqs
    }

# --- Strategy 2: Random Uniform Selection ---
def run_random_selection(profile: dict, seed: int):
    rng = random.Random(seed)
    assessed = {}
    turns_to_70 = None
    target_70 = get_target_70()
    redundancy_count = 0
    
    for turn in range(TOTAL_TURNS):
        cid = rng.choice(BENCHMARK_CONCEPTS)["id"]
        if cid in assessed and assessed[cid]["mean_score"] >= 0.70:
            redundancy_count += 1
            
        score = simulate_student_response(profile[cid], seed * 100 + turn)
        if cid not in assessed:
            assessed[cid] = {"attempts": 1, "scores": [score], "mean_score": score}
        else:
            assessed[cid]["attempts"] += 1
            assessed[cid]["scores"].append(score)
            assessed[cid]["mean_score"] = sum(assessed[cid]["scores"]) / len(assessed[cid]["scores"])
            
        if len(assessed) >= target_70 and turns_to_70 is None:
            turns_to_70 = turn + 1
            
    coverage_pct = (len(assessed) / len(BENCHMARK_CONCEPTS)) * 100.0
    redundancy_rate = (redundancy_count / TOTAL_TURNS) * 100.0
    unresolved_prereqs = sum(1 for c in BENCHMARK_CONCEPTS if c["prereq_depth"] >= 0.8 and c["id"] not in assessed)
    
    return {
        "coverage_pct": coverage_pct,
        "turns_to_70": turns_to_70 if turns_to_70 else TOTAL_TURNS + 1,
        "redundancy_rate": redundancy_rate,
        "unresolved_prereqs": unresolved_prereqs
    }

# --- Strategy 3: Difficulty-Only Adaptive (Local Trapping Baseline) ---
def run_difficulty_only(profile: dict, seed: int):
    rng = random.Random(seed)
    assessed = {}
    turns_to_70 = None
    target_70 = get_target_70()
    redundancy_count = 0
    
    LOCAL_NEIGHBORS = {
        "FastAPI Gateway": ["Asynchronous Event Loops", "PostgreSQL Composite Indexing"],
        "Asynchronous Event Loops": ["FastAPI Gateway"],
        "Docker Containerization": ["Kubernetes Pod Orchestration", "Horizontal Pod Autoscaler"],
        "Kubernetes Pod Orchestration": ["Docker Containerization", "Horizontal Pod Autoscaler"],
        "Horizontal Pod Autoscaler": ["Kubernetes Pod Orchestration", "Docker Containerization"],
        "PostgreSQL Composite Indexing": ["FastAPI Gateway"],
        "BM25Okapi Lexical Retrieval": ["Gemini Dense Embeddings", "Reciprocal Rank Fusion"],
        "Gemini Dense Embeddings": ["BM25Okapi Lexical Retrieval", "Reciprocal Rank Fusion"],
        "Reciprocal Rank Fusion": ["BM25Okapi Lexical Retrieval", "Gemini Dense Embeddings"],
        "Markov Decision Process": ["Deep Q-Network (DQN)"],
        "Deep Q-Network (DQN)": ["Markov Decision Process"],
        "XGBoost Weakness Ensemble": ["FastAPI Gateway"]
    }
    
    current_cid = "FastAPI Gateway"
    
    for turn in range(TOTAL_TURNS):
        cid = current_cid
        if cid in assessed and assessed[cid]["mean_score"] >= 0.70:
            redundancy_count += 1
            
        score = simulate_student_response(profile[cid], seed * 100 + turn)
        if cid not in assessed:
            assessed[cid] = {"attempts": 1, "scores": [score], "mean_score": score}
        else:
            assessed[cid]["attempts"] += 1
            assessed[cid]["scores"].append(score)
            assessed[cid]["mean_score"] = sum(assessed[cid]["scores"]) / len(assessed[cid]["scores"])
            
        if len(assessed) >= target_70 and turns_to_70 is None:
            turns_to_70 = turn + 1
            
        neighbors = LOCAL_NEIGHBORS.get(cid, list(LOCAL_NEIGHBORS.keys()))
        current_cid = rng.choice(neighbors)
            
    coverage_pct = (len(assessed) / len(BENCHMARK_CONCEPTS)) * 100.0
    redundancy_rate = (redundancy_count / TOTAL_TURNS) * 100.0
    unresolved_prereqs = sum(1 for c in BENCHMARK_CONCEPTS if c["prereq_depth"] >= 0.8 and c["id"] not in assessed)
    
    return {
        "coverage_pct": coverage_pct,
        "turns_to_70": turns_to_70 if turns_to_70 else TOTAL_TURNS + 1,
        "redundancy_rate": redundancy_rate,
        "unresolved_prereqs": unresolved_prereqs
    }

# --- Strategy 4: Competitive Baseline — Greedy Knowledge-Gap Alone (S(c) = G_c) ---
def run_greedy_gap_only(profile: dict, seed: int):
    """
    Evaluates selecting solely based on unassessed gap G_c without graph prerequisite
    topological depth P_c, epistemic uncertainty U_c, or recency/redundancy penalty R_c.
    """
    assessed = {}
    turns_to_70 = None
    target_70 = get_target_70()
    redundancy_count = 0
    
    for turn in range(TOTAL_TURNS):
        # Candidate selection based purely on highest gap G_c
        candidate_scores = []
        for c in BENCHMARK_CONCEPTS:
            cid = c["id"]
            if cid not in assessed:
                g_c = 1.0  # Unassessed candidate
            else:
                g_c = max(0.0, 1.0 - assessed[cid]["mean_score"])
            candidate_scores.append((cid, g_c))
            
        # Sort by gap descending (ties broken by concept list order)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        cid = candidate_scores[0][0]
        
        if cid in assessed and assessed[cid]["mean_score"] >= 0.70:
            redundancy_count += 1
            
        score = simulate_student_response(profile[cid], seed * 100 + turn)
        if cid not in assessed:
            assessed[cid] = {"attempts": 1, "scores": [score], "mean_score": score}
        else:
            assessed[cid]["attempts"] += 1
            assessed[cid]["scores"].append(score)
            assessed[cid]["mean_score"] = sum(assessed[cid]["scores"]) / len(assessed[cid]["scores"])
            
        if len(assessed) >= target_70 and turns_to_70 is None:
            turns_to_70 = turn + 1
            
    coverage_pct = (len(assessed) / len(BENCHMARK_CONCEPTS)) * 100.0
    redundancy_rate = (redundancy_count / TOTAL_TURNS) * 100.0
    unresolved_prereqs = sum(1 for c in BENCHMARK_CONCEPTS if c["prereq_depth"] >= 0.8 and c["id"] not in assessed)
    
    return {
        "coverage_pct": coverage_pct,
        "turns_to_70": turns_to_70 if turns_to_70 else TOTAL_TURNS + 1,
        "redundancy_rate": redundancy_rate,
        "unresolved_prereqs": unresolved_prereqs
    }

# --- Strategy 5: Proposed UCS (Knowledge-Gap-Aware Selection) ---
def run_ucs_selection(profile: dict, seed: int):
    engine = UCSEngine(w_gap=0.35, w_prereq=0.25, w_uncertainty=0.25, w_redundancy=0.15)
    engine.load_concept_graph(BENCHMARK_CONCEPTS, BENCHMARK_RELATIONSHIPS)
    
    assessed = {}
    turns_to_70 = None
    target_70 = get_target_70()
    redundancy_count = 0
    
    for turn in range(TOTAL_TURNS):
        decision = engine.select_next_concept(turn)
        cid = decision["selected_concept"]
        
        if cid in assessed and assessed[cid]["mean_score"] >= 0.70:
            redundancy_count += 1
            
        score = simulate_student_response(profile[cid], seed * 100 + turn)
        engine.update_concept_score(cid, score, turn)
        
        if cid not in assessed:
            assessed[cid] = {"attempts": 1, "scores": [score], "mean_score": score}
        else:
            assessed[cid]["attempts"] += 1
            assessed[cid]["scores"].append(score)
            assessed[cid]["mean_score"] = sum(assessed[cid]["scores"]) / len(assessed[cid]["scores"])
            
        if len(assessed) >= target_70 and turns_to_70 is None:
            turns_to_70 = turn + 1
            
    coverage_pct = (len(assessed) / len(BENCHMARK_CONCEPTS)) * 100.0
    redundancy_rate = (redundancy_count / TOTAL_TURNS) * 100.0
    unresolved_prereqs = sum(1 for c in BENCHMARK_CONCEPTS if c["prereq_depth"] >= 0.8 and c["id"] not in assessed)
    
    return {
        "coverage_pct": coverage_pct,
        "turns_to_70": turns_to_70 if turns_to_70 else TOTAL_TURNS + 1,
        "redundancy_rate": redundancy_rate,
        "unresolved_prereqs": unresolved_prereqs
    }

def run_benchmark_e1():
    print("=" * 80)
    print("AIVA RESEARCH BENCHMARK E1: Question Selection Strategy Comparison")
    print(f"Target >=70% coverage threshold: ceil(0.70 * {len(BENCHMARK_CONCEPTS)}) = {get_target_70()} concepts")
    print("=" * 80)
    
    strategies = {
        "Fixed-Order Sequential": run_fixed_order,
        "Random Uniform Selection": run_random_selection,
        "Difficulty-Only (Local Trapping Baseline)": run_difficulty_only,
        "Greedy Knowledge-Gap Alone": run_greedy_gap_only,
        "Proposed UCS (Knowledge-Gap-Aware)": run_ucs_selection
    }
    
    results = {}
    
    for s_name, s_fn in strategies.items():
        cov_all, turns_all, red_all, prereq_all = [], [], [], []
        
        for p_name, profile in STUDENT_PROFILES.items():
            for seed in range(NUM_SEEDS):
                res = s_fn(profile, seed)
                cov_all.append(res["coverage_pct"])
                turns_all.append(res["turns_to_70"])
                red_all.append(res["redundancy_rate"])
                prereq_all.append(res["unresolved_prereqs"])
                
        results[s_name] = {
            "mean_coverage_pct": round(float(np.mean(cov_all)), 2),
            "std_coverage_pct": round(float(np.std(cov_all)), 2),
            "mean_turns_to_70pct": round(float(np.mean(turns_all)), 2),
            "mean_redundancy_rate_pct": round(float(np.mean(red_all)), 2),
            "mean_unresolved_prereqs": round(float(np.mean(prereq_all)), 2)
        }
        
        print(f"\nStrategy: {s_name}")
        print(f"  Concept Coverage (at 10 turns):      {results[s_name]['mean_coverage_pct']:.1f}% ± {results[s_name]['std_coverage_pct']:.1f}%")
        print(f"  Efficiency (Turns to >=70% Coverage): {results[s_name]['mean_turns_to_70pct']:.2f} turns")
        print(f"  Redundancy Rate (% re-asking known):  {results[s_name]['mean_redundancy_rate_pct']:.1f}%")
        print(f"  Unresolved Core Prerequisites (avg):  {results[s_name]['mean_unresolved_prereqs']:.2f}")

    full_output = {
        "benchmark_metadata": {
            "benchmark_name": "E1: Question Selection Strategy Comparison",
            "execution_timestamp": datetime.datetime.now().isoformat(),
            "concepts_evaluated": len(BENCHMARK_CONCEPTS),
            "turns_per_viva": TOTAL_TURNS,
            "target_70pct_concepts": get_target_70(),
            "target_70pct_rule": "math.ceil(0.70 * N_concepts)",
            "seeds_per_profile": NUM_SEEDS,
            "profiles_tested": list(STUDENT_PROFILES.keys()),
            "study_type": "Controlled Monte Carlo MDP Simulation across Defined Student Mastery Archetypes",
            "ucs_weights": {"w_gap": 0.35, "w_prereq": 0.25, "w_uncertainty": 0.25, "w_redundancy": 0.15}
        },
        "comparative_metrics": results
    }
    
    # Save to both AIVA_backend/experiments and root experiments
    backend_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "results_question_selection.json"))
    with open(backend_out, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
        
    root_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "results_question_selection.json"))
    with open(root_out, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
        
    print(f"\nResults successfully written to:")
    print(f"  1. {backend_out}")
    print(f"  2. {root_out}")
    return full_output

if __name__ == "__main__":
    run_benchmark_e1()
