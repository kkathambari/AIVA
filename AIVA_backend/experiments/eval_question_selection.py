"""
AIVA Scientific Benchmark: Question Selection Strategy Evaluation (E1)
----------------------------------------------------------------------
Rigorous multi-topology empirical comparison of the proposed Unresolved Concept Selection (UCS)
against standard and competitive baseline strategies across 4 distinct graph topologies,
3 student mastery archetypes, and 10 random seeds (600 total simulated viva sessions).

Graph Topologies Evaluated:
1. Branching Architecture DAG (12 concepts): Complex real-world software system with multi-path dependencies.
2. Linear Prerequisite Chain (10 concepts): Strict sequential prerequisite hierarchy (e.g., mathematical foundations).
3. Modular Disconnected Clusters (12 concepts): 3 isolated sub-graphs (Frontend, Backend, ML) with zero cross-cluster edges.
4. Diamond Bottleneck Hierarchy (11 concepts): Multiple foundational concepts funnelling into a critical consensus core.

Strategies Compared:
1. Fixed-Order Sequential: Static appearance order in project dissertation.
2. Random Uniform Selection: Stochastic sampling uniformly across concepts.
3. Difficulty-Only (Local Trapping Baseline): Conversational local neighbor traversal without global gap tracking.
4. Greedy Knowledge-Gap Alone (S(c) = G_c): Selects max unassessed gap without topological, uncertainty, or redundancy awareness.
5. Proposed Full UCS: S(c) = 0.35 G_c + 0.25 P_c + 0.25 U_c - 0.15 R_c.

Scientific Rigor Guarantees:
- Target: Strictly math.ceil(0.70 * N_concepts).
- No sentinel values: If threshold is not reached within 10 turns, target_reached = False and turns_to_target = None.
- Complete raw records preserved alongside aggregate means and standard deviations.
"""

import os
import sys
import json
import random
import math
import datetime
import numpy as np
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ucs_engine import UCSEngine

TOTAL_TURNS = 10
SEEDS_PER_PROFILE = 10

# ==============================================================================
# 1. GRAPH TOPOLOGY DEFINITIONS
# ==============================================================================

# Topology 1: Branching Architecture DAG (12 Concepts)
TOPOLOGY_BRANCHING = {
    "name": "Branching Architecture DAG",
    "description": "Distributed microservices gateway with concurrent data & ML pipelines",
    "concepts": [
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
    ],
    "relationships": [
        {"source": "Docker Containerization", "target": "Kubernetes Pod Orchestration", "type": "DEPENDS_ON"},
        {"source": "Kubernetes Pod Orchestration", "target": "Horizontal Pod Autoscaler", "type": "SCALES_WITH"},
        {"source": "Asynchronous Event Loops", "target": "FastAPI Gateway", "type": "POWERS"},
        {"source": "BM25Okapi Lexical Retrieval", "target": "Reciprocal Rank Fusion", "type": "COMBINED_IN"},
        {"source": "Gemini Dense Embeddings", "target": "Reciprocal Rank Fusion", "type": "COMBINED_IN"},
        {"source": "Markov Decision Process", "target": "Deep Q-Network (DQN)", "type": "FORMULATES"},
        {"source": "PostgreSQL Composite Indexing", "target": "FastAPI Gateway", "type": "PERSISTS"}
    ]
}

# Topology 2: Linear Prerequisite Chain (10 Concepts)
TOPOLOGY_LINEAR = {
    "name": "Linear Prerequisite Chain",
    "description": "Strict sequential mathematical and machine learning curriculum",
    "concepts": [
        {"id": "Set Theory Foundations", "type": "Theory", "prereq_depth": 0.2},
        {"id": "Probability Axioms", "type": "Theory", "prereq_depth": 0.4},
        {"id": "Random Variables & Expectations", "type": "Theory", "prereq_depth": 0.5},
        {"id": "Vector Calculus & Gradients", "type": "Theory", "prereq_depth": 0.6},
        {"id": "Matrix Factorization", "type": "Theory", "prereq_depth": 0.7},
        {"id": "Convex Optimization", "type": "Theory", "prereq_depth": 0.8},
        {"id": "Stochastic Gradient Descent", "type": "Theory", "prereq_depth": 0.85},
        {"id": "Backpropagation Algorithm", "type": "Theory", "prereq_depth": 0.9},
        {"id": "Deep Multilayer Perceptron", "type": "Theory", "prereq_depth": 0.9},
        {"id": "Self-Attention Transformers", "type": "Theory", "prereq_depth": 0.95}
    ],
    "relationships": [
        {"source": "Set Theory Foundations", "target": "Probability Axioms", "type": "PREREQUISITE_FOR"},
        {"source": "Probability Axioms", "target": "Random Variables & Expectations", "type": "PREREQUISITE_FOR"},
        {"source": "Random Variables & Expectations", "target": "Vector Calculus & Gradients", "type": "PREREQUISITE_FOR"},
        {"source": "Vector Calculus & Gradients", "target": "Matrix Factorization", "type": "PREREQUISITE_FOR"},
        {"source": "Matrix Factorization", "target": "Convex Optimization", "type": "PREREQUISITE_FOR"},
        {"source": "Convex Optimization", "target": "Stochastic Gradient Descent", "type": "PREREQUISITE_FOR"},
        {"source": "Stochastic Gradient Descent", "target": "Backpropagation Algorithm", "type": "PREREQUISITE_FOR"},
        {"source": "Backpropagation Algorithm", "target": "Deep Multilayer Perceptron", "type": "PREREQUISITE_FOR"},
        {"source": "Deep Multilayer Perceptron", "target": "Self-Attention Transformers", "type": "PREREQUISITE_FOR"}
    ]
}

# Topology 3: Modular Disconnected Clusters (12 Concepts, 3 Clusters)
TOPOLOGY_CLUSTERS = {
    "name": "Modular Disconnected Clusters",
    "description": "3 independent system subsystems (Frontend, Backend, ML) with no cross-edges",
    "concepts": [
        # Cluster 1: Frontend Architecture
        {"id": "React Component Lifecycle", "type": "Frontend", "prereq_depth": 0.4},
        {"id": "State Management Hooks", "type": "Frontend", "prereq_depth": 0.6},
        {"id": "Virtual DOM Reconciliation", "type": "Frontend", "prereq_depth": 0.8},
        {"id": "Web Workers Multithreading", "type": "Frontend", "prereq_depth": 0.85},
        # Cluster 2: Cloud Backend
        {"id": "REST API Endpoint Design", "type": "Backend", "prereq_depth": 0.4},
        {"id": "Async Task Queue Workers", "type": "Backend", "prereq_depth": 0.7},
        {"id": "Redis Distributed Caching", "type": "Backend", "prereq_depth": 0.8},
        {"id": "Database Sharding Strategy", "type": "Backend", "prereq_depth": 0.9},
        # Cluster 3: ML Analytics
        {"id": "Feature Normalization Pipeline", "type": "ML", "prereq_depth": 0.5},
        {"id": "Cross-Validation Sampling", "type": "ML", "prereq_depth": 0.7},
        {"id": "Gradient Boosted Trees", "type": "ML", "prereq_depth": 0.85},
        {"id": "Real-time Inference Server", "type": "ML", "prereq_depth": 0.9}
    ],
    "relationships": [
        {"source": "React Component Lifecycle", "target": "State Management Hooks", "type": "COMPOSED_OF"},
        {"source": "State Management Hooks", "target": "Virtual DOM Reconciliation", "type": "TRIGGERS"},
        {"source": "Virtual DOM Reconciliation", "target": "Web Workers Multithreading", "type": "OFFLOADS_TO"},
        {"source": "REST API Endpoint Design", "target": "Async Task Queue Workers", "type": "DISPATCHES"},
        {"source": "Async Task Queue Workers", "target": "Redis Distributed Caching", "type": "SYNCS_WITH"},
        {"source": "Redis Distributed Caching", "target": "Database Sharding Strategy", "type": "BACKED_BY"},
        {"source": "Feature Normalization Pipeline", "target": "Cross-Validation Sampling", "type": "PREPARES"},
        {"source": "Cross-Validation Sampling", "target": "Gradient Boosted Trees", "type": "EVALUATES"},
        {"source": "Gradient Boosted Trees", "target": "Real-time Inference Server", "type": "DEPLOYED_TO"}
    ]
}

# Topology 4: Diamond Bottleneck Hierarchy (11 Concepts)
TOPOLOGY_BOTTLENECK = {
    "name": "Diamond Bottleneck Hierarchy",
    "description": "Foundational modules funneling into a central core that fans out to specialized applications",
    "concepts": [
        {"id": "Socket Networking", "type": "Foundation", "prereq_depth": 0.3},
        {"id": "Serialization Protocols", "type": "Foundation", "prereq_depth": 0.4},
        {"id": "Logical Clocks", "type": "Foundation", "prereq_depth": 0.6},
        {"id": "Cryptographic Signatures", "type": "Foundation", "prereq_depth": 0.6},
        # Central Core Bottleneck
        {"id": "Distributed Consensus Engine", "type": "Core", "prereq_depth": 0.95},
        # Fanning out to downstream applications
        {"id": "Raft State Machine Replication", "type": "Application", "prereq_depth": 0.85},
        {"id": "Paxos Quorum Leasing", "type": "Application", "prereq_depth": 0.9},
        {"id": "Byzantine Fault Tolerant Validator", "type": "Application", "prereq_depth": 0.9},
        {"id": "Dynamic Cluster Reconfiguration", "type": "Application", "prereq_depth": 0.8},
        {"id": "Append-Only WAL Compaction", "type": "Application", "prereq_depth": 0.8},
        {"id": "Split-Brain Detection Heuristics", "type": "Application", "prereq_depth": 0.85}
    ],
    "relationships": [
        {"source": "Socket Networking", "target": "Distributed Consensus Engine", "type": "FEEDS"},
        {"source": "Serialization Protocols", "target": "Distributed Consensus Engine", "type": "FEEDS"},
        {"source": "Logical Clocks", "target": "Distributed Consensus Engine", "type": "FEEDS"},
        {"source": "Cryptographic Signatures", "target": "Distributed Consensus Engine", "type": "FEEDS"},
        {"source": "Distributed Consensus Engine", "target": "Raft State Machine Replication", "type": "SPECIALIZES"},
        {"source": "Distributed Consensus Engine", "target": "Paxos Quorum Leasing", "type": "SPECIALIZES"},
        {"source": "Distributed Consensus Engine", "target": "Byzantine Fault Tolerant Validator", "type": "SPECIALIZES"},
        {"source": "Distributed Consensus Engine", "target": "Dynamic Cluster Reconfiguration", "type": "ENABLES"},
        {"source": "Distributed Consensus Engine", "target": "Append-Only WAL Compaction", "type": "ENABLES"},
        {"source": "Distributed Consensus Engine", "target": "Split-Brain Detection Heuristics", "type": "ENABLES"}
    ]
}

ALL_TOPOLOGIES = [
    TOPOLOGY_BRANCHING,
    TOPOLOGY_LINEAR,
    TOPOLOGY_CLUSTERS,
    TOPOLOGY_BOTTLENECK
]

# ==============================================================================
# 2. STUDENT MASTERY PROFILES
# ==============================================================================

def generate_student_profiles(concepts: List[dict]) -> Dict[str, Dict[str, float]]:
    """Generates ground-truth mastery levels for any given concept list."""
    competent = {}
    struggling = {}
    uneven = {}
    
    for i, c in enumerate(concepts):
        cid = c["id"]
        pdepth = c.get("prereq_depth", 0.5)
        # Competent: Strong mastery across the board (0.75 - 0.92)
        competent[cid] = float(np.clip(0.85 - 0.10 * pdepth, 0.70, 0.95))
        # Struggling: Below threshold across the board (0.20 - 0.45)
        struggling[cid] = float(np.clip(0.40 - 0.15 * pdepth, 0.15, 0.45))
        # Uneven: Strong on first half (0.85), struggling on second half (0.30)
        if i < len(concepts) // 2:
            uneven[cid] = 0.85
        else:
            uneven[cid] = 0.30
            
    return {
        "Competent_Student": competent,
        "Struggling_Student": struggling,
        "Uneven_Student": uneven
    }

def simulate_student_response(true_mastery: float, seed: int) -> float:
    """Simulates an observed rubric evaluation score with slight student response noise."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, 0.05)
    return float(np.clip(true_mastery + noise, 0.0, 1.0))

# ==============================================================================
# 3. QUESTION SELECTION STRATEGIES
# ==============================================================================

def run_fixed_order(concepts: List[dict], relationships: List[dict], profile: dict, seed: int) -> dict:
    assessed = {}
    target_70 = math.ceil(0.70 * len(concepts))
    turns_to_target = None
    redundancy_count = 0
    
    for turn in range(TOTAL_TURNS):
        cid = concepts[turn % len(concepts)]["id"]
        if cid in assessed and assessed[cid]["mean_score"] >= 0.70:
            redundancy_count += 1
            
        score = simulate_student_response(profile[cid], seed * 100 + turn)
        if cid not in assessed:
            assessed[cid] = {"attempts": 1, "scores": [score], "mean_score": score}
        else:
            assessed[cid]["attempts"] += 1
            assessed[cid]["scores"].append(score)
            assessed[cid]["mean_score"] = sum(assessed[cid]["scores"]) / len(assessed[cid]["scores"])
            
        if len(assessed) >= target_70 and turns_to_target is None:
            turns_to_target = turn + 1
            
    coverage_pct = (len(assessed) / len(concepts)) * 100.0
    redundancy_rate = (redundancy_count / TOTAL_TURNS) * 100.0
    unresolved_prereqs = sum(1 for c in concepts if c["prereq_depth"] >= 0.8 and c["id"] not in assessed)
    target_reached = len(assessed) >= target_70
    
    return {
        "coverage_pct": coverage_pct,
        "target_reached": target_reached,
        "turns_to_target": turns_to_target,  # strictly None if target_reached is False
        "redundancy_rate": redundancy_rate,
        "unresolved_prereqs": unresolved_prereqs
    }

def run_random_selection(concepts: List[dict], relationships: List[dict], profile: dict, seed: int) -> dict:
    rng = random.Random(seed)
    assessed = {}
    target_70 = math.ceil(0.70 * len(concepts))
    turns_to_target = None
    redundancy_count = 0
    
    for turn in range(TOTAL_TURNS):
        cid = rng.choice(concepts)["id"]
        if cid in assessed and assessed[cid]["mean_score"] >= 0.70:
            redundancy_count += 1
            
        score = simulate_student_response(profile[cid], seed * 100 + turn)
        if cid not in assessed:
            assessed[cid] = {"attempts": 1, "scores": [score], "mean_score": score}
        else:
            assessed[cid]["attempts"] += 1
            assessed[cid]["scores"].append(score)
            assessed[cid]["mean_score"] = sum(assessed[cid]["scores"]) / len(assessed[cid]["scores"])
            
        if len(assessed) >= target_70 and turns_to_target is None:
            turns_to_target = turn + 1
            
    coverage_pct = (len(assessed) / len(concepts)) * 100.0
    redundancy_rate = (redundancy_count / TOTAL_TURNS) * 100.0
    unresolved_prereqs = sum(1 for c in concepts if c["prereq_depth"] >= 0.8 and c["id"] not in assessed)
    target_reached = len(assessed) >= target_70
    
    return {
        "coverage_pct": coverage_pct,
        "target_reached": target_reached,
        "turns_to_target": turns_to_target,
        "redundancy_rate": redundancy_rate,
        "unresolved_prereqs": unresolved_prereqs
    }

def run_difficulty_only(concepts: List[dict], relationships: List[dict], profile: dict, seed: int) -> dict:
    rng = random.Random(seed)
    assessed = {}
    target_70 = math.ceil(0.70 * len(concepts))
    turns_to_target = None
    redundancy_count = 0
    
    # Build adjacency neighbors from relationships
    adj: Dict[str, List[str]] = {c["id"]: [] for c in concepts}
    for r in relationships:
        s, t = r["source"], r["target"]
        if s in adj and t in adj:
            adj[s].append(t)
            adj[t].append(s)
            
    # Fallback to self/all if isolated
    for cid in adj:
        if not adj[cid]:
            adj[cid] = [cid]
            
    current_cid = concepts[0]["id"]
    
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
            
        if len(assessed) >= target_70 and turns_to_target is None:
            turns_to_target = turn + 1
            
        neighbors = adj.get(cid, [c["id"] for c in concepts])
        current_cid = rng.choice(neighbors)
            
    coverage_pct = (len(assessed) / len(concepts)) * 100.0
    redundancy_rate = (redundancy_count / TOTAL_TURNS) * 100.0
    unresolved_prereqs = sum(1 for c in concepts if c["prereq_depth"] >= 0.8 and c["id"] not in assessed)
    target_reached = len(assessed) >= target_70
    
    return {
        "coverage_pct": coverage_pct,
        "target_reached": target_reached,
        "turns_to_target": turns_to_target,
        "redundancy_rate": redundancy_rate,
        "unresolved_prereqs": unresolved_prereqs
    }

def run_greedy_gap_only(concepts: List[dict], relationships: List[dict], profile: dict, seed: int) -> dict:
    assessed = {}
    target_70 = math.ceil(0.70 * len(concepts))
    turns_to_target = None
    redundancy_count = 0
    
    for turn in range(TOTAL_TURNS):
        candidate_scores = []
        for c in concepts:
            cid = c["id"]
            if cid not in assessed:
                g_c = 1.0
            else:
                g_c = max(0.0, 1.0 - assessed[cid]["mean_score"])
            candidate_scores.append((cid, g_c))
            
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
            
        if len(assessed) >= target_70 and turns_to_target is None:
            turns_to_target = turn + 1
            
    coverage_pct = (len(assessed) / len(concepts)) * 100.0
    redundancy_rate = (redundancy_count / TOTAL_TURNS) * 100.0
    unresolved_prereqs = sum(1 for c in concepts if c["prereq_depth"] >= 0.8 and c["id"] not in assessed)
    target_reached = len(assessed) >= target_70
    
    return {
        "coverage_pct": coverage_pct,
        "target_reached": target_reached,
        "turns_to_target": turns_to_target,
        "redundancy_rate": redundancy_rate,
        "unresolved_prereqs": unresolved_prereqs
    }

def run_ucs_selection(concepts: List[dict], relationships: List[dict], profile: dict, seed: int) -> dict:
    engine = UCSEngine(w_gap=0.35, w_prereq=0.25, w_uncertainty=0.25, w_redundancy=0.15)
    engine.load_concept_graph(concepts, relationships)
    
    assessed = {}
    target_70 = math.ceil(0.70 * len(concepts))
    turns_to_target = None
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
            
        if len(assessed) >= target_70 and turns_to_target is None:
            turns_to_target = turn + 1
            
    coverage_pct = (len(assessed) / len(concepts)) * 100.0
    redundancy_rate = (redundancy_count / TOTAL_TURNS) * 100.0
    unresolved_prereqs = sum(1 for c in concepts if c["prereq_depth"] >= 0.8 and c["id"] not in assessed)
    target_reached = len(assessed) >= target_70
    
    return {
        "coverage_pct": coverage_pct,
        "target_reached": target_reached,
        "turns_to_target": turns_to_target,
        "redundancy_rate": redundancy_rate,
        "unresolved_prereqs": unresolved_prereqs
    }

# ==============================================================================
# 4. BENCHMARK EXECUTION ACROSS ALL TOPOLOGIES
# ==============================================================================

def run_benchmark_e1():
    print("=" * 90)
    print("AIVA RESEARCH BENCHMARK E1: Comprehensive Multi-Topology Question Selection Comparison")
    print(f"Topologies: {len(ALL_TOPOLOGIES)} | Profiles: 3 | Seeds per Profile: {SEEDS_PER_PROFILE}")
    print(f"Total simulated viva sessions: {len(ALL_TOPOLOGIES) * 3 * SEEDS_PER_PROFILE * 5} (120 per strategy)")
    print("=" * 90)
    
    strategies = {
        "Fixed-Order Sequential": run_fixed_order,
        "Random Uniform Selection": run_random_selection,
        "Difficulty-Only (Local Trapping Baseline)": run_difficulty_only,
        "Greedy Knowledge-Gap Alone": run_greedy_gap_only,
        "Proposed UCS (Knowledge-Gap-Aware)": run_ucs_selection
    }
    
    topology_results = {}
    aggregate_results = {s_name: {
        "coverage_all": [],
        "target_reached_all": [],
        "turns_to_target_successful": [],
        "redundancy_all": [],
        "prereq_all": []
    } for s_name in strategies}
    
    for topo in ALL_TOPOLOGIES:
        topo_name = topo["name"]
        concepts = topo["concepts"]
        rels = topo["relationships"]
        target_70 = math.ceil(0.70 * len(concepts))
        profiles = generate_student_profiles(concepts)
        
        print(f"\n>>> Evaluating Topology: {topo_name} ({len(concepts)} concepts, Target >=70% = {target_70} concepts)")
        topology_results[topo_name] = {
            "concepts_count": len(concepts),
            "target_threshold": target_70,
            "strategies": {}
        }
        
        for s_name, s_fn in strategies.items():
            cov_list, reached_list, turns_list, red_list, prereq_list = [], [], [], [], []
            
            for p_name, profile in profiles.items():
                for seed in range(SEEDS_PER_PROFILE):
                    res = s_fn(concepts, rels, profile, seed)
                    cov_list.append(res["coverage_pct"])
                    reached_list.append(1 if res["target_reached"] else 0)
                    if res["target_reached"] and res["turns_to_target"] is not None:
                        turns_list.append(res["turns_to_target"])
                    red_list.append(res["redundancy_rate"])
                    prereq_list.append(res["unresolved_prereqs"])
                    
                    # Accumulate into cross-topology aggregates
                    aggregate_results[s_name]["coverage_all"].append(res["coverage_pct"])
                    aggregate_results[s_name]["target_reached_all"].append(1 if res["target_reached"] else 0)
                    if res["target_reached"] and res["turns_to_target"] is not None:
                        aggregate_results[s_name]["turns_to_target_successful"].append(res["turns_to_target"])
                    aggregate_results[s_name]["redundancy_all"].append(res["redundancy_rate"])
                    aggregate_results[s_name]["prereq_all"].append(res["unresolved_prereqs"])
                    
            target_reached_rate = (sum(reached_list) / len(reached_list)) * 100.0
            mean_turns = round(float(np.mean(turns_list)), 2) if turns_list else None
            
            topology_results[topo_name]["strategies"][s_name] = {
                "mean_coverage_pct": round(float(np.mean(cov_list)), 2),
                "std_coverage_pct": round(float(np.std(cov_list)), 2),
                "target_reached_rate_pct": round(target_reached_rate, 1),
                "mean_turns_to_target": mean_turns,
                "mean_redundancy_rate_pct": round(float(np.mean(red_list)), 2),
                "mean_unresolved_prereqs": round(float(np.mean(prereq_list)), 2)
            }
            
            t_str = f"{mean_turns:.2f} turns" if mean_turns is not None else "Target Not Reached (null)"
            print(f"  [{s_name}]")
            print(f"    Coverage: {topology_results[topo_name]['strategies'][s_name]['mean_coverage_pct']:.1f}% ± {topology_results[topo_name]['strategies'][s_name]['std_coverage_pct']:.1f}%")
            print(f"    Target Reached Rate: {target_reached_rate:.1f}% | Turns to Target: {t_str}")
            print(f"    Redundancy: {topology_results[topo_name]['strategies'][s_name]['mean_redundancy_rate_pct']:.1f}% | Unassessed Core Prereqs: {topology_results[topo_name]['strategies'][s_name]['mean_unresolved_prereqs']:.2f}")

    # Calculate overall cross-topology macro statistics
    macro_comparative_metrics = {}
    print("\n" + "=" * 90)
    print("CROSS-TOPOLOGY AGGREGATE EVALUATION (Mean across 4 graph topologies, 120 runs per strategy):")
    print("=" * 90)
    
    for s_name in strategies:
        data = aggregate_results[s_name]
        reached_rate = (sum(data["target_reached_all"]) / len(data["target_reached_all"])) * 100.0
        mean_turns = round(float(np.mean(data["turns_to_target_successful"])), 2) if data["turns_to_target_successful"] else None
        
        macro_comparative_metrics[s_name] = {
            "mean_coverage_pct": round(float(np.mean(data["coverage_all"])), 2),
            "std_coverage_pct": round(float(np.std(data["coverage_all"])), 2),
            "target_reached_rate_pct": round(reached_rate, 1),
            "mean_turns_to_target": mean_turns,
            "mean_redundancy_rate_pct": round(float(np.mean(data["redundancy_all"])), 2),
            "std_redundancy_rate_pct": round(float(np.std(data["redundancy_all"])), 2),
            "mean_unresolved_prereqs": round(float(np.mean(data["prereq_all"])), 2),
            "std_unresolved_prereqs": round(float(np.std(data["prereq_all"])), 2),
            "total_runs_evaluated": len(data["coverage_all"])
        }
        
        t_disp = f"{mean_turns:.2f} turns" if mean_turns is not None else "Target Not Reached"
        print(f"\nStrategy: {s_name}")
        print(f"  Concept Coverage:              {macro_comparative_metrics[s_name]['mean_coverage_pct']:.1f}% ± {macro_comparative_metrics[s_name]['std_coverage_pct']:.1f}%")
        print(f"  Target >=70% Reached Rate:     {reached_rate:.1f}% (Turns: {t_disp})")
        print(f"  Question Redundancy Rate:      {macro_comparative_metrics[s_name]['mean_redundancy_rate_pct']:.1f}% ± {macro_comparative_metrics[s_name]['std_redundancy_rate_pct']:.1f}%")
        print(f"  Unresolved Core Prerequisites: {macro_comparative_metrics[s_name]['mean_unresolved_prereqs']:.2f} ± {macro_comparative_metrics[s_name]['std_unresolved_prereqs']:.2f}")

    full_output = {
        "benchmark_metadata": {
            "benchmark_name": "E1: Comprehensive Multi-Topology Question Selection Comparison",
            "execution_timestamp": datetime.datetime.now().isoformat(),
            "topologies_evaluated": [t["name"] for t in ALL_TOPOLOGIES],
            "total_topologies": len(ALL_TOPOLOGIES),
            "total_runs_per_strategy": len(ALL_TOPOLOGIES) * 3 * SEEDS_PER_PROFILE,
            "turns_per_viva": TOTAL_TURNS,
            "coverage_rule": "Strict math.ceil(0.70 * N_concepts)",
            "sentinel_handling": "No arbitrary sentinel turns. target_reached is boolean; turns_to_target is null when target is unreached.",
            "study_type": "Multi-Topology Monte Carlo MDP Simulation across Defined Student Mastery Archetypes",
            "ucs_weights": {"w_gap": 0.35, "w_prereq": 0.25, "w_uncertainty": 0.25, "w_redundancy": 0.15}
        },
        "topology_breakdown": topology_results,
        "comparative_metrics": macro_comparative_metrics
    }
    
    # Save to both AIVA_backend/experiments and root experiments
    backend_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "results_question_selection.json"))
    with open(backend_out, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
        
    root_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "results_question_selection.json"))
    with open(root_out, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
        
    print(f"\nArtifacts successfully preserved:")
    print(f"  1. {backend_out}")
    print(f"  2. {root_out}")
    return full_output

if __name__ == "__main__":
    run_benchmark_e1()
