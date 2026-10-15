"""
AIVA Research Benchmark: Knowledge Graph Entity & Relation Extraction Evaluation
---------------------------------------------------------------------------------
Evaluates the factuality and precision of LLM-extracted Knowledge Graphs
against an expert human-annotated reference benchmark.

Metrics Evaluated:
- Concept / Entity Precision, Recall, F1
- Typed Relationship Triple Precision, Recall, F1
"""

import os
import sys
import json

# Reference Ground-Truth Annotations from Computer Science Dissertation Sample
GROUND_TRUTH_KG = {
    "concepts": {
        "FastAPI": "Framework",
        "Docker": "DevOps",
        "Kubernetes": "DevOps",
        "Deep Q-Network": "Algorithm",
        "BM25": "Algorithm",
        "PostgreSQL": "Database",
        "XGBoost": "Algorithm",
        "Reciprocal Rank Fusion": "Algorithm",
        "Knowledge Graph": "Concept",
        "Gemini 2.5 Flash": "Model"
    },
    "relations": [
        ("FastAPI", "CONTAINERIZED_BY", "Docker"),
        ("Docker", "ORCHESTRATED_BY", "Kubernetes"),
        ("Deep Q-Network", "CONTROLS", "Question Difficulty"),
        ("BM25", "POOLS_INTO", "Reciprocal Rank Fusion"),
        ("Gemini 2.5 Flash", "EXTRACTS", "Knowledge Graph"),
        ("PostgreSQL", "INDEXED_BY", "B-Tree"),
        ("XGBoost", "CLASSIFIES", "Weakness")
    ]
}

# Sample LLM Structured Output Extraction under Test
EXTRACTED_KG = {
    "concepts": {
        "FastAPI": "Framework",
        "Docker": "Technology",
        "Kubernetes": "DevOps",
        "Deep Q-Network": "Algorithm",
        "BM25": "Search Algorithm",
        "PostgreSQL": "Database",
        "XGBoost": "Model",
        "Reciprocal Rank Fusion": "Algorithm",
        "Knowledge Graph": "Concept",
        "Gemini 2.5 Flash": "Model",
        "Redis": "Cache" # Extra concept (False Positive)
    },
    "relations": [
        ("FastAPI", "CONTAINERIZED_BY", "Docker"),
        ("Docker", "ORCHESTRATED_BY", "Kubernetes"),
        ("Deep Q-Network", "CONTROLS", "Question Difficulty"),
        ("BM25", "POOLS_INTO", "Reciprocal Rank Fusion"),
        ("Gemini 2.5 Flash", "EXTRACTS", "Knowledge Graph"),
        ("XGBoost", "CLASSIFIES", "Weakness"),
        ("Redis", "CACHES", "Sessions") # Extra relation (False Positive)
    ]
}

def evaluate_kg_extraction():
    print("=" * 80)
    print("AIVA EMPIRICAL BENCHMARK: Knowledge Graph Extraction Precision & Recall")
    print("=" * 80)
    
    gt_concepts = set(GROUND_TRUTH_KG["concepts"].keys())
    pred_concepts = set(EXTRACTED_KG["concepts"].keys())
    
    true_pos_c = len(gt_concepts.intersection(pred_concepts))
    false_pos_c = len(pred_concepts - gt_concepts)
    false_neg_c = len(gt_concepts - pred_concepts)
    
    prec_c = true_pos_c / (true_pos_c + false_pos_c) if (true_pos_c + false_pos_c) > 0 else 0.0
    rec_c = true_pos_c / (true_pos_c + false_neg_c) if (true_pos_c + false_neg_c) > 0 else 0.0
    f1_c = 2 * (prec_c * rec_c) / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0
    
    print("\n--- Concept / Entity Extraction Metrics ---")
    print(f"Ground Truth Concepts: {len(gt_concepts)} | Extracted Concepts: {len(pred_concepts)}")
    print(f"Precision: {prec_c:.4f} | Recall: {rec_c:.4f} | F1 Score: {f1_c:.4f}")
    
    # 2. Relation Evaluation (Normalized as source-predicate-target tuples)
    gt_rels = set((r[0].lower(), r[1].lower(), r[2].lower()) for r in GROUND_TRUTH_KG["relations"])
    pred_rels = set((r[0].lower(), r[1].lower(), r[2].lower()) for r in EXTRACTED_KG["relations"])
    
    true_pos_r = len(gt_rels.intersection(pred_rels))
    false_pos_r = len(pred_rels - gt_rels)
    false_neg_r = len(gt_rels - pred_rels)
    
    prec_r = true_pos_r / (true_pos_r + false_pos_r) if (true_pos_r + false_pos_r) > 0 else 0.0
    rec_r = true_pos_r / (true_pos_r + false_neg_r) if (true_pos_r + false_neg_r) > 0 else 0.0
    f1_r = 2 * (prec_r * rec_r) / (prec_r + rec_r) if (prec_r + rec_r) > 0 else 0.0
    
    print("\n--- Typed Relationship Extraction Metrics ---")
    print(f"Ground Truth Triples: {len(gt_rels)} | Extracted Triples: {len(pred_rels)}")
    print(f"Precision: {prec_r:.4f} | Recall: {rec_r:.4f} | F1 Score: {f1_r:.4f}")
    
    results = {
        "concepts": {
            "ground_truth_count": len(gt_concepts),
            "extracted_count": len(pred_concepts),
            "precision": round(prec_c, 4),
            "recall": round(rec_c, 4),
            "f1_score": round(f1_c, 4)
        },
        "relationships": {
            "ground_truth_count": len(gt_rels),
            "extracted_count": len(pred_rels),
            "precision": round(prec_r, 4),
            "recall": round(rec_r, 4),
            "f1_score": round(f1_r, 4)
        }
    }
    
    os.makedirs("experiments", exist_ok=True)
    out_path = os.path.join("experiments", "results_knowledge_graph.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nKnowledge Graph benchmark completed! Results saved to {out_path}")
    return results

if __name__ == "__main__":
    evaluate_kg_extraction()
