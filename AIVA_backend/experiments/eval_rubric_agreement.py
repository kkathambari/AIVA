"""
AIVA Research Benchmark: Human Faculty Calibration & Rubric Agreement Evaluation
---------------------------------------------------------------------------------
Evaluates inter-rater agreement between independent human faculty evaluations
and the automated AIVA four-dimensional viva answer-scoring rubric on the pilot dataset.

Metrics Evaluated:
- Mean Absolute Error (MAE) on 0.0 - 1.0 continuous scale
- Pearson Correlation Coefficient (r)
- Quadratic Weighted Cohen's Kappa (kappa) discretized into 5 grade bands
"""

import os
import sys
import json
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

def discretize_grade_band(score: float) -> int:
    """Discretizes 0.0 - 1.0 score into 5 standard academic rubric intervals."""
    if score >= 0.85:
        return 4  # Excellent (A)
    elif score >= 0.70:
        return 3  # Good (B)
    elif score >= 0.55:
        return 2  # Satisfactory (C)
    elif score >= 0.40:
        return 1  # Marginal Pass (D)
    else:
        return 0  # Fail / Unsatisfactory (F)

def evaluate_rubric_agreement():
    print("=" * 80)
    print("AIVA EMPIRICAL BENCHMARK: Pilot Human-Faculty Agreement Evaluation (N=15)")
    print("=" * 80)
    
    # Path to benchmark dataset
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmark_data", "pilot_evaluator_dataset.json"))
    if not os.path.exists(dataset_path):
        dataset_path = os.path.abspath(os.path.join("benchmark_data", "pilot_evaluator_dataset.json"))
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    samples = data["samples"]
    print(f"Loaded {len(samples)} pilot viva response calibration pairs.")
    
    human_composites = []
    aiva_composites = []
    
    dim_maes = {"correctness": [], "completeness": [], "evidence_alignment": [], "reasoning_depth": []}
    
    print("\n{:<10} {:<32} {:<12} {:<12} {:<8}".format("ID", "Topic", "Human Score", "AIVA Score", "Diff"))
    print("-" * 75)
    
    for s in samples:
        h_score = s["human_faculty_scores"]["composite"]
        a_score = s["aiva_system_scores"]["composite"]
        human_composites.append(h_score)
        aiva_composites.append(a_score)
        diff = a_score - h_score
        
        for dim in dim_maes.keys():
            dim_maes[dim].append(abs(s["aiva_system_scores"][dim] - s["human_faculty_scores"][dim]))
            
        print("{:<10} {:<32} {:<12.4f} {:<12.4f} {:<+8.4f}".format(
            s["id"], s["topic"][:30], h_score, a_score, diff
        ))
        
    h_arr = np.array(human_composites)
    a_arr = np.array(aiva_composites)
    
    mae = float(np.mean(np.abs(a_arr - h_arr)))
    r_val, p_val = pearsonr(h_arr, a_arr)
    
    h_discrete = [discretize_grade_band(s) for s in h_arr]
    a_discrete = [discretize_grade_band(s) for s in a_arr]
    qw_kappa = cohen_kappa_score(h_discrete, a_discrete, weights="quadratic")
    
    print("\n" + "=" * 80)
    print("SUMMARY CALIBRATION METRICS:")
    print(f"  Sample Size (N):                     {len(samples)}")
    print(f"  Mean Absolute Error (MAE):           {mae:.4f}")
    print(f"  Pearson Correlation Coefficient (r): {r_val:.4f} (p = {p_val:.2e})")
    print(f"  Quadratic Weighted Cohen's Kappa:    {qw_kappa:.4f}")
    print("\nDimensional MAE Breakdown:")
    for dim, errs in dim_maes.items():
        print(f"  - {dim:<20}: {np.mean(errs):.4f}")
        
    results = {
        "benchmark_name": "AIVA Pilot Faculty Calibration Study",
        "sample_size": len(samples),
        "protocol_version": "1.0-empirical",
        "composite_metrics": {
            "mean_absolute_error": round(mae, 4),
            "pearson_correlation": round(float(r_val), 4),
            "pearson_p_value": float(f"{p_val:.2e}"),
            "quadratic_weighted_kappa": round(float(qw_kappa), 4)
        },
        "dimensional_mae": {dim: round(float(np.mean(errs)), 4) for dim, errs in dim_maes.items()},
        "dataset_reference": "benchmark_data/pilot_evaluator_dataset.json",
        "notes": "Pilot calibration experiment on N=15 diverse student viva responses. Full multi-faculty cohort validation across institutions is scheduled as an active Phase 2 milestone."
    }
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "results_rubric_agreement.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults successfully written to: {out_file}")
    return results

if __name__ == "__main__":
    evaluate_rubric_agreement()
