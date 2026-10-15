"""
AIVA Research Benchmark: Human Faculty Calibration & Rubric Agreement Evaluation
---------------------------------------------------------------------------------
Evaluates:
1. Inter-human reliability between two independent faculty evaluators (Rater 1 vs Rater 2)
2. Agreement between Consensus Human Faculty Ratings and AIVA Automated 4-D Rubric

Metrics Evaluated:
- Mean Absolute Error (MAE) on 0.0 - 1.0 continuous scale
- Pearson Correlation Coefficient (r) with two-tailed p-value
- Quadratic Weighted Cohen's Kappa (kappa) discretized into 5 standard academic grade bands
"""

import os
import sys
import json
import datetime
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

def discretize_grade_band(score: float) -> int:
    """Discretizes 0.0 - 1.0 score into 5 standard academic rubric intervals."""
    if score >= 0.85:
        return 4  # Excellent / High Distinction (A)
    elif score >= 0.70:
        return 3  # Good / Commendable (B)
    elif score >= 0.55:
        return 2  # Satisfactory / Average (C)
    elif score >= 0.40:
        return 1  # Marginal Pass (D)
    else:
        return 0  # Fail / Unsatisfactory (F)

def evaluate_rubric_agreement():
    print("=" * 80)
    print("AIVA EMPIRICAL BENCHMARK: Human Faculty Inter-Rater Reliability & Rubric Agreement")
    print("=" * 80)
    
    # Path to benchmark dataset
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmark_data", "pilot_evaluator_dataset.json"))
    if not os.path.exists(dataset_path):
        dataset_path = os.path.abspath(os.path.join("benchmark_data", "pilot_evaluator_dataset.json"))
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    samples = data["samples"]
    provenance = data.get("provenance_and_protocol", {})
    print(f"Loaded {len(samples)} pilot viva response calibration pairs.")
    print(f"Rater 1: {provenance['raters'][0]['title']} ({provenance['raters'][0]['department']})")
    print(f"Rater 2: {provenance['raters'][1]['title']} ({provenance['raters'][1]['department']})")
    
    r1_scores = []
    r2_scores = []
    consensus_scores = []
    aiva_scores = []
    
    dim_maes = {"correctness": [], "completeness": [], "evidence_alignment": [], "reasoning_depth": []}
    
    print("\n{:<10} {:<30} {:<10} {:<10} {:<10} {:<10} {:<8}".format(
        "ID", "Topic", "Rater 1", "Rater 2", "Human Avg", "AIVA", "Diff"
    ))
    print("-" * 92)
    
    for s in samples:
        r1 = s["rater_1_scores"]["composite"]
        r2 = s["rater_2_scores"]["composite"]
        h_consensus = s["consensus_human_scores"]["composite"]
        aiva = s["aiva_system_scores"]["composite"]
        
        r1_scores.append(r1)
        r2_scores.append(r2)
        consensus_scores.append(h_consensus)
        aiva_scores.append(aiva)
        diff = aiva - h_consensus
        
        for dim in dim_maes.keys():
            dim_maes[dim].append(abs(s["aiva_system_scores"][dim] - s["consensus_human_scores"][dim]))
            
        print("{:<10} {:<30} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<+8.4f}".format(
            s["id"], s["topic"][:28], r1, r2, h_consensus, aiva, diff
        ))
        
    r1_arr = np.array(r1_scores)
    r2_arr = np.array(r2_scores)
    h_arr = np.array(consensus_scores)
    a_arr = np.array(aiva_scores)
    
    # 1. Inter-Human Reliability (R1 vs R2)
    inter_human_mae = float(np.mean(np.abs(r1_arr - r2_arr)))
    r_inter_human, p_inter_human = pearsonr(r1_arr, r2_arr)
    r1_disc = [discretize_grade_band(s) for s in r1_arr]
    r2_disc = [discretize_grade_band(s) for s in r2_arr]
    kappa_inter_human = cohen_kappa_score(r1_disc, r2_disc, weights="quadratic")
    
    # 2. Human Consensus vs AIVA Automated Rubric
    mae_aiva = float(np.mean(np.abs(a_arr - h_arr)))
    r_aiva, p_aiva = pearsonr(h_arr, a_arr)
    h_disc = [discretize_grade_band(s) for s in h_arr]
    a_disc = [discretize_grade_band(s) for s in a_arr]
    kappa_aiva = cohen_kappa_score(h_disc, a_disc, weights="quadratic")
    
    print("\n" + "=" * 80)
    print("INTER-HUMAN FACULTY AGREEMENT (Rater 1 vs Rater 2):")
    print(f"  Mean Absolute Difference:            {inter_human_mae:.4f}")
    print(f"  Pearson Correlation Coefficient (r): {r_inter_human:.4f} (p = {p_inter_human:.2e})")
    print(f"  Quadratic Weighted Cohen's Kappa:    {kappa_inter_human:.4f}")
    print("=" * 80)
    print("CONSENSUS HUMAN vs AIVA AUTOMATED RUBRIC:")
    print(f"  Sample Size (N):                     {len(samples)}")
    print(f"  Mean Absolute Error (MAE):           {mae_aiva:.4f}")
    print(f"  Pearson Correlation Coefficient (r): {r_aiva:.4f} (p = {p_aiva:.2e})")
    print(f"  Quadratic Weighted Cohen's Kappa:    {kappa_aiva:.4f}")
    print("\nDimensional MAE Breakdown (vs Human Consensus):")
    for dim, errs in dim_maes.items():
        print(f"  - {dim:<20}: {np.mean(errs):.4f}")
        
    results = {
        "benchmark_name": "AIVA Pilot Faculty Calibration & Inter-Rater Reliability Study",
        "protocol_version": "2.0-verified",
        "execution_timestamp": datetime.datetime.now().isoformat(),
        "sample_size": len(samples),
        "dataset_reference": "benchmark_data/pilot_evaluator_dataset.json",
        "inter_human_reliability": {
            "raters": ["Rater_1 (Assoc. Prof)", "Rater_2 (Asst. Prof)"],
            "mean_absolute_difference": round(inter_human_mae, 4),
            "pearson_correlation": round(float(r_inter_human), 4),
            "pearson_p_value": float(f"{p_inter_human:.2e}"),
            "quadratic_weighted_kappa": round(float(kappa_inter_human), 4)
        },
        "human_consensus_vs_aiva": {
            "mean_absolute_error": round(mae_aiva, 4),
            "pearson_correlation": round(float(r_aiva), 4),
            "pearson_p_value": float(f"{p_aiva:.2e}"),
            "quadratic_weighted_kappa": round(float(kappa_aiva), 4)
        },
        "dimensional_mae": {dim: round(float(np.mean(errs)), 4) for dim, errs in dim_maes.items()},
        "provenance_disclosure": {
            "scoring_procedure": "Double-blind independent faculty scoring across 4-D rubric anchors.",
            "pilot_qualification": "This is a pilot calibration study (N=15) establishing rubric alignment before large-scale multi-institutional cohort trials (N >= 60)."
        }
    }
    
    # Write to AIVA_backend/experiments/
    backend_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
    os.makedirs(backend_out_dir, exist_ok=True)
    backend_file = os.path.join(backend_out_dir, "results_rubric_agreement.json")
    with open(backend_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    # Also write to root experiments/ to prevent path discrepancies
    root_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))
    os.makedirs(root_out_dir, exist_ok=True)
    root_file = os.path.join(root_out_dir, "results_rubric_agreement.json")
    with open(root_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults written to:")
    print(f"  1. {backend_file}")
    print(f"  2. {root_file}")
    return results

if __name__ == "__main__":
    evaluate_rubric_agreement()
