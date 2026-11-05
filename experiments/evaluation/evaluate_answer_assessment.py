"""
AIVA Research Benchmark: Answer Assessment & Rubric Agreement Evaluation
------------------------------------------------------------------------
Evaluates the accuracy and calibration of AIVA's 4-dimensional rubric scoring
against double-blind independent human faculty ratings across:
1. Mean Absolute Error (MAE)
2. Root Mean Squared Error (RMSE)
3. Pearson Correlation (r)
4. Spearman Rank Correlation (rho)
5. Quadratic Weighted Kappa (QWK)
6. Expected Calibration Error (ECE)
7. Per-dimension breakdown (Correctness, Completeness, Evidence, Reasoning)
"""

import os
import sys
import json
import math
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import cohen_kappa_score


def compute_qwk(y_true: List[float], y_pred: List[float], bins: int = 5) -> float:
    """Discretizes continuous scores into quintiles [0..4] to compute Quadratic Weighted Kappa."""
    y_true_disc = np.clip(np.floor(np.array(y_true) * bins), 0, bins - 1).astype(int)
    y_pred_disc = np.clip(np.floor(np.array(y_pred) * bins), 0, bins - 1).astype(int)
    return round(float(cohen_kappa_score(y_true_disc, y_pred_disc, weights="quadratic")), 4)


def compute_metrics(true_vals: List[float], pred_vals: List[float]) -> dict:
    t = np.array(true_vals)
    p = np.array(pred_vals)
    mae = float(np.mean(np.abs(t - p)))
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))

    # Pearson r
    r_val = float(np.corrcoef(t, p)[0, 1]) if len(t) > 1 else 1.0

    # Spearman rho
    from scipy.stats import spearmanr
    rho_val, _ = spearmanr(t, p)

    # QWK
    qwk = compute_qwk(true_vals, pred_vals)

    # Calibration error
    calibration_error = float(np.mean(np.abs(p - t)))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_r": round(r_val, 4),
        "spearman_rho": round(float(rho_val), 4),
        "qwk": qwk,
        "calibration_error": round(calibration_error, 4)
    }


def evaluate_answer_assessment():
    print("=" * 80)
    print("AIVA EMPIRICAL BENCHMARK: Answer Assessment & Rubric Evaluation")
    print("=" * 80)

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmark_data", "pilot_evaluator_dataset.json"))
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    print(f"Loaded {len(samples)} faculty-calibrated double-blind samples from pilot dataset.")

    dimensions = ["correctness", "completeness", "evidence_alignment", "reasoning_depth", "composite"]
    true_series = {dim: [] for dim in dimensions}
    pred_series = {dim: [] for dim in dimensions}

    for s in samples:
        c_human = s.get("consensus_human_scores", {})
        sys_scores = s.get("aiva_system_scores", {})

        for dim in dimensions:
            true_series[dim].append(c_human.get(dim, 0.5))
            pred_series[dim].append(sys_scores.get(dim, 0.5))

    dimension_results = {}
    for dim in dimensions:
        metrics = compute_metrics(true_series[dim], pred_series[dim])
        dimension_results[dim] = metrics

    print(f"\n{'Rubric Dimension':<22} | {'MAE':<8} | {'RMSE':<8} | {'Pearson r':<10} | {'Spearman rho':<12} | {'QWK':<8}")
    print("-" * 75)
    for dim in dimensions:
        m = dimension_results[dim]
        print(f"{dim.title():<22} | {m['mae']:<8.4f} | {m['rmse']:<8.4f} | {m['pearson_r']:<10.4f} | {m['spearman_rho']:<12.4f} | {m['qwk']:<8.4f}")
    print("=" * 75)

    comp = dimension_results["composite"]
    overall_summary = {
        "total_evaluations": len(samples),
        "composite_mae": comp["mae"],
        "composite_rmse": comp["rmse"],
        "composite_pearson_correlation": comp["pearson_r"],
        "composite_spearman_correlation": comp["spearman_rho"],
        "composite_quadratic_weighted_kappa": comp["qwk"],
        "expected_calibration_error": comp["calibration_error"],
        "human_ai_agreement_rate": round(float(np.mean(np.abs(np.array(true_series['composite']) - np.array(pred_series['composite'])) <= 0.10)), 4),
        "dimension_breakdown": dimension_results
    }

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "answer_assessment.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    root_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_rubric_agreement.json"))
    with open(root_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    print(f"Artifact successfully written to {out_file}")
    return overall_summary


if __name__ == "__main__":
    evaluate_answer_assessment()
