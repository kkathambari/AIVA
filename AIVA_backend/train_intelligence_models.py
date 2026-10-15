"""
RESEARCH PROVENANCE & METHODOLOGY NOTICE:
------------------------------------------
The models trained via generate_synthetic_data() (XGBoost Weakness Classifier and
Logistic Regression Readiness Predictor) are SYNTHETIC ARCHITECTURAL DEMONSTRATORS.

Labels (weakness and exam readiness) are generated using parameterized synthetic distributions 
and heuristic decision rules. They validate pipeline latency, feature extraction, and API contracts.

CRITICAL RESEARCH QUALIFICATION:
- These models do NOT constitute psychometrically validated predictions of human student outcomes.
- In research manuscripts, these models MUST be explicitly presented as simulation-based demonstrators.
- To produce defensible empirical claims for publication, use train_on_empirical_data() with
  real human evaluator scores and GroupKFold cross-validation grouped by student/project to
  prevent train-test data leakage.
"""

import os
import json
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

IS_SYNTHETIC_SIMULATION = True
PROVENANCE_METADATA = {
    "is_synthetic_simulation": True,
    "provenance": "Rule-parameterized synthetic demonstration model",
    "empirical_validation_status": "Architectural prototype; human benchmark evaluation pending Phase 2 data collection."
}

def generate_synthetic_data(n_samples=1000, seed=42):
    """
    Generates synthetic feature distributions for system latency and architectural verification.
    """
    np.random.seed(seed)
    
    # 1. Weakness Detection Dataset
    # Features: [score, difficulty, coverage, answer_length, confidence]
    scores = np.random.uniform(0.0, 1.0, n_samples)
    difficulties = np.random.uniform(0.0, 4.0, n_samples)  # 0 to 4 difficulty levels
    coverages = np.random.uniform(0.0, 1.0, n_samples)
    answer_lengths = np.random.uniform(0.0, 1.0, n_samples)  # normalized
    confidences = np.random.uniform(0.0, 1.0, n_samples)
    
    X_weakness = np.column_stack((scores, difficulties, coverages, answer_lengths, confidences))
    
    # Generate labels: 1 if weak, 0 if strong
    y_weakness = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        p = 0.05
        if scores[i] < 0.45:
            p += 0.70
        elif scores[i] < 0.65:
            p += 0.35
            
        if difficulties[i] < 1.5:  # Low difficulty
            if scores[i] < 0.7:
                p += 0.20
        
        if answer_lengths[i] < 0.25:
            p += 0.15
            
        y_weakness[i] = 1 if np.random.random() < p else 0

    # 2. Readiness Prediction Dataset
    # Features: [avg_score, coverage_pct, difficulty_trend, weakness_count]
    avg_scores = np.random.uniform(0.1, 1.0, n_samples)
    coverage_pcts = np.random.uniform(10.0, 100.0, n_samples)
    difficulty_trends = np.random.uniform(-1.0, 1.0, n_samples)
    weakness_counts = np.random.randint(0, 8, n_samples)
    
    X_readiness = np.column_stack((avg_scores, coverage_pcts, difficulty_trends, weakness_counts))
    
    y_readiness = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        score_val = (avg_scores[i] * 0.45 + 
                     (coverage_pcts[i] / 100.0) * 0.30 + 
                     (difficulty_trends[i] + 1) * 0.08 - 
                     (weakness_counts[i] * 0.05))
        
        prob = 1.0 / (1.0 + np.exp(-10 * (score_val - 0.45)))
        y_readiness[i] = 1 if np.random.random() < prob else 0
        
    return (X_weakness, y_weakness), (X_readiness, y_readiness)

def train_and_save_synthetic_models():
    """Trains and serializes demonstration models with explicit synthetic provenance metadata."""
    os.makedirs("models", exist_ok=True)
    
    print("=" * 70)
    print("AIVA RESEARCH PIPELINE: Training Synthetic Architectural Models")
    print("NOTE: Qualified as synthetic demonstrators. Real outcomes require empirical labeling.")
    print("=" * 70)
    
    (X_w, y_w), (X_r, y_r) = generate_synthetic_data(n_samples=1500)
    
    # 1. Weakness Detection (XGBoost)
    X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(X_w, y_w, test_size=0.2, random_state=42)
    clf_xgb = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        random_state=42,
        eval_metric="logloss"
    )
    clf_xgb.fit(X_w_train, y_w_train)
    
    preds_w = clf_xgb.predict(X_w_test)
    probs_w = clf_xgb.predict_proba(X_w_test)[:, 1]
    
    print("\n--- Synthetic Weakness Detection Metrics (XGBoost) ---")
    print(f"Accuracy:  {accuracy_score(y_w_test, preds_w):.4f}")
    print(f"F1 Score:  {f1_score(y_w_test, preds_w):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_w_test, probs_w):.4f}")
    
    xgb_model_path = os.path.join("models", "weakness_xgb.json")
    clf_xgb.save_model(xgb_model_path)
    
    # 2. Readiness Prediction (Logistic Regression)
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    clf_lr = LogisticRegression(random_state=42)
    clf_lr.fit(X_r_train, y_r_train)
    
    preds_r = clf_lr.predict(X_r_test)
    probs_r = clf_lr.predict_proba(X_r_test)[:, 1]
    
    print("\n--- Synthetic Readiness Prediction Metrics (Logistic Regression) ---")
    print(f"Accuracy:  {accuracy_score(y_r_test, preds_r):.4f}")
    print(f"F1 Score:  {f1_score(y_r_test, preds_r):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_r_test, probs_r):.4f}")
    
    lr_model_path = os.path.join("models", "readiness_lr.joblib")
    joblib.dump(clf_lr, lr_model_path)
    
    # Save provenance metadata file
    meta_path = os.path.join("models", "model_provenance.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(PROVENANCE_METADATA, f, indent=2)
    print(f"\nModel metadata successfully persisted to {meta_path}")

def train_on_empirical_data(csv_path: str, output_dir: str = "models", group_col: str = "student_id"):
    """
    Template for Phase 2: Ingests human-annotated viva assessment data
    and trains models using GroupKFold cross-validation to prevent inter-candidate leakage.
    """
    import pandas as pd
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Empirical dataset not found at {csv_path}. Please complete Phase 2 annotation first.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded empirical evaluation dataset with {len(df)} samples across {df[group_col].nunique()} distinct {group_col}s.")
    # GroupKFold splits ensuring candidates are partitioned exclusively into train OR test
    gkf = GroupKFold(n_splits=5)
    return gkf

if __name__ == "__main__":
    train_and_save_synthetic_models()

