import os
import json
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def generate_synthetic_data(n_samples=1000, seed=42):
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
    # Logic: weak if low score, or medium score on low difficulty, or low answer length with low score
    y_weakness = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # Base probability of being weak
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
    # trend: slope of difficulty, between -1.0 and 1.0
    difficulty_trends = np.random.uniform(-1.0, 1.0, n_samples)
    weakness_counts = np.random.randint(0, 8, n_samples)
    
    X_readiness = np.column_stack((avg_scores, coverage_pcts, difficulty_trends, weakness_counts))
    
    # Generate labels: 1 if ready (pass), 0 if not ready (fail)
    # Logic: high average score, high coverage, positive difficulty trend, fewer weaknesses
    y_readiness = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        score_val = (avg_scores[i] * 0.45 + 
                     (coverage_pcts[i] / 100.0) * 0.30 + 
                     (difficulty_trends[i] + 1) * 0.08 - 
                     (weakness_counts[i] * 0.05))
        
        # Sigmoid probability
        prob = 1.0 / (1.0 + np.exp(-10 * (score_val - 0.45)))
        y_readiness[i] = 1 if np.random.random() < prob else 0
        
    return (X_weakness, y_weakness), (X_readiness, y_readiness)

def train_and_save_models():
    os.makedirs("models", exist_ok=True)
    
    print("Generating synthetic datasets...")
    (X_w, y_w), (X_r, y_r) = generate_synthetic_data(n_samples=1500)
    
    # --- 1. Train Weakness Detection (XGBoost) ---
    print("\n--- Training Weakness Detection Model (XGBoost) ---")
    X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(X_w, y_w, test_size=0.2, random_state=42)
    
    # Train XGBClassifier
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
    
    print(f"Accuracy:  {accuracy_score(y_w_test, preds_w):.4f}")
    print(f"Precision: {precision_score(y_w_test, preds_w):.4f}")
    print(f"Recall:    {recall_score(y_w_test, preds_w):.4f}")
    print(f"F1 Score:  {f1_score(y_w_test, preds_w):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_w_test, probs_w):.4f}")
    
    xgb_model_path = os.path.join("models", "weakness_xgb.json")
    clf_xgb.save_model(xgb_model_path)
    print(f"XGBoost model saved to {xgb_model_path}")
    
    # --- 2. Train Readiness Prediction (Logistic Regression) ---
    print("\n--- Training Readiness Prediction Model (Logistic Regression) ---")
    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    
    clf_lr = LogisticRegression(random_state=42)
    clf_lr.fit(X_r_train, y_r_train)
    
    preds_r = clf_lr.predict(X_r_test)
    probs_r = clf_lr.predict_proba(X_r_test)[:, 1]
    
    print(f"Accuracy:  {accuracy_score(y_r_test, preds_r):.4f}")
    print(f"Precision: {precision_score(y_r_test, preds_r):.4f}")
    print(f"Recall:    {recall_score(y_r_test, preds_r):.4f}")
    print(f"F1 Score:  {f1_score(y_r_test, preds_r):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_r_test, probs_r):.4f}")
    
    lr_model_path = os.path.join("models", "readiness_lr.joblib")
    joblib.dump(clf_lr, lr_model_path)
    print(f"Logistic Regression model saved to {lr_model_path}")

if __name__ == "__main__":
    train_and_save_models()
