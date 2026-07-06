import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
import json

class AnalyticsEngine:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.xgb_model_path = os.path.join(self.model_dir, "weakness_xgb.json")
        self.lr_model_path = os.path.join(self.model_dir, "readiness_lr.joblib")
        
        self.xgb_model = None
        self.lr_model = None
        self._load_models()
        
    def _load_models(self):
        # Load XGBoost model
        if os.path.exists(self.xgb_model_path):
            try:
                self.xgb_model = xgb.XGBClassifier()
                self.xgb_model.load_model(self.xgb_model_path)
                print("AnalyticsEngine: XGBoost weakness detection model loaded successfully.")
            except Exception as e:
                print(f"Error loading XGBoost model: {e}")
                self.xgb_model = None
                
        # Load Logistic Regression model
        if os.path.exists(self.lr_model_path):
            try:
                self.lr_model = joblib.load(self.lr_model_path)
                print("AnalyticsEngine: Logistic Regression readiness model loaded successfully.")
            except Exception as e:
                print(f"Error loading Logistic Regression model: {e}")
                self.lr_model = None

    def calculate_coverage(self, graph_nodes: list, chat_history: str, threshold: float = 0.12) -> dict:
        """
        Module 5: Concept Coverage Analysis (Node-by-Node comparison using TF-IDF and Cosine Similarity)
        """
        if not graph_nodes:
            return {
                "overall_coverage_percent": 0.0,
                "coverage_percentage": 0.0,
                "coverage_map": {},
                "concept_breakdown": {}
            }
            
        if not chat_history.strip():
            return {
                "overall_coverage_percent": 0.0,
                "coverage_percentage": 0.0,
                "coverage_map": {node['id']: 0 for node in graph_nodes},
                "concept_breakdown": {node['id']: 0.0 for node in graph_nodes}
            }
            
        total_concepts = len(graph_nodes)
        tested_concepts = 0
        
        coverage_map = {}
        concept_breakdown = {}
        
        documents = [chat_history.lower()]
        concept_texts = []
        for node in graph_nodes:
            text = f"{node['id']} {node.get('type', '')}".lower()
            concept_texts.append(text)
            
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
            vectorizer.fit(documents + concept_texts)
            chat_vec = vectorizer.transform(documents)
            
            for i, node in enumerate(graph_nodes):
                node_vec = vectorizer.transform([concept_texts[i]])
                sim = float(cosine_similarity(chat_vec, node_vec)[0][0])
                
                concept_breakdown[node['id']] = round(sim * 100, 1)
                
                # If similarity crosses the threshold, mark as covered
                if sim >= threshold:
                    coverage_map[node['id']] = 1  # 1 mention/covered
                    tested_concepts += 1
                else:
                    coverage_map[node['id']] = 0
                    
        except Exception as e:
            print(f"Error in TF-IDF Coverage calculation: {e}")
            # Fallback to string matching
            history_lower = chat_history.lower()
            for node in graph_nodes:
                node_name = node['id'].lower()
                if node_name in history_lower:
                    coverage_map[node['id']] = 1
                    concept_breakdown[node['id']] = 100.0
                    tested_concepts += 1
                else:
                    coverage_map[node['id']] = 0
                    concept_breakdown[node['id']] = 0.0
                    
        overall_coverage = (tested_concepts / total_concepts) * 100.0 if total_concepts > 0 else 0.0
        
        return {
            "overall_coverage_percent": round(overall_coverage, 2),
            "coverage_percentage": round(overall_coverage, 2),
            "coverage_map": coverage_map,
            "concept_breakdown": concept_breakdown
        }

    def detect_weaknesses(self, performance_data: list) -> list:
        """
        Module 6: Weakness Detection Engine using XGBoost Classifier
        performance_data format: [{"concept": "Random Forest", "score": 0.4, "difficulty": 2, "coverage": 0.15, "answer_length": 0.4, "confidence": 0.8}, ...]
        """
        weaknesses = []
        if not performance_data:
            return weaknesses
            
        # Group performance data by concept to aggregate multiple attempts
        concept_performance = {}
        for item in performance_data:
            concept = item.get("concept")
            if not concept:
                continue
            if concept not in concept_performance:
                concept_performance[concept] = []
            concept_performance[concept].append(item)
            
        for concept, attempts in concept_performance.items():
            # Calculate average metrics for this concept
            avg_score = sum(a.get("score", 0.5) for a in attempts) / len(attempts)
            avg_difficulty = sum(a.get("difficulty", 2.0) for a in attempts) / len(attempts)
            avg_coverage = sum(a.get("coverage", 0.5) for a in attempts) / len(attempts)
            avg_length = sum(a.get("answer_length", 0.5) for a in attempts) / len(attempts)
            avg_confidence = sum(a.get("confidence", 0.8) for a in attempts) / len(attempts)
            
            # Predict using XGBoost if loaded
            if self.xgb_model is not None:
                try:
                    features = np.array([[avg_score, avg_difficulty, avg_coverage, avg_length, avg_confidence]], dtype=np.float32)
                    is_weak = self.xgb_model.predict(features)[0]
                    if is_weak == 1:
                        weaknesses.append(concept)
                except Exception as e:
                    print(f"XGBoost weakness prediction error: {e}")
                    # Rule-based fallback
                    if avg_score < 0.6:
                        weaknesses.append(concept)
            else:
                # Rule-based fallback if model is not loaded
                if avg_score < 0.6:
                    weaknesses.append(concept)
                    
        return weaknesses

    def predict_readiness(self, coverage: float, avg_score: float, difficulty_trend: float, weakness_count: int) -> dict:
        """
        Module 7: Viva Readiness Prediction using Logistic Regression Classifier
        """
        probability = 0.5
        
        if self.lr_model is not None:
            try:
                # Features: [avg_score, coverage_pct, difficulty_trend, weakness_count]
                features = np.array([[avg_score, coverage, difficulty_trend, float(weakness_count)]], dtype=np.float32)
                # Predict probability of class 1 (Pass)
                prob_pass = self.lr_model.predict_proba(features)[0][1]
                probability = float(prob_pass)
            except Exception as e:
                print(f"Logistic Regression readiness prediction error: {e}")
                # Fallback to analytical calculation
                probability = avg_score * 0.45 + (coverage / 100.0) * 0.30 + (difficulty_trend + 1) * 0.08 - (weakness_count * 0.05)
                probability = float(np.clip(probability, 0.0, 1.0))
        else:
            # Fallback to analytical calculation
            probability = avg_score * 0.45 + (coverage / 100.0) * 0.30 + (difficulty_trend + 1) * 0.08 - (weakness_count * 0.05)
            probability = float(np.clip(probability, 0.0, 1.0))
            
        probability_pct = round(probability * 100.0, 2)
        
        # Determine level and recommendations
        if probability_pct >= 75.0:
            level = "Ready (Highly Prepared)"
            rec = "Excellent performance. Ready for the actual viva. Focus on maintaining confidence."
        elif probability_pct >= 60.0:
            level = "Borderline (Prepared)"
            rec = "Good grasp of core concepts. Revise the detected weak areas to secure a top grade."
        else:
            level = "Not Ready (Needs Revision)"
            rec = "Significant gaps detected in core concepts. Spend more time in Learn Mode before retaking the viva."
            
        return {
            "probability": probability_pct,
            "readiness_level": level,
            "confidence_score": 0.85 if self.lr_model is not None else 0.70,
            "recommendation": rec
        }
