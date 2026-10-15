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
        
        self.is_synthetic_simulation = True
        self.model_provenance = "Trained on synthetic rule-parameterized distributions for architectural and latency verification."
        
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

    @staticmethod
    def calculate_rubric_composite(
        correctness: float,
        completeness: float,
        evidence_alignment: float,
        reasoning_depth: float
    ) -> float:
        """
        Calculates composite answer quality score based on 4-dimensional rubric:
        - Correctness (35%): Technical veracity of the explanation.
        - Completeness (25%): Covers essential core facets of the question.
        - Evidence Alignment (20%): Supported by facts/design in the project report.
        - Reasoning Depth (20%): Articulates 'why', design justifications, and trade-offs.
        """
        composite = (
            0.35 * correctness +
            0.25 * completeness +
            0.20 * evidence_alignment +
            0.20 * reasoning_depth
        )
        return round(float(np.clip(composite, 0.0, 1.0)), 4)

    @staticmethod
    def compute_evaluator_agreement(human_scores: list, model_scores: list) -> dict:
        """
        Computes inter-rater reliability metrics between LLM rubric ratings and human evaluator ground-truth:
        - Mean Absolute Error (MAE)
        - Pearson correlation coefficient (r)
        - Quadratic weighted Cohen's Kappa
        """
        if len(human_scores) == 0 or len(human_scores) != len(model_scores):
            return {"error": "Mismatched or empty score lists for agreement computation."}

        h = np.array(human_scores, dtype=np.float64)
        m = np.array(model_scores, dtype=np.float64)

        mae = float(np.mean(np.abs(h - m)))
        
        # Pearson correlation
        if np.std(h) > 1e-9 and np.std(m) > 1e-9:
            pearson_r = float(np.corrcoef(h, m)[0, 1])
        else:
            pearson_r = 1.0 if np.allclose(h, m) else 0.0

        # Discretize into 5 bins [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] for Cohen's Kappa
        from sklearn.metrics import cohen_kappa_score
        h_bins = np.clip(np.digitize(h, bins=[0.2, 0.4, 0.6, 0.8]) , 0, 4)
        m_bins = np.clip(np.digitize(m, bins=[0.2, 0.4, 0.6, 0.8]), 0, 4)
        kappa = float(cohen_kappa_score(h_bins, m_bins, weights="quadratic"))

        return {
            "sample_size": len(human_scores),
            "mae": round(mae, 4),
            "pearson_r": round(pearson_r, 4),
            "quadratic_weighted_kappa": round(kappa, 4)
        }

    def calculate_coverage(self, graph_nodes: list, chat_history: str, threshold: float = 0.12) -> dict:
        """
        Module 5: Lexical & Semantic Concept Mention Coverage Analysis (TF-IDF overlap against Knowledge Graph entities).
        NOTE: This measures topical lexical coverage across utterances, NOT psychometrically validated comprehension.
        """
        qualification_note = "Measures topical lexical/semantic mention coverage against KG concept nodes, not validated student comprehension."
        if not graph_nodes:
            return {
                "overall_coverage_percent": 0.0,
                "coverage_percentage": 0.0,
                "coverage_map": {},
                "concept_breakdown": {},
                "metric_type": "lexical_semantic_mention_coverage",
                "qualification": qualification_note
            }
            
        if not chat_history.strip():
            return {
                "overall_coverage_percent": 0.0,
                "coverage_percentage": 0.0,
                "coverage_map": {node['id']: 0 for node in graph_nodes},
                "concept_breakdown": {node['id']: 0.0 for node in graph_nodes},
                "metric_type": "lexical_semantic_mention_coverage",
                "qualification": qualification_note
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
            print(f"Error in TF-IDF Coverage calculation: {repr(e)}")
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
            "concept_breakdown": concept_breakdown,
            "metric_type": "lexical_semantic_mention_coverage",
            "qualification": "Measures topical lexical/semantic mention coverage against KG concept nodes, not validated student comprehension."
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
            
            is_weak = False
            risk_probability = max(0.05, min(0.99, 1.0 - avg_score))
            
            # Predict using XGBoost if loaded
            if self.xgb_model is not None:
                try:
                    features = np.array([[avg_score, avg_difficulty, avg_coverage, avg_length, avg_confidence]], dtype=np.float32)
                    pred = self.xgb_model.predict(features)[0]
                    is_weak = (pred == 1)
                    if hasattr(self.xgb_model, "predict_proba"):
                        proba = self.xgb_model.predict_proba(features)[0]
                        if len(proba) > 1:
                            risk_probability = float(proba[1])
                except Exception as e:
                    print(f"XGBoost weakness prediction error: {repr(e)}")
                    is_weak = (avg_score < 0.6)
            else:
                # Rule-based fallback if model is not loaded
                is_weak = (avg_score < 0.6)
                
            if is_weak:
                risk_pct = round(risk_probability * 100, 1)
                severity = "Critical Risk" if risk_pct >= 80 else ("High Risk" if risk_pct >= 60 else "Moderate Concern")
                reason = "Turn evaluation score below pass threshold" if avg_score < 0.4 else "Incomplete explanation depth and technical coverage"
                
                weaknesses.append({
                    "concept": concept,
                    "risk_probability": risk_pct,
                    "average_score": round(avg_score * 100, 1),
                    "difficulty_level": round(avg_difficulty, 1),
                    "coverage": round(avg_coverage * 100, 1),
                    "attempts": len(attempts),
                    "severity": severity,
                    "reason": reason,
                    "recommendation": f"Review key algorithms, equations, and implementation details of {concept}.",
                    "is_synthetic_simulation": self.is_synthetic_simulation,
                    "provenance": self.model_provenance
                })
                    
        return weaknesses

    def analyze_fluency(self, answer_text: str) -> dict:
        """
        Module 8: Fluency, Filler-Word & Repetitive Word Analyzer
        Counts occurrences of unwanted filler words and repetitive word phrases to compute fluency.
        """
        if not answer_text or not answer_text.strip():
            return {
                "fluency_score": 100.0,
                "filler_words_used": {},
                "repetitive_words_used": {},
                "total_filler_count": 0,
                "total_repetitive_count": 0,
                "repetition_penalty": 0.0,
                "feedback": "No answer provided."
            }
            
        import re
        filler_words = ["like", "so", "basically", "literally", "umm", "uh", "you know", "i mean", "actually", "just", "well", "right"]
        text_lower = answer_text.lower()
        cleaned_text = re.sub(r'[^\w\s]', ' ', text_lower)
        words = cleaned_text.split()
        total_words = len(words)
        
        if total_words == 0:
            return {
                "fluency_score": 100.0,
                "filler_words_used": {},
                "repetitive_words_used": {},
                "total_filler_count": 0,
                "total_repetitive_count": 0,
                "repetition_penalty": 0.0,
                "feedback": "No answer provided."
            }
            
        # 1. Detect Filler Words
        filler_counts = {}
        total_filler_count = 0
        for fw in filler_words:
            if " " in fw:
                count = text_lower.count(fw)
            else:
                count = words.count(fw)
                
            if count > 0:
                filler_counts[fw] = count
                total_filler_count += count

        # 2. Detect Repetitive Words
        # A) Consecutive word repeats (e.g. "we we", "is is")
        consecutive_repeats = re.findall(r'\b(\w+)\s+\1\b', text_lower)
        
        # B) High-frequency over-used non-stop words (repeated >= 3 times in a single turn)
        stopwords = {
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
            "is", "it", "this", "that", "are", "was", "were", "as", "by", "from", "be"
        }
        word_freq = {}
        for w in words:
            if len(w) > 2 and w not in stopwords:
                word_freq[w] = word_freq.get(w, 0) + 1
                
        repetitive_counts = {}
        for w in consecutive_repeats:
            repetitive_counts[w] = repetitive_counts.get(w, 0) + 1
            
        for w, count in word_freq.items():
            if count >= 3 and w not in filler_counts:
                repetitive_counts[w] = count
                
        total_repetitive_count = sum(repetitive_counts.values())
        
        # Penalties:
        # Filler penalty: 1.8 points for every 1% of words that are fillers
        filler_ratio = total_filler_count / total_words
        filler_penalty = (filler_ratio * 100) * 1.8
        
        # Repetition penalty: 2.0 points per repetition percentage
        repetition_ratio = total_repetitive_count / total_words
        repetition_penalty = (repetition_ratio * 100) * 2.0
        
        total_penalty = filler_penalty + repetition_penalty
        fluency_score = max(0.0, min(100.0, 100.0 - total_penalty))
        
        feedback_parts = []
        if total_filler_count > 0:
            feedback_parts.append(f"{total_filler_count} filler words")
        if total_repetitive_count > 0:
            feedback_parts.append(f"{total_repetitive_count} repetitive words")
            
        if fluency_score >= 88:
            feedback = "Excellent fluency and clear vocabulary articulation."
        elif fluency_score >= 70:
            feedback = f"Good communication, but hesitation detected: {'; '.join(feedback_parts)}."
        else:
            feedback = f"Frequent repetitions and filler words detected: {'; '.join(feedback_parts)}. Focus on concise phrasing."
            
        return {
            "fluency_score": round(fluency_score, 1),
            "filler_words_used": filler_counts,
            "repetitive_words_used": repetitive_counts,
            "total_filler_count": total_filler_count,
            "total_repetitive_count": total_repetitive_count,
            "repetition_penalty": round(repetition_penalty, 1),
            "feedback": feedback
        }

    def predict_readiness(self, coverage: float, avg_score: float, difficulty_trend: float, weakness_count: int, fluency_score: float = 100.0) -> dict:
        """
        Module 7: Viva Readiness Prediction using Logistic Regression Classifier
        Incorporates average score, coverage, difficulty trend, detected weaknesses, and verbal fluency/repetition penalties.
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
                print(f"Logistic Regression readiness prediction error: {repr(e)}")
                # Fallback to analytical calculation
                probability = avg_score * 0.45 + (coverage / 100.0) * 0.30 + (difficulty_trend + 1) * 0.08 - (weakness_count * 0.05)
                probability = float(np.clip(probability, 0.0, 1.0))
        else:
            # Fallback to analytical calculation
            probability = avg_score * 0.45 + (coverage / 100.0) * 0.30 + (difficulty_trend + 1) * 0.08 - (weakness_count * 0.05)
            probability = float(np.clip(probability, 0.0, 1.0))
            
        # Apply verbal fluency & repetitive words penalty to preparedness score (up to 12% reduction for severe hesitation/repetition)
        fluency_penalty_pct = 0.0
        if fluency_score < 90.0:
            fluency_deduction = ((90.0 - fluency_score) / 90.0) * 0.12
            probability = max(0.0, probability - fluency_deduction)
            fluency_penalty_pct = round(fluency_deduction * 100, 1)

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
            "recommendation": rec,
            "fluency_penalty_pct": fluency_penalty_pct,
            "is_synthetic_simulation": self.is_synthetic_simulation,
            "provenance": self.model_provenance
        }
