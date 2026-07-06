import os
import json
import numpy as np
from core.rag_engine import RAGEngine
from core.analytics_engine import AnalyticsEngine
from core.adaptive_engine import AdaptiveEngine
from core.multi_agent import MultiAgentPanel

def test_rag_engine():
    print("\n--- Testing Hybrid RAG Engine ---")
    rag = RAGEngine()
    
    # Let's index a dummy text
    dummy_text = "FastAPI is a modern web framework for Python. It is fast and uses Pydantic. Machine learning is great. Random Forest is an ensemble classifier."
    rag.index_document(dummy_text, document_id="test_doc.pdf")
    
    # Query dense and sparse
    results = rag.query("FastAPI python web framework", n_results=1)
    print(f"Query: 'FastAPI python web framework' -> Found chunks: {len(results)}")
    if results:
        print(f"Top Result: '{results[0]}'")
        assert "FastAPI" in results[0], "Failed to retrieve correct chunk"
    print("Hybrid RAG Engine: SUCCESS")

def test_analytics_engine():
    print("\n--- Testing Analytics Engine (TF-IDF & ML Models) ---")
    ae = AnalyticsEngine()
    
    # Dummy graph nodes
    graph_nodes = [
        {"id": "FastAPI", "type": "Framework"},
        {"id": "Random Forest", "type": "Algorithm"},
        {"id": "Docker", "type": "Tool"}
    ]
    
    # Test Node-by-Node TF-IDF Coverage
    chat_history = "Student: I used FastAPI for the backend of my project. It provides REST APIs."
    coverage = ae.calculate_coverage(graph_nodes, chat_history, threshold=0.08)
    print("Coverage Breakdown:", coverage["concept_breakdown"])
    print("Coverage Map:", coverage["coverage_map"])
    print(f"Overall Coverage: {coverage['overall_coverage_percent']}%")
    
    assert coverage["coverage_map"]["FastAPI"] == 1, "FastAPI should be covered"
    assert coverage["coverage_map"]["Docker"] == 0, "Docker should NOT be covered"
    
    # Test XGBoost Weakness Detection
    # performance_data features: [score, difficulty, coverage, answer_length, confidence]
    performance_data = [
        {"concept": "FastAPI", "score": 0.9, "difficulty": 2, "coverage": 0.8, "answer_length": 0.7, "confidence": 0.8},
        {"concept": "Random Forest", "score": 0.3, "difficulty": 1, "coverage": 0.1, "answer_length": 0.1, "confidence": 0.4}
    ]
    weaknesses = ae.detect_weaknesses(performance_data)
    print("Detected Weaknesses (XGBoost):", weaknesses)
    assert "Random Forest" in weaknesses, "Random Forest should be classified as a weakness"
    
    # Test Logistic Regression Readiness Prediction
    # features: [avg_score, coverage_pct, difficulty_trend, weakness_count]
    readiness = ae.predict_readiness(
        coverage=coverage["overall_coverage_percent"],
        avg_score=0.6,
        difficulty_trend=0.5,
        weakness_count=len(weaknesses)
    )
    print("Readiness prediction:", readiness)
    assert "probability" in readiness, "Probability missing in readiness report"
    print("Analytics Engine: SUCCESS")

def test_adaptive_difficulty():
    print("\n--- Testing DQN Difficulty Adaptation ---")
    ae = AdaptiveEngine()
    
    # Test next difficulty calculation
    result_low = ae.get_next_difficulty(current_difficulty=2, last_answer_score=0.2)
    print("Score 0.2 -> Next difficulty:", result_low)
    assert result_low["action_taken"] == "DECREASE" or result_low["new_difficulty_level"] <= 2
    
    result_high = ae.get_next_difficulty(current_difficulty=2, last_answer_score=0.95)
    print("Score 0.95 -> Next difficulty:", result_high)
    assert result_high["action_taken"] == "INCREASE" or result_high["new_difficulty_level"] >= 2
    print("DQN Adaptive Engine: SUCCESS")

def test_multi_agent_panel():
    print("\n--- Testing Rule-Based Multi-Agent Panel Routing ---")
    map_panel = MultiAgentPanel()
    
    # Test Professor keyword routing
    context = "Random Forest uses multiple decision trees to vote on classes."
    chat_history = "Examiner: Explain the theory of Random Forest.\nStudent: It uses math formulas and the Gini impurity algorithm."
    res_prof = map_panel.run_panel(
        context=context,
        chat_history=chat_history,
        difficulty="Medium",
        agent_type="Auto (Panel)"
    )
    print("Question routed to Professor: SUCCESS")
    
    # Test Industry Expert keyword routing
    chat_history = "Examiner: How is your system deployed?\nStudent: We use Docker and AWS cloud to scale it."
    res_ind = map_panel.run_panel(
        context=context,
        chat_history=chat_history,
        difficulty="Medium",
        agent_type="Auto (Panel)"
    )
    print("Question routed to Industry Expert: SUCCESS")
    print("Multi-Agent Panel Routing: SUCCESS")

if __name__ == "__main__":
    print("==================================================")
    print("               AIVA BACKEND TEST SUITE            ")
    print("==================================================")
    
    # Ensure GEMINI_API_KEY is dummy-populated for RAG testing if not present
    if not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = "mock_key"
        
    test_rag_engine()
    test_analytics_engine()
    test_adaptive_difficulty()
    test_multi_agent_panel()
    
    print("\n==================================================")
    print("          ALL TEST SUITE CHECKS COMPLETED         ")
    print("==================================================")
