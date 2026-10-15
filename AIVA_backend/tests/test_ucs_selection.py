"""
Unit test asserting the mathematical properties and selection logic of the 
Unresolved Concept Selection (UCS) engine:
    S(c) = w1 * G_c + w2 * P_c + w3 * U_c - w4 * R_c
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ucs_engine import UCSEngine

def test_ucs_initialization_and_weights():
    engine = UCSEngine(w_gap=0.35, w_prereq=0.25, w_uncertainty=0.25, w_redundancy=0.15)
    assert engine.w1 == 0.35
    assert engine.w2 == 0.25
    assert engine.w3 == 0.25
    assert engine.w4 == 0.15
    print("[PASS] UCS initialization and weights verified.")

def test_ucs_selection_and_redundancy():
    engine = UCSEngine()
    
    nodes = [
        {"id": "Docker Containerization", "type": "Architecture", "prereq_depth": 0.9},
        {"id": "Kubernetes Orchestration", "type": "Cloud", "prereq_depth": 0.7},
        {"id": "Horizontal Pod Autoscaler", "type": "DevOps", "prereq_depth": 0.4},
        {"id": "HMAC-SHA256 Authentication", "type": "Security", "prereq_depth": 0.8}
    ]
    relationships = [
        {"source": "Docker Containerization", "target": "Kubernetes Orchestration", "type": "PREREQUISITE_FOR"},
        {"source": "Kubernetes Orchestration", "target": "Horizontal Pod Autoscaler", "type": "CONFIGURES"}
    ]
    
    engine.load_concept_graph(nodes, relationships)
    assert len(engine.concept_states) == 4
    
    # Turn 0: Docker Containerization has highest prerequisite depth (0.9) and unprobed gap (1.0)
    decision_0 = engine.select_next_concept(turn=0)
    selected_0 = decision_0["selected_concept"]
    assert selected_0 == "Docker Containerization", f"Expected 'Docker Containerization', got {selected_0}"
    print(f"[PASS] Turn 0 correctly prioritized root prerequisite '{selected_0}' (S_c={decision_0['score_metrics']['S_c']})")
    
    # Student answers Docker Containerization well (score 0.90)
    engine.update_concept_score(selected_0, score=0.90, turn=0)
    
    # Turn 1: Docker Containerization should have high redundancy penalty R_c and low gap G_c
    metrics_docker = engine.calculate_metrics("Docker Containerization", turn=1)
    assert metrics_docker["R_c"] > 0.0, "Redundancy penalty R_c must be active after being asked"
    assert metrics_docker["G_c"] == 0.1, "Gap G_c should be 1.0 - 0.9 = 0.1"
    
    # Next concept should NOT be Docker Containerization
    decision_1 = engine.select_next_concept(turn=1)
    selected_1 = decision_1["selected_concept"]
    assert selected_1 != "Docker Containerization", "UCS must avoid circular re-asking of mastered concept"
    print(f"[PASS] Turn 1 successfully avoided circular trapping, selecting '{selected_1}' instead of '{selected_0}'")

def test_ucs_struggle_concept_resolution():
    engine = UCSEngine()
    nodes = [
        {"id": "Binary Search Trees", "type": "Algorithm", "prereq_depth": 0.9},
        {"id": "Red-Black Balanced Trees", "type": "Data Structure", "prereq_depth": 0.6}
    ]
    relationships = [
        {"source": "Binary Search Trees", "target": "Red-Black Balanced Trees", "type": "PREREQUISITE_FOR"}
    ]
    engine.load_concept_graph(nodes, relationships)
    
    # Student struggles on Red-Black Trees (score 0.20)
    engine.update_concept_score("Red-Black Balanced Trees", score=0.20, turn=0)
    
    # Turn 1: BST is unresolved and has high prerequisite importance
    decision_1 = engine.select_next_concept(turn=1)
    assert decision_1["selected_concept"] == "Binary Search Trees"
    assert decision_1["recommended_persona"] == "Professor"
    print(f"[PASS] Correctly diagnosed prerequisite and recommended '{decision_1['recommended_persona']}' for '{decision_1['selected_concept']}'")

if __name__ == "__main__":
    test_ucs_initialization_and_weights()
    test_ucs_selection_and_redundancy()
    test_ucs_struggle_concept_resolution()
    print("\n[ALL PASS] All UCS engine unit tests completed successfully.")
