"""
End-to-end live integration test verifying that AIVA's FastAPI examination endpoint
(`/multi_agent_question`) connects to the Unresolved Concept Selection (UCS) engine,
grounds question context in the selected knowledge gap, and logs traceable audit trails.
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app, session, generate_multi_agent_question, MultiAgentRequest, startup_event

def run_e2e_ucs_integration_test():
    print("\n--- Starting End-to-End Live Viva UCS Integration Test ---")
    
    # 1. Initialize backend services
    startup_event()
    session.reset()
    
    # Pre-seed session with representative engineering concept graph
    sample_nodes = [
        {"id": "FastAPI Architecture", "type": "Architecture", "prereq_depth": 0.9},
        {"id": "Asynchronous Endpoints", "type": "Theory", "prereq_depth": 0.8},
        {"id": "Pydantic Validation", "type": "Architecture", "prereq_depth": 0.5}
    ]
    sample_edges = [
        {"source": "FastAPI Architecture", "target": "Asynchronous Endpoints", "type": "POWERS"},
        {"source": "FastAPI Architecture", "target": "Pydantic Validation", "type": "VALIDATES_WITH"}
    ]
    session.graph_nodes = sample_nodes
    session.graph_relationships = sample_edges
    session.ucs_engine.load_concept_graph(sample_nodes, sample_edges)
    
    assert len(session.ucs_engine.concept_states) == 3, "UCS Engine must contain 3 concept states"
    
    # Turn 1: Initial Question Generation via UCS
    req_turn1 = MultiAgentRequest(
        conversation_history=[],
        agent_type="auto",
        difficulty=2,
        routing_strategy="ucs_guided",
        question_selection_strategy="ucs"
    )
    
    # Run async endpoint synchronously
    res_turn1 = asyncio.run(generate_multi_agent_question(req_turn1))
    
    print("\nTurn 1 Generated:")
    print("  Selected Concept: ", res_turn1.get("selected_concept"))
    print("  Routing Strategy: ", res_turn1.get("routing_strategy"))
    print("  Selected Agent:   ", res_turn1.get("agent"))
    print("  Routing Reason:   ", res_turn1.get("routing_reason"))
    print("  UCS Score S(c):   ", res_turn1.get("ucs_metrics", {}).get("S_c"))
    print("  Question Text:    ", res_turn1.get("question")[:80] + "...")
    
    # Verification assertions for Turn 1
    assert res_turn1.get("question_selection_strategy") == "ucs", "Must record UCS question selection strategy"
    assert res_turn1.get("selected_concept") == "FastAPI Architecture", "Must prioritize root prerequisite node"
    assert "S_c" in res_turn1.get("ucs_metrics", {}), "Must return UCS score metrics"
    assert len(session.turn_audit_trail) == 1, "Must log Turn 1 in audit trail"
    assert session.turn_audit_trail[-1]["selected_concept"] == "FastAPI Architecture"
    
    # Turn 2: Student responds to Turn 1 question
    student_response = "We used FastAPI Architecture with asynchronous async def endpoints to handle high concurrency."
    conv_history = [
        {"role": "assistant", "agent": res_turn1.get("agent"), "content": res_turn1.get("question")},
        {"role": "user", "content": student_response}
    ]
    
    req_turn2 = MultiAgentRequest(
        conversation_history=conv_history,
        agent_type="auto",
        difficulty=2,
        routing_strategy="ucs_guided",
        question_selection_strategy="ucs"
    )
    
    res_turn2 = asyncio.run(generate_multi_agent_question(req_turn2))
    
    print("\nTurn 2 Generated:")
    print("  Previous Answer Evaluation Concept: ", res_turn2.get("evaluation", {}).get("concept"))
    print("  Previous Answer Evaluation Score:   ", res_turn2.get("evaluation", {}).get("score"))
    print("  Selected Next Concept:              ", res_turn2.get("selected_concept"))
    print("  UCS Score S(c):                     ", res_turn2.get("ucs_metrics", {}).get("S_c"))
    print("  Question Text:                      ", res_turn2.get("question")[:80] + "...")
    
    # Verification assertions for Turn 2
    assert len(session.turn_audit_trail) == 2, "Must log Turn 2 in audit trail"
    assert res_turn2.get("selected_concept") != "FastAPI Architecture", "Must advance to an unresolved concept without circular repetition"
    
    summary = session.ucs_engine.get_coverage_summary()
    print("\nUCS Coverage Summary after 2 turns:")
    print("  Assessed Count:   ", summary.get("assessed_count"))
    print("  Coverage Percent: ", summary.get("coverage_pct"), "%")
    assert summary.get("assessed_count") >= 1, "Must have recorded concept assessments in UCS"
    
    print("\n[PASS] End-to-End Live Viva UCS Integration Test SUCCEEDED!")

if __name__ == "__main__":
    run_e2e_ucs_integration_test()
