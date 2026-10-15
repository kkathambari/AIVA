"""
Unit test asserting the strict four-persona sequence for baseline round-robin routing
and testing keyword routing behavior against labelled test cases.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.multi_agent import MultiAgentPanel

def test_round_robin_sequence():
    panel = MultiAgentPanel()
    expected_sequence = ["Examiner", "Critic", "Industry Expert", "Professor", "Examiner", "Critic", "Industry Expert", "Professor"]
    actual_sequence = []
    
    history_str = ""
    for step in range(8):
        history_str += f"Student: This is student answer #{step+1}.\n"
        state = {
            "context": "Context",
            "chat_history": history_str,
            "difficulty": "Medium",
            "agent_type": "auto",
            "routing_strategy": "round_robin"
        }
        decision = panel.router_node(state)
        agent = decision["selected_agent"]
        actual_sequence.append(agent)
        history_str += f"Examiner ({agent}): Question #{step+1}\n"
        
    print("Expected Round-Robin Sequence:", expected_sequence)
    print("Actual Round-Robin Sequence:  ", actual_sequence)
    assert actual_sequence == expected_sequence, f"Round-robin sequence mismatch! Expected {expected_sequence}, got {actual_sequence}"
    print("[PASS] Round-robin sequence unit test succeeded!")

def test_keyword_routing_accuracy():
    panel = MultiAgentPanel()
    test_cases = [
        ("We analyzed the time complexity and mathematical formula of the big-o bound.", "Professor"),
        ("We deployed Docker containers into AWS cloud Kubernetes pods with low latency.", "Industry Expert"),
        ("The primary drawback, bottleneck, and flaw is memory constraint under peak load.", "Critic")
    ]
    
    for student_msg, expected_agent in test_cases:
        history_str = f"Student: {student_msg}\n"
        state = {
            "context": "Context",
            "chat_history": history_str,
            "difficulty": "Medium",
            "agent_type": "auto",
            "routing_strategy": "keyword_balanced"
        }
        decision = panel.router_node(state)
        assert decision["selected_agent"] == expected_agent, f"Expected {expected_agent}, got {decision['selected_agent']}"
        print(f"[PASS] Keyword routing correctly identified {expected_agent} for: '{student_msg[:45]}...'")

if __name__ == "__main__":
    test_round_robin_sequence()
    test_keyword_routing_accuracy()
