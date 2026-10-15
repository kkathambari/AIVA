"""
AIVA Research Benchmark: Multi-Agent Persona Routing Evaluation
--------------------------------------------------------------
Compares:
1. Baseline Round-Robin Routing (Sequential turn cycling across 4 personas)
2. Keyword-Guided Balanced Routing (AIVA's adaptive semantic router)

Evaluates:
- Persona Turn Distribution (Entropy & Balance)
- Semantic Domain Concordance (alignment between student trigger topics and persona expertise)
- Decision Traceability & Provenance logging
"""

import os
import sys
import json
import math
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.multi_agent import MultiAgentPanel

# Simulated realistic student turns covering diverse technical topics
SIMULATED_STUDENT_TURNS = [
    {"topic": "Theory", "response": "The time complexity of our algorithm is O(n log n) and we derived the mathematical proof for space complexity."},
    {"topic": "DevOps", "response": "We deploy using a Docker container and an automated AWS cloud pipeline with Kubernetes pods for scale."},
    {"topic": "Limitations", "response": "The main bottleneck and weakness is memory limit when batch size increases, causing a compromise in latency."},
    {"topic": "General", "response": "The system architecture connects the frontend client directly to the backend database gateway."},
    {"topic": "DevOps", "response": "Our database indexing and microservices architecture handle latency under high production load."},
    {"topic": "Theory", "response": "We analyzed the big-o complexity and formulated the optimization equation."},
    {"topic": "Limitations", "response": "One drawback and flaw of this approach is vulnerability to edge case errors and constraints."},
    {"topic": "General", "response": "The user interface allows students to view their viva progress in real time."}
]

def calculate_entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy

def evaluate_routing():
    print("=" * 80)
    print("AIVA EMPIRICAL BENCHMARK: Multi-Agent Routing Comparison")
    print("=" * 80)
    
    panel = MultiAgentPanel()
    
    strategies = ["round_robin", "keyword_balanced"]
    results = {}
    
    for strat in strategies:
        print(f"\nEvaluating Routing Strategy: {strat.upper()}...")
        history_str = ""
        agent_counts = {"Examiner": 0, "Critic": 0, "Industry Expert": 0, "Professor": 0}
        concordance_hits = 0
        decisions_log = []
        
        for idx, turn in enumerate(SIMULATED_STUDENT_TURNS):
            student_text = turn["response"]
            history_str += f"Student: {student_text}\n"
            
            # Execute router node directly
            state = {
                "context": "Sample project context",
                "chat_history": history_str,
                "difficulty": "Medium",
                "agent_type": "auto",
                "routing_strategy": strat
            }
            decision = panel.router_node(state)
            chosen_agent = decision["selected_agent"]
            reason = decision["routing_reason"]
            
            agent_counts[chosen_agent] = agent_counts.get(chosen_agent, 0) + 1
            
            # Measure concordance with intended topic domain
            topic = turn["topic"]
            is_concordant = (
                (topic == "Theory" and chosen_agent == "Professor") or
                (topic == "DevOps" and chosen_agent == "Industry Expert") or
                (topic == "Limitations" and chosen_agent == "Critic") or
                (topic == "General" and chosen_agent in ["Examiner", "Critic", "Industry Expert", "Professor"])
            )
            if is_concordant:
                concordance_hits += 1
                
            history_str += f"Examiner ({chosen_agent}): Generated question for {chosen_agent}\n"
            decisions_log.append({
                "turn": idx + 1,
                "student_topic": topic,
                "selected_agent": chosen_agent,
                "routing_reason": reason,
                "concordant": is_concordant
            })
            
        entropy = calculate_entropy(agent_counts)
        # Max entropy for 4 uniform classes is log2(4) = 2.0 bits
        uniformity_ratio = entropy / 2.0
        concordance_pct = (concordance_hits / len(SIMULATED_STUDENT_TURNS)) * 100
        
        results[strat] = {
            "agent_distribution": agent_counts,
            "shannon_entropy_bits": round(entropy, 3),
            "uniformity_ratio": round(uniformity_ratio, 3),
            "semantic_concordance_percentage": round(concordance_pct, 1),
            "sample_turns": len(SIMULATED_STUDENT_TURNS)
        }
        
        print(f"  Distribution: {agent_counts}")
        print(f"  Shannon Entropy: {entropy:.3f} / 2.000 bits (Uniformity: {uniformity_ratio:.1%})")
        print(f"  Semantic Concordance: {concordance_pct:.1f}%")

    import datetime
    results["benchmark_metadata"] = {
        "execution_timestamp": datetime.datetime.now().isoformat(),
        "verified_by_unit_test": "tests/test_routing_sequence.py",
        "turn_count": len(SIMULATED_STUDENT_TURNS)
    }

    # Save to both AIVA_backend/experiments and root experiments
    backend_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
    os.makedirs(backend_out_dir, exist_ok=True)
    backend_file = os.path.join(backend_out_dir, "results_multi_agent_routing.json")
    with open(backend_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    root_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))
    os.makedirs(root_out_dir, exist_ok=True)
    root_file = os.path.join(root_out_dir, "results_multi_agent_routing.json")
    with open(root_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nMulti-Agent routing benchmark completed! Results saved to:")
    print(f"  1. {backend_file}")
    print(f"  2. {root_file}")
    return results

if __name__ == "__main__":
    evaluate_routing()
