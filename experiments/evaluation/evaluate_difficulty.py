"""
AIVA Research Benchmark: Adaptive Difficulty Controller Subsystem Evaluation
----------------------------------------------------------------------------
Evaluates three difficulty transition controllers:
1. Deep Q-Network (DQN) Reinforcement Learning Controller.
2. Rule-Based State Transition Controller (Thresholding heuristics).
3. Fixed Difficulty Controller (Ablation baseline, Static Level 2).

Evaluated across:
- 4 Student Mastery Profiles (beginner, intermediate, advanced, struggling_foundations).
- 10-turn dynamic trajectory simulations.
- Metrics: Difficulty Appropriateness, Over-Challenging Rate, Under-Challenging Rate,
  Adaptation Latency, and Convergence Error.
"""

import os
import sys
import json
import random
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "AIVA_backend")))
from core.adaptive_engine import AdaptiveEngine


class RuleBasedController:
    """Threshold-based heuristic controller."""
    def get_next_difficulty(self, current_difficulty: int, last_score: float) -> int:
        if last_score >= 0.80:
            return min(4, current_difficulty + 1)
        elif last_score < 0.40:
            return max(0, current_difficulty - 1)
        return current_difficulty


class FixedController:
    """Static ablation controller remaining at constant Medium difficulty (2)."""
    def get_next_difficulty(self, current_difficulty: int, last_score: float) -> int:
        return 2


def evaluate_adaptive_controllers():
    print("=" * 80)
    print("AIVA EMPIRICAL BENCHMARK: Adaptive Difficulty Subsystem Evaluation")
    print("=" * 80)

    adaptive_engine = AdaptiveEngine()
    rule_controller = RuleBasedController()
    fixed_controller = FixedController()

    controllers = {
        "DQN_Reinforcement_Learning": adaptive_engine,
        "Rule_Based_Heuristic": rule_controller,
        "Fixed_Difficulty_Static": fixed_controller
    }

    # Student profiles and their true target difficulty (Zone of Proximal Development)
    # Scale: 0: Fundamentals, 1: Easy, 2: Medium, 3: Hard, 4: Research Level
    profiles = {
        "beginner": {"zpd": 1, "mean_score_offset": -0.20},
        "intermediate": {"zpd": 2, "mean_score_offset": 0.05},
        "advanced": {"zpd": 4, "mean_score_offset": 0.25},
        "struggling_foundations": {"zpd": 0, "mean_score_offset": -0.30}
    }

    SEEDS = list(range(1, 31))  # 30 seeds for statistical validity
    TURNS = 10

    results = {c: {
        "appropriateness": [],
        "over_challenging": [],
        "under_challenging": [],
        "adaptation_latency": [],
        "zpd_tracking_error": []
    } for c in controllers}

    for c_name, controller in controllers.items():
        for p_name, p_data in profiles.items():
            target_zpd = p_data["zpd"]
            offset = p_data["mean_score_offset"]

            for seed in SEEDS:
                rng = random.Random(seed)
                current_diff = 1  # Always start at Easy (Level 1)
                latency = None

                diff_history = [current_diff]
                appropriateness_hits = 0
                over_challs = 0
                under_challs = 0
                tracking_errors = []

                for turn in range(TURNS):
                    # Student score depends on difference between true ZPD and presented difficulty
                    diff_delta = current_diff - target_zpd
                    base_score = 0.65 - (diff_delta * 0.20) + offset
                    score = max(0.05, min(1.0, rng.gauss(base_score, 0.08)))

                    # Track ZPD convergence
                    if abs(current_diff - target_zpd) <= 0.5:
                        appropriateness_hits += 1
                        if latency is None:
                            latency = turn + 1
                    elif current_diff > target_zpd + 1:
                        over_challs += 1
                    elif current_diff < target_zpd - 1:
                        under_challs += 1

                    tracking_errors.append(abs(current_diff - target_zpd))

                    # Next difficulty
                    if c_name == "DQN_Reinforcement_Learning":
                        decision = controller.get_next_difficulty(current_diff, score, controller_type="dqn")
                        next_diff = decision["new_difficulty_level"]
                    elif c_name == "Rule_Based_Heuristic":
                        decision = controller.get_action(current_diff, score) if hasattr(controller, "get_action") else {"new_difficulty_level": controller.get_next_difficulty(current_diff, score)}
                        next_diff = decision["new_difficulty_level"]
                    else:
                        next_diff = 2

                    current_diff = next_diff
                    diff_history.append(current_diff)

                if latency is None:
                    latency = TURNS

                results[c_name]["appropriateness"].append(appropriateness_hits / TURNS)
                results[c_name]["over_challenging"].append(over_challs / TURNS)
                results[c_name]["under_challenging"].append(under_challs / TURNS)
                results[c_name]["adaptation_latency"].append(latency)
                results[c_name]["zpd_tracking_error"].append(float(np.mean(tracking_errors)))

    summary = {}
    for c_name, mdict in results.items():
        summary[c_name] = {
            "difficulty_appropriateness": round(float(np.mean(mdict["appropriateness"])), 4),
            "over_challenging_rate": round(float(np.mean(mdict["over_challenging"])), 4),
            "under_challenging_rate": round(float(np.mean(mdict["under_challenging"])), 4),
            "mean_adaptation_latency_turns": round(float(np.mean(mdict["adaptation_latency"])), 2),
            "mean_zpd_tracking_error": round(float(np.mean(mdict["zpd_tracking_error"])), 4)
        }

    print(f"\n{'Controller Strategy':<30} | {'Appropriateness':<16} | {'Over-Chall':<12} | {'Under-Chall':<12} | {'Latency':<8}")
    print("-" * 85)
    for c_name, s in summary.items():
        print(f"{c_name:<30} | {s['difficulty_appropriateness'] * 100:<15.1f}% | {s['over_challenging_rate'] * 100:<11.1f}% | {s['under_challenging_rate'] * 100:<11.1f}% | {s['mean_adaptation_latency_turns']:<8.2f}")
    print("=" * 85)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "difficulty_evaluation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    root_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results_adaptive_controllers.json"))
    with open(root_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Artifact written to {out_file}")
    return summary


if __name__ == "__main__":
    evaluate_adaptive_controllers()
