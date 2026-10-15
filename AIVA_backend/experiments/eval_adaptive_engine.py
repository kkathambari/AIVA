"""
AIVA Research Benchmark: Adaptive Controller Evaluation Suite
------------------------------------------------------------
Compares three difficulty adaptation controllers:
1. Fixed Difficulty Baseline (Constant level 2)
2. Rule-Based Adaptation Baseline (Threshold-based policy: <0.4 dec, >0.7 inc)
3. Deep Q-Network (DQN) Controller

Simulates performance across three standardized student skill profiles:
- Low Skill (skill ~ N(0.30, 0.05))
- Medium Skill (skill ~ N(0.60, 0.05))
- High Skill (skill ~ N(0.85, 0.05))

Evaluates:
- Mean Student Score
- Flow Band Ratio (percentage of steps inside target learning zone [0.40, 0.70])
- Difficulty Transitions (measure of policy volatility/stability)
- Mean Cumulative Reward
"""

import os
import sys
import json
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.adaptive_engine import AdaptiveEngine, FixedDifficultyController, RuleBasedController, VivaEnv

def simulate_viva_session(controller_type: str, student_skill: float, engine: AdaptiveEngine, seed: int, steps: int = 20):
    np.random.seed(seed)
    
    current_difficulty = 1  # Start at Easy
    scores = []
    flow_steps = 0
    transitions = 0
    cumulative_reward = 0.0
    
    last_score = 0.5
    for step in range(steps):
        # 1. Controller chooses action based on current state
        decision = engine.get_next_difficulty(current_difficulty, last_score, controller_type=controller_type)
        new_diff = decision["new_difficulty_level"]
        action_code = decision.get("action_code", 1)
        
        if new_diff != current_difficulty:
            transitions += 1
            
        current_difficulty = new_diff
        
        # 2. Simulate student response score based on true skill and difficulty
        # Response model: expected = skill - (diff * 0.15) + 0.30 + noise
        noise = np.random.uniform(-0.08, 0.08)
        expected = student_skill - (current_difficulty * 0.15) + 0.30
        simulated_score = float(np.clip(expected + noise, 0.0, 1.0))
        scores.append(simulated_score)
        
        # 3. Calculate Flow Channel Reward
        if simulated_score < 0.40:
            reward = 1.0 if action_code == 0 else -1.0
        elif simulated_score > 0.70:
            reward = 1.0 if action_code == 2 else -1.0
        else:
            flow_steps += 1
            reward = 1.0 if action_code == 1 else -0.5
            
        cumulative_reward += reward
        last_score = simulated_score
        
    return {
        "mean_score": float(np.mean(scores)),
        "flow_band_ratio": float(flow_steps / steps),
        "transitions": transitions,
        "cumulative_reward": cumulative_reward
    }

def run_adaptive_controller_benchmark():
    print("=" * 80)
    print("AIVA EMPIRICAL BENCHMARK: Adaptive Difficulty Controllers Comparison")
    print("=" * 80)
    
    engine = AdaptiveEngine()
    
    controllers = ["fixed", "rule_based", "dqn"]
    profiles = {
        "Low Skill (0.30)": 0.30,
        "Medium Skill (0.60)": 0.60,
        "High Skill (0.85)": 0.85
    }
    
    seeds = [42, 101, 2024, 777, 999]
    episodes_per_seed = 20
    steps_per_episode = 20
    
    results = {}
    
    for ctrl in controllers:
        results[ctrl] = {}
        print(f"\nEvaluating Controller: {ctrl.upper()}...")
        for prof_name, skill_val in profiles.items():
            all_scores = []
            all_flow_ratios = []
            all_transitions = []
            all_rewards = []
            
            for s_idx, seed in enumerate(seeds):
                for ep in range(episodes_per_seed):
                    run_seed = seed + ep * 13
                    res = simulate_viva_session(ctrl, skill_val, engine, seed=run_seed, steps=steps_per_episode)
                    all_scores.append(res["mean_score"])
                    all_flow_ratios.append(res["flow_band_ratio"])
                    all_transitions.append(res["transitions"])
                    all_rewards.append(res["cumulative_reward"])
                    
            summary = {
                "mean_score": round(float(np.mean(all_scores)), 3),
                "std_score": round(float(np.std(all_scores)), 3),
                "flow_band_percentage": round(float(np.mean(all_flow_ratios) * 100), 2),
                "avg_transitions_per_session": round(float(np.mean(all_transitions)), 2),
                "mean_cumulative_reward": round(float(np.mean(all_rewards)), 2),
                "sample_episodes": len(all_scores)
            }
            results[ctrl][prof_name] = summary
            print(f"  [{prof_name}] -> Flow%: {summary['flow_band_percentage']}%, Transitions: {summary['avg_transitions_per_session']}, Reward: {summary['mean_cumulative_reward']}")

    import datetime
    results["benchmark_metadata"] = {
        "execution_timestamp": datetime.datetime.now().isoformat(),
        "episodes_per_profile": len(all_scores),
        "steps_per_episode": steps_per_episode,
        "dqn_model_path": "models/dqn_difficulty_model.zip"
    }

    # Save to both AIVA_backend/experiments and root experiments
    backend_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
    os.makedirs(backend_out_dir, exist_ok=True)
    backend_file = os.path.join(backend_out_dir, "results_adaptive_controllers.json")
    with open(backend_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    root_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))
    os.makedirs(root_out_dir, exist_ok=True)
    root_file = os.path.join(root_out_dir, "results_adaptive_controllers.json")
    with open(root_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nBenchmark completed successfully! Results written to:")
    print(f"  1. {backend_file}")
    print(f"  2. {root_file}")
    return results

if __name__ == "__main__":
    run_adaptive_controller_benchmark()
