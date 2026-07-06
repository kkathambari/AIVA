import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
import os
import random

class VivaEnv(gym.Env):
    """
    Genuine Reinforcement Learning Environment for VivaForge.
    Simulates a student's performance to train the Adaptive Difficulty Engine.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super(VivaEnv, self).__init__()
        # Actions: 0: Decrease Diff, 1: Keep Diff, 2: Increase Diff
        self.action_space = spaces.Discrete(3)
        
        # State: [Current Difficulty (0-4), Last Answer Score (0.0 - 1.0)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0.0]), 
            high=np.array([4, 1.0]), 
            dtype=np.float32
        )
        
        self.current_difficulty = 1
        self.last_score = 0.5
        self.student_skill = 0.5 # Hidden variable for simulation
        self.steps = 0
        self.max_steps = 20 # A typical viva length

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_difficulty = random.randint(0, 2)
        self.last_score = 0.5
        # Simulate different student skill levels for training generalization
        self.student_skill = random.uniform(0.3, 0.9) 
        self.steps = 0
        return np.array([self.current_difficulty, self.last_score], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        
        # Apply action to difficulty
        if action == 0: # Decrease
            self.current_difficulty = max(0, self.current_difficulty - 1)
        elif action == 2: # Increase
            self.current_difficulty = min(4, self.current_difficulty + 1)
            
        # Simulate student's next score based on their skill and the new difficulty
        # Higher difficulty lowers the score, higher skill raises it.
        expected_score = self.student_skill - (self.current_difficulty * 0.15) + 0.3
        noise = random.uniform(-0.1, 0.1)
        self.last_score = float(np.clip(expected_score + noise, 0.0, 1.0))
        
        # Reward Function (The Flow Channel)
        reward = 0
        if self.last_score < 0.4:
            # Student is struggling (Anxiety)
            if action == 0: reward = 1.0   # Agent helped by decreasing diff
            else: reward = -1.0            # Agent failed to help
        elif self.last_score > 0.7:
            # Student is finding it too easy (Boredom)
            if action == 2: reward = 1.0   # Agent challenged them
            else: reward = -1.0            # Agent failed to challenge
        else:
            # Student is in the Flow Channel (Optimal learning/testing)
            if action == 1: reward = 1.0   # Agent maintained flow
            else: reward = -0.5            # Unnecessary disruption

        done = self.steps >= self.max_steps
        truncated = False
        info = {"student_skill": self.student_skill}
        
        return np.array([self.current_difficulty, self.last_score], dtype=np.float32), reward, done, truncated, info

class AdaptiveEngine:
    def __init__(self, model_path="models/dqn_viva"):
        self.model_path = model_path
        self.env = VivaEnv()
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        if os.path.exists(f"{self.model_path}.zip"):
            self.model = DQN.load(self.model_path, env=self.env)
        else:
            # Initialize untreated model
            self.model = DQN(
                "MlpPolicy", 
                self.env, 
                learning_rate=1e-3,
                buffer_size=10000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.99,
                target_update_interval=500,
                verbose=0
            )

    def train_model(self, total_timesteps=10000):
        """
        Pre-trains the RL model offline by simulating thousands of viva interactions.
        """
        print(f"Starting DQN Training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.model_path)
        print(f"Training Complete. Model saved to {self.model_path}")

    def get_next_difficulty(self, current_difficulty: int, last_answer_score: float) -> dict:
        """
        Uses the trained RL model to predict whether to increase, decrease, or maintain difficulty.
        """
        state = np.array([current_difficulty, last_answer_score], dtype=np.float32)
        # deterministic=True takes the best learned action, False allows exploration
        action, _states = self.model.predict(state, deterministic=True)
        
        new_diff = current_difficulty
        action_name = "KEEP"
        if action == 0:
            new_diff = max(0, current_difficulty - 1)
            action_name = "DECREASE"
        elif action == 2:
            new_diff = min(4, current_difficulty + 1)
            action_name = "INCREASE"
            
        difficulty_mapping = {
            0: "Fundamentals",
            1: "Easy",
            2: "Medium",
            3: "Hard",
            4: "Research Level"
        }
            
        return {
            "current_difficulty": current_difficulty,
            "last_score": last_answer_score,
            "action_taken": action_name,
            "new_difficulty_level": new_diff,
            "new_difficulty_name": difficulty_mapping[new_diff]
        }

