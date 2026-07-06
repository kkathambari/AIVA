from core.adaptive_engine import AdaptiveEngine
import os

def train_rl():
    print("Initializing RL Adaptive Engine...")
    engine = AdaptiveEngine()
    
    # We will train the model for 15,000 timesteps to ensure it converges
    print("Training DQN model for 15,000 steps...")
    engine.train_model(total_timesteps=15000)
    print("RL Model training finished and saved.")

if __name__ == "__main__":
    train_rl()
