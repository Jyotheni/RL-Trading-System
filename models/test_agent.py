import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO
from env.trading_env import TradingEnv

def test_agent():

    print("Loading trained model...")

    model = PPO.load("ppo_trading_model")

    data_path = "utils/data/raw/btc_usd_1h.csv"

    env = TradingEnv(data_path)

    obs, _ = env.reset()

    done = False

    trades = []

    print("Starting evaluation...")

    while not done:

        action, _ = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        trades.append({
            "step": env.current_step,
            "action": int(action),
            "balance": info["balance"],
            "position": info["position"]
        })

    print("Evaluation finished")

    df = pd.DataFrame(trades)

    df.to_csv("trading_results.csv", index=False)

    print("Results saved to trading_results.csv")

    print("Final Balance:", info["balance"])

if __name__ == "__main__":
    test_agent()