import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO


class TradingEnv(gym.Env):

    def __init__(self, data_path, window_size=30, initial_balance=10000):

        self.df = pd.read_csv(data_path)
        self.df.columns = [c.lower() for c in self.df.columns]

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)

        self.feature_df = self.df.drop(columns=["timestamp"], errors="ignore")
        self.feature_df = self.feature_df.apply(pd.to_numeric, errors="coerce")
        self.feature_df.fillna(0, inplace=True)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_data = scaler.fit_transform(self.feature_df.values)

        self.window_size = window_size
        self.initial_balance = initial_balance

        self.current_step = window_size
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(window_size, self.feature_data.shape[1]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0

        return self._get_state(), {}

    def _get_state(self):

        state = self.feature_data[
            self.current_step - self.window_size:self.current_step
        ]

        return np.clip(state, -1, 1)

    def step(self, action):

        price = float(self.df.iloc[self.current_step]["close"])
        reward = 0

        if action == 1 and self.position == 0:

            self.position = 1
            self.entry_price = price

        elif action == 2 and self.position == 1:

            profit = price - self.entry_price
            reward = np.tanh(profit / 100)
            self.balance += profit
            self.position = 0

        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        obs = self._get_state()

        info = {
            "balance": self.balance
        }

        return obs, reward, terminated, truncated, info


def plot_equity():

    print("Loading trained model...")

    model = PPO.load("ppo_trading_model")

    env = TradingEnv("utils/data/raw/btc_usd_1h.csv")

    obs, _ = env.reset()

    done = False

    equity = [env.balance]

    while not done:

        action, _ = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        equity.append(info["balance"])

    plt.figure(figsize=(10,5))
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    plot_equity()