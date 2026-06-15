import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
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

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_data = self.scaler.fit_transform(self.feature_df.values).astype(np.float32)

        self.feature_data = np.nan_to_num(self.feature_data)

        self.window_size = window_size
        self.initial_balance = initial_balance

        self.current_step = window_size
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(window_size, self.feature_data.shape[1]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0

        return self._get_state(), {}

    def _get_state(self):

        state = self.feature_data[
            self.current_step - self.window_size:self.current_step
        ]

        return np.clip(state, -1.0, 1.0)

    def step(self, action):

        price = float(self.df.iloc[self.current_step]["close"])
        reward = 0.0

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price

        elif action == 2 and self.position == 1:
            profit = price - self.entry_price
            reward = np.tanh(profit / 100.0)
            self.balance += profit
            self.position = 0

        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        observation = self._get_state()

        info = {
            "balance": self.balance,
            "position": self.position
        }

        return observation, float(reward), terminated, truncated, info


def calculate_metrics(portfolio):

    portfolio = np.array(portfolio)

    returns = np.diff(portfolio) / portfolio[:-1]

    total_return = (portfolio[-1] - portfolio[0]) / portfolio[0]

    sharpe = 0
    if returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    cumulative_max = np.maximum.accumulate(portfolio)
    drawdown = (portfolio - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    return total_return, sharpe, max_drawdown


def run_backtest():

    print("Loading trained PPO model...")

    model = PPO.load("ppo_trading_model")

    data_path = "utils/data/raw/btc_usd_1h.csv"

    env = TradingEnv(data_path)

    obs, _ = env.reset()

    done = False

    portfolio_values = [env.balance]

    print("Starting backtest...")

    while not done:

        action, _ = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        portfolio_values.append(info["balance"])

    total_return, sharpe, max_dd = calculate_metrics(portfolio_values)

    print("\n===== BACKTEST RESULTS =====")

    print("Final Balance:", portfolio_values[-1])
    print("ROI:", round(total_return * 100, 2), "%")
    print("Sharpe Ratio:", round(sharpe, 3))
    print("Max Drawdown:", round(max_dd * 100, 2), "%")


if __name__ == "__main__":
    run_backtest()