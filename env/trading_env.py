import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class TradingEnv(gym.Env):

    def __init__(self, data_path, window_size=30, initial_balance=10000):

        # Load dataset
        self.df = pd.read_csv(data_path)

        # Convert all column names to lowercase
        self.df.columns = [c.lower() for c in self.df.columns]

        # Remove invalid values
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)

        # Keep only numeric columns except timestamp
        self.feature_df = self.df.drop(columns=["timestamp"], errors="ignore")

        # Force numeric conversion
        self.feature_df = self.feature_df.apply(pd.to_numeric, errors="coerce")

        # Replace remaining NaN
        self.feature_df.fillna(0, inplace=True)

        # CRITICAL FIX: Normalize features to prevent NaN in network
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_data = self.scaler.fit_transform(self.feature_df.values).astype(np.float32)

        # Final safety check on normalized data
        self.feature_data = np.nan_to_num(self.feature_data, nan=0.0, posinf=0.0, neginf=0.0)

        self.window_size = window_size
        self.initial_balance = initial_balance

        self.current_step = window_size
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0

        # Actions: HOLD, BUY, SELL
        self.action_space = spaces.Discrete(3)

        # CRITICAL FIX: Use bounded observation space instead of inf
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

        # Final safety check - clamp to [-1, 1]
        state = np.clip(state, -1.0, 1.0)

        return state

    def step(self, action):

        price = float(self.df.iloc[self.current_step]["close"])

        reward = 0.0

        if action == 1 and self.position == 0:  # BUY
            self.position = 1
            self.entry_price = price

        elif action == 2 and self.position == 1:  # SELL
            profit = price - self.entry_price
            # CRITICAL FIX: Normalize reward to prevent extreme values
            reward = np.tanh(profit / 100.0)  # Clip reward to [-1, 1]
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