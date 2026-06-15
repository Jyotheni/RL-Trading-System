import time
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 30
SYMBOL = "AAPL"   # change to any stock
INTERVAL = "1m"

print("Loading trained PPO model...")
model = PPO.load("ppo_trading_model")

scaler = MinMaxScaler(feature_range=(-1, 1))

price_history = []

balance = 10000
position = 0
entry_price = 0

print("\n===== LIVE TRADING STARTED =====\n")

while True:

    # Fetch latest data
    data = yf.download(
        SYMBOL,
        period="1d",
        interval=INTERVAL,
        progress=False
    )

    if len(data) < WINDOW_SIZE:
        time.sleep(10)
        continue

    df = data.tail(WINDOW_SIZE)

    features = df[["Open","High","Low","Close","Volume"]].values

    features = scaler.fit_transform(features)

    state = np.array([features], dtype=np.float32)

    action, _ = model.predict(state)

    price = df["Close"].iloc[-1]
    price = float(price.item())

    action = int(action[0])
    action_name = ["HOLD", "BUY", "SELL"][action]

    print(f"""
Stock : {SYMBOL}
Price : {price}

AI Decision
-----------
Action : {action_name}
Balance: {balance}
""")

    if action == 1 and position == 0:

        position = 1
        entry_price = price

        print(f"BUY at {price}")

    elif action == 2 and position == 1:

        profit = price - entry_price
        balance += profit
        position = 0

        print(f"SELL at {price} | Profit: {profit}")

    time.sleep(10)   # wait for next candle