from fastapi import FastAPI
import yfinance as yf
from stable_baselines3 import PPO
import numpy as np
import pandas as pd

app = FastAPI()

# load trained model
model = PPO.load("C:\\Users\\ADMIN\\Documents\\8th sem\\RL\\RL PRO\\rl-trading-system\\ppo_trading_model.zip")

SYMBOL = "AAPL"

@app.get("/set-symbol")
def set_symbol(symbol:str):
    global SYMBOL
    SYMBOL = symbol
    return {"symbol":SYMBOL}

balance = 10000
position = 0
entry_price = 0


@app.get("/")
def home():
    return {"message": "RL Trading API running"}


@app.get("/price")
def get_price():

    data = yf.download(SYMBOL, period="1d", interval="1m", progress=False)

    price = float(data["Close"].iloc[-1])

    return {"symbol": SYMBOL, "price": price}


@app.get("/ai-decision")
def ai_decision():

    global balance, position, entry_price

    data = yf.download(SYMBOL, period="1d", interval="1m", progress=False)

    df = data.tail(30)

    if len(df) < 30:
        return {"message": "Not enough data"}

    features = df[["Open","High","Low","Close","Volume"]].values

    state = np.array([features], dtype=np.float32)

    action, _ = model.predict(state)

    action = int(action)

    price = float(df["Close"].iloc[-1])

    action_name = ["HOLD","BUY","SELL"][action]

    if action == 1 and position == 0:
        position = 1
        entry_price = price

    elif action == 2 and position == 1:
        profit = price - entry_price
        balance += profit
        position = 0

    return {
        "symbol": SYMBOL,
        "action": action_name,
        "price": price,
        "balance": balance
    }