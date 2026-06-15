import pandas as pd
import numpy as np
import os


def generate_features(input_path, output_path):

    # Load dataset
    df = pd.read_csv(input_path)

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # -------- Basic Returns --------
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # -------- Moving Averages --------
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_30"] = df["close"].rolling(window=30).mean()

    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_30"] = df["close"].ewm(span=30, adjust=False).mean()

    # -------- RSI --------
    delta = df["close"].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # -------- Volatility --------
    df["volatility"] = df["return"].rolling(window=10).std()

    # -------- Volume Change --------
    df["volume_change"] = df["volume"].pct_change()

    # Remove NaN rows caused by rolling calculations
    df.dropna(inplace=True)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save processed data
    df.to_csv(output_path, index=False)

    print("Feature engineering completed")
    print("Saved to:", output_path)
    print(df.head())


if __name__ == "__main__":

    input_file = r"C:\Users\ADMIN\Documents\8th sem\RL\RL PRO\rl-trading-system\utils\data\raw\btc_usd_1h.csv"

    output_file = r"C:\Users\ADMIN\Documents\8th sem\RL\RL PRO\rl-trading-system\utils\data\processed\features.csv"

    generate_features(input_file, output_file)