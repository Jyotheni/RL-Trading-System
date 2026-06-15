import yfinance as yf
import pandas as pd
import os


def download_data(symbol="BTC-USD", interval="1h", period="730d"):
    
    data = yf.download(
        symbol,
        interval=interval,
        period=period
    )

    data.reset_index(inplace=True)

    data = data[[
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]]

    data.columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    return data


if __name__ == "__main__":

    df = download_data()

    # Create the directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)

    df.to_csv("data/raw/btc_usd_1h.csv", index=False)

    print("Data downloaded successfully")
    print(df.head())