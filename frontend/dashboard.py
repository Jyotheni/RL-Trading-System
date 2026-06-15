import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import time
import random

st.set_page_config(
    page_title="AI Quantum Trading Terminal",
    layout="wide"
)

st.markdown("""
<style>
/* Main background */
body {
    background-color: #0b1e3d;
}

/* App background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0b1e3d, #1c3b6b);
    color: white;
}

/* Cards */
[data-testid="stMetric"], .css-1r6slb0 {
    background-color: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 10px;
}

/* Headers */
h1, h2, h3 {
    color: #ffffff;
}

/* Buttons */
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
}

/* Charts spacing */
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Quantum Trading Terminal")

# ---------------- STOCK SELECTOR ----------------

stocks = ["AAPL","TSLA","MSFT","NVDA","AMZN"]

symbol = st.selectbox("Select Stock",stocks)

# ---------------- API DATA ----------------

price_data = requests.get("http://127.0.0.1:8000/price").json()
ai_data = requests.get("http://127.0.0.1:8000/ai-decision").json()

price = price_data["price"]
action = ai_data["action"]
balance = ai_data["balance"]

# ---------------- METRICS ----------------

col1,col2,col3 = st.columns(3)

col1.metric("💰 Live Price",round(price,2))
col2.metric("🤖 AI Decision",action)
col3.metric("🏦 Balance",round(balance,2))

# ---------------- MARKET DATA ----------------

market = yf.download(symbol,period="1d",interval="1m", progress=False)

left, right = st.columns([3,1])

# ---------------- CANDLESTICK CHART ----------------

# CREATE LAYOUT
left, right = st.columns([3,1])

# ---------------- CANDLESTICK CHART ----------------
with left:

    st.subheader("📊 Live Candlestick Chart")

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=market.index,
        open=market["Open"],
        high=market["High"],
        low=market["Low"],
        close=market["Close"],
        name="Price"
    ))

    # BUY / SELL markers
    buy_points = market.sample(3)
    sell_points = market.sample(3)

    fig.add_trace(go.Scatter(
        x=buy_points.index,
        y=buy_points["Close"],
        mode="markers",
        marker=dict(color="green",size=10),
        name="BUY"
    ))

    fig.add_trace(go.Scatter(
        x=sell_points.index,
        y=sell_points["Close"],
        mode="markers",
        marker=dict(color="red",size=10),
        name="SELL"
    ))

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, width="stretch")

# ---------------- LIVE TRADING GRAPH ----------------

st.subheader("💓 Live Trading")

if "ecg_prices" not in st.session_state:
    st.session_state.ecg_prices = []

if "ecg_time" not in st.session_state:
    st.session_state.ecg_time = []

# get live price
price_data = requests.get("http://127.0.0.1:8000/price").json()
price = price_data["price"]

# create ECG-like variation (adds spikes effect)
noise = random.uniform(-0.3, 0.3)
spike = random.choice([0, 0, 0, 1.5, -1.5])  # occasional spikes

ecg_price = price + noise + spike

# store values
st.session_state.ecg_prices.append(ecg_price)
import datetime
st.session_state.ecg_time.append(datetime.datetime.now())

# keep only last 60 points (scroll effect)
st.session_state.ecg_prices = st.session_state.ecg_prices[-60:]
st.session_state.ecg_time = st.session_state.ecg_time[-60:]

df = pd.DataFrame({
    "Time": st.session_state.ecg_time,
    "Price": st.session_state.ecg_prices
})

# ECG-style plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Time"],
    y=df["Price"],
    mode="lines",
    line=dict(color="#00FF00", width=3),  # neon green ECG color
))

fig.update_layout(
    template="plotly_white",
    height=350,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    plot_bgcolor="black",   # ECG dark screen
    paper_bgcolor="white",
    margin=dict(l=10, r=10, t=30, b=10)
)

st.plotly_chart(fig, width="stretch")


# ---------------- ORDER BOOK ----------------

st.subheader("📑 Order Book")

bids = pd.DataFrame({
    "Price":[price-random.uniform(0.1,1) for _ in range(5)],
    "Volume":[random.randint(50,500) for _ in range(5)]
})

asks = pd.DataFrame({
    "Price":[price+random.uniform(0.1,1) for _ in range(5)],
    "Volume":[random.randint(50,500) for _ in range(5)]
})

col1,col2 = st.columns(2)

col1.write("### 🟢 Bids")
col1.dataframe(bids,width="stretch")

col2.write("### 🔴 Asks")
col2.dataframe(asks,width="stretch")

# ---------------- PORTFOLIO ALLOCATION ----------------

st.subheader("📊 Portfolio Allocation")

portfolio = {
    "AAPL":random.randint(10,30),
    "TSLA":random.randint(10,30),
    "MSFT":random.randint(10,30),
    "NVDA":random.randint(10,30)
}

pie = go.Figure(data=[go.Pie(
    labels=list(portfolio.keys()),
    values=list(portfolio.values())
)])

pie.update_layout(template="plotly_white")

st.plotly_chart(pie,width="stretch")

# ---------------- AI vs BUY HOLD ----------------

st.subheader("⚔ AI vs Buy & Hold Strategy")

prices = market["Close"]

# safe calculation
if len(prices) > 1:
    buy_hold_profit = float(prices.values[-1]) - float(prices.values[0])
else:
    buy_hold_profit = 0.0

# ✅ DEFINE THIS BEFORE USING
ai_profit = float(balance - 10000)

compare = pd.DataFrame({
    "Strategy":["AI Trading","Buy & Hold"],
    "Profit":[ai_profit, buy_hold_profit]
})

bar = go.Figure()

colors = ["green" if float(p) > 0 else "red" for p in compare["Profit"].values]

bar.add_trace(go.Bar(
    x=compare["Strategy"],
    y=compare["Profit"],
    marker_color=colors
))

bar.update_layout(
    template="plotly_white",
    title="Strategy Performance"
)

st.plotly_chart(bar, width="stretch")

# ---------------- AUTO REFRESH ----------------

time.sleep(5)
st.rerun()