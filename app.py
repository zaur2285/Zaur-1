
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="КриптоАнализ — Заур", layout="wide")

st.title("🔍 Прогнозирование криптовалют — Заур")

symbol = st.selectbox("Выберите монету", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])

def get_data(symbol):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=60&limit=200"
    response = requests.get(url).json()
    if "result" in response and "list" in response["result"]:
        df = pd.DataFrame(response["result"]["list"])
        df.columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close"] = df["close"].astype(float)
        return df[["timestamp", "close"]]
    return pd.DataFrame()

df = get_data(symbol)
if df.empty:
    st.error("Ошибка при загрузке данных.")
    st.stop()

st.line_chart(df.set_index("timestamp")["close"])

# Фичи и прогноз
df["target"] = df["close"].shift(-1) > df["close"]
df["return"] = df["close"].pct_change()
df["ma_10"] = df["close"].rolling(10).mean()
df = df.dropna()

X = df[["return", "ma_10"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
model = RandomForestClassifier()
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[-1]
prediction = "📈 Рост" if proba[1] > 0.5 else "📉 Падение"

st.subheader("📊 Прогноз:")
st.metric(label="Ожидается", value=prediction, delta=f"{proba[1]*100:.2f}% вероятность роста")
