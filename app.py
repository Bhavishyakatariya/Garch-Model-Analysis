import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from garch_models import GarchModels

st.set_page_config(page_title="Volatility Dashboard", layout="wide")

st.title("📊 Live Volatility Prediction System")

# ==============================
# LOAD MODELS
# ==============================

@st.cache_resource
def load_models():
    model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("features.pkl")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_models()

# ==============================
# SIDEBAR CONFIG
# ==============================

st.sidebar.header("Settings")

period = st.sidebar.selectbox(
    "Select Data Period",
    ["6mo", "1y", "2y", "5y"],
    index=3
)

# ==============================
# FETCH DATA
# ==============================

@st.cache_data
def fetch_data(period):
    tickers = {
        "NIFTY": "^NSEI",
        "Gold": "GC=F",
        "USDINR": "INR=X",
        "NIFTY_IT": "^CNXIT",
        "CrudeOil": "CL=F"
    }

    data = {}

    for key, ticker in tickers.items():
        df = yf.download(ticker, period=period)

        if df.empty:
            continue

        if key == "NIFTY":
            data['Close'] = df['Close']
            data['Open'] = df['Open']
            data['High'] = df['High']
            data['Low'] = df['Low']
        else:
            data[key] = df['Close']

    df_all = pd.concat(data, axis=1)
    df_all.columns = df_all.columns.get_level_values(0)

    return df_all.dropna()

df_all = fetch_data(period)

st.write("Data Shape:", df_all.shape)

# ==============================
# FEATURE CREATION
# ==============================

df_all['overnight_return'] = np.log(
    df_all['Open'] / df_all['Close'].shift(1)
)

df_all['intraday_return'] = np.log(
    df_all['Close'] / df_all['Open']
)

df_all['hl_range'] = (
    (df_all['High'] - df_all['Low']) / df_all['Close']
)

df_all = df_all.dropna()

# ==============================
# PREPROCESS
# ==============================

dp = DataPipeline()
df = dp.preprocess(df_all)

# ==============================
# GARCH LIVE
# ==============================

df_garch = df.copy()
df_garch['returns'] *= 10
df_garch[['Gold','USDINR','NIFTY_IT','CrudeOil']] *= 10

garch = GarchModels()
garch.fit(
    df_garch['returns'],
    df_garch[['Gold','USDINR','NIFTY_IT','CrudeOil']]
)

vol_dict = garch.get_volatility()

# ==============================
# FEATURE ENGINEERING
# ==============================

fe = FeatureEngineer()
df_feat = fe.create_features(df, vol_dict)

df_feat['overnight_return'] = df['overnight_return']
df_feat['intraday_return'] = df['intraday_return']
df_feat['hl_range'] = df['hl_range']

df_feat = df_feat.dropna()

# ==============================
# ALIGN FEATURES
# ==============================

missing_cols = set(feature_cols) - set(df_feat.columns)

for col in missing_cols:
    df_feat[col] = 0

X = df_feat[feature_cols]
X_scaled = scaler.transform(X)

# ==============================
# PREDICTION
# ==============================

preds = model.predict(X_scaled)
df_feat['predicted_vol'] = preds

# ==============================
# DASHBOARD OUTPUT
# ==============================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Volatility Chart")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df_feat['rolling_std'], label="Actual")
    ax.plot(df_feat['predicted_vol'], label="Predicted")
    ax.legend()

    st.pyplot(fig)

with col2:
    st.subheader("📊 Latest Values")

    latest = df_feat[['rolling_std','predicted_vol']].tail(1)

    st.metric("Actual Volatility", round(latest['rolling_std'].values[0],6))
    st.metric("Predicted Volatility", round(latest['predicted_vol'].values[0],6))

# ==============================
# TABLE
# ==============================

st.subheader("📋 Recent Data")
st.dataframe(df_feat[['rolling_std','predicted_vol']].tail(10))