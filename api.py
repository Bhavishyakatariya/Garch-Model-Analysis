from fastapi import FastAPI
import pandas as pd
import numpy as np
import yfinance as yf
import joblib

from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from garch_models import GarchModels

app = FastAPI()

# ==============================
# LOAD MODELS (ONCE)
# ==============================

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("features.pkl")

# ==============================
# FUNCTION: FETCH + PROCESS
# ==============================

def get_data():
    tickers = {
        "NIFTY": "^NSEI",
        "Gold": "GC=F",
        "USDINR": "INR=X",
        "NIFTY_IT": "^CNXIT",
        "CrudeOil": "CL=F"
    }

    data = {}

    for key, ticker in tickers.items():
        df = yf.download(ticker, period="2y")

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

    df_all = df_all.dropna()

    # Features
    df_all['overnight_return'] = np.log(
        df_all['Open'] / df_all['Close'].shift(1)
    )

    df_all['intraday_return'] = np.log(
        df_all['Close'] / df_all['Open']
    )

    df_all['hl_range'] = (
        (df_all['High'] - df_all['Low']) / df_all['Close']
    )

    return df_all.dropna()

# ==============================
# API ENDPOINT
# ==============================

@app.get("/predict")
def predict():

    df_all = get_data()

    # Preprocess
    dp = DataPipeline()
    df = dp.preprocess(df_all)

    # GARCH
    df_garch = df.copy()
    df_garch['returns'] *= 10
    df_garch[['Gold','USDINR','NIFTY_IT','CrudeOil']] *= 10

    garch = GarchModels()
    garch.fit(
        df_garch['returns'],
        df_garch[['Gold','USDINR','NIFTY_IT','CrudeOil']]
    )

    vol_dict = garch.get_volatility()

    # Feature engineering
    fe = FeatureEngineer()
    df_feat = fe.create_features(df, vol_dict)

    df_feat['overnight_return'] = df['overnight_return']
    df_feat['intraday_return'] = df['intraday_return']
    df_feat['hl_range'] = df['hl_range']

    df_feat = df_feat.dropna()

    # Align features
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0

    X = df_feat[feature_cols]
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    df_feat['predicted_vol'] = preds

    latest = df_feat[['rolling_std','predicted_vol']].tail(1)

    return {
        "actual_vol": float(latest['rolling_std'].values[0]),
        "predicted_vol": float(latest['predicted_vol'].values[0])
    }