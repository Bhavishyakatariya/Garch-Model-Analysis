import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Ensure local classes are imported
from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer

# ==============================
# LOAD SAVED MODELS
# ==============================
# Adjust paths if your models are in a specific folder
model = joblib.load("xgb_model.pkl")
garch = joblib.load("garch.pkl")
feature_cols = joblib.load("features.pkl")
# Note: If you didn't save a scaler in your training script, 
# you might need to skip scaling or ensure it's saved.

# ==============================
# FETCH DATA (Fixed for Multi-Index)
# ==============================
tickers = {
    "NIFTY": "^NSEI",
    "Gold": "GC=F",
    "USDINR": "INR=X",
    "NIFTY_IT": "^CNXIT",
    "CrudeOil": "CL=F"
}

data_list = []

for key, ticker in tickers.items():
    print(f"Downloading {ticker}...")
    # multi_level_index=False flattens the columns immediately
    df_raw = yf.download(ticker, period="2y", multi_level_index=False)
    
    if key == "NIFTY":
        # Keep OHLC for Nifty to use in better volatility features
        temp_df = df_raw[['Open', 'High', 'Low', 'Close']].copy()
        temp_df.columns = [f"NIFTY_{c}" for c in temp_df.columns]
        data_list.append(temp_df)
    else:
        # For macro variables, we only need Close
        data_list.append(df_raw['Close'].rename(key))

# Combine all into one dataframe
df_all = pd.concat(data_list, axis=1).dropna()

# ==============================
# CREATE ENHANCED FEATURES
# ==============================
# 1. Real HL Range using High and Low (not just Close)
df_all['hl_range'] = (df_all['NIFTY_High'] - df_all['NIFTY_Low']) / df_all['NIFTY_Close']

# 2. Intraday Return (Close vs Open)
df_all['intraday_return'] = (df_all['NIFTY_Close'] - df_all['NIFTY_Open']) / df_all['NIFTY_Open']

# 3. Overnight Return (Open vs Prev Close)
df_all['overnight_return'] = np.log(df_all['NIFTY_Open'] / df_all['NIFTY_Close'].shift(1))

# 4. Standard Returns for GARCH
df_all['returns'] = np.log(df_all['NIFTY_Close'] / df_all['NIFTY_Close'].shift(1))

df_all = df_all.dropna()

# ==============================
# PREPROCESS & GARCH
# ==============================
# Prepare the dictionary for the FeatureEngineer
# Note: This uses the GARCH models you trained earlier
vol_dict = garch.get_volatility()

# Align GARCH volatility with the current dataframe dates
# We take the latest available volatility estimates
current_vol = {
    "egarch_vol": pd.Series(vol_dict['egarch_vol'], index=df_all.index).ffill(),
    "gjr_vol": pd.Series(vol_dict['gjr_vol'], index=df_all.index).ffill(),
    "exo_vol": pd.Series(vol_dict['exo_vol'], index=df_all.index).ffill()
}

# ==============================
# FEATURE ENGINEERING
# ==============================
fe = FeatureEngineer()
# Map the NIFTY_Close back to 'returns' if your FeatureEngineer expects it
df_processed = df_all.rename(columns={'NIFTY_Close': 'Close', 'NIFTY_Open': 'Open', 
                                     'NIFTY_High': 'High', 'NIFTY_Low': 'Low'})

df_feat = fe.create_features(df_processed, current_vol)

# ==============================
# PREDICTION
# ==============================
# Ensure only the columns used during training are passed to the model
X = df_feat[feature_cols]

# Predict
df_feat['predicted_vol'] = model.predict(X)

# ==============================
# PLOT & OUTPUT
# ==============================
plt.figure(figsize=(12, 6))
plt.plot(df_feat.index, df_feat['rolling_std'], label="Target Vol (Rolling Std)", alpha=0.5)
plt.plot(df_feat.index, df_feat['predicted_vol'], label="Predicted Volatility", color='red', linewidth=2)
plt.title("Nifty 50 Volatility Prediction (Using OHLC + GARCH)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n===== LATEST PREDICTIONS =====")
print(df_feat[['predicted_vol']].tail())