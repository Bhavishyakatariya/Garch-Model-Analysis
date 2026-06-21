import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineer
from garch_models import GarchModels

# ==============================
# LOAD SAVED MODELS
# ==============================

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("features.pkl")

# ==============================
# FETCH DATA
# ==============================

tickers = {
    "NIFTY": "^NSEI",
    "Gold": "GC=F",
    "USDINR": "INR=X",
    "NIFTY_IT": "^CNXIT",
    "CrudeOil": "CL=F"
}

data = {}

for key, ticker in tickers.items():
    df = yf.download(ticker, period="5mo")

    if df.empty:
        print(f"❌ Failed {ticker}")
        continue

    # 🔥 For NIFTY take OHLC
    if key == "NIFTY":
        data['Close'] = df['Close']
        data['Open'] = df['Open']
        data['High'] = df['High']
        data['Low'] = df['Low']
    else:
        data[key] = df['Close']

# ==============================
# SAFE MERGE
# ==============================

df_all = pd.concat(data, axis=1)
df_all.columns = df_all.columns.get_level_values(0)
df_all = df_all.dropna()

print("Data shape:", df_all.shape)

# ==============================
# FEATURE CREATION (CORRECT)
# ==============================

# Overnight return
df_all['overnight_return'] = np.log(
    df_all['Open'] / df_all['Close'].shift(1)
)

# Intraday return
df_all['intraday_return'] = np.log(
    df_all['Close'] / df_all['Open']
)

# High-Low range
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
# GARCH (LIVE FIT)
# ==============================

df_garch = df.copy()

# 🔥 Scaling
df_garch['returns'] = df_garch['returns'] * 10
df_garch[['Gold','USDINR','NIFTY_IT','CrudeOil']] = \
    df_garch[['Gold','USDINR','NIFTY_IT','CrudeOil']] * 10

garch_live = GarchModels()

garch_live.fit(
    df_garch['returns'],
    df_garch[['Gold','USDINR','NIFTY_IT','CrudeOil']]
)

vol_dict = garch_live.get_volatility()

# ==============================
# FEATURE ENGINEERING
# ==============================

fe = FeatureEngineer()
df_feat = fe.create_features(df, vol_dict)

# Add extra features
df_feat['overnight_return'] = df['overnight_return']
df_feat['hl_range'] = df['hl_range']
df_feat['intraday_return'] = df['intraday_return']

df_feat = df_feat.dropna()

# ==============================
# ALIGN FEATURES (CRITICAL FIX)
# ==============================

missing_cols = set(feature_cols) - set(df_feat.columns)

if len(missing_cols) > 0:
    print("⚠ Missing columns:", missing_cols)
    for col in missing_cols:
        df_feat[col] = 0   # fallback (safe)

# Ensure correct order
X = df_feat[feature_cols]

# ==============================
# PREDICTION
# ==============================

X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)

df_feat['predicted_vol'] = preds

# ==============================
# PLOT
# ==============================

plt.figure(figsize=(10,5))
plt.plot(df_feat['rolling_std'], label="Actual Volatility")
plt.plot(df_feat['predicted_vol'], label="Predicted Volatility")
plt.legend()
plt.title("Live Volatility Prediction")
plt.show()

# ==============================
# LATEST OUTPUT
# ==============================

print("\n===== LATEST PREDICTION =====")
print(df_feat[['rolling_std','predicted_vol']].tail())