import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_rel

from data_pipeline import DataPipeline
from garch_models import GarchModels
from feature_engineering import FeatureEngineer

# ==============================
# CONFIG
# ==============================
FEATURES = [
    'Gold', 'USDINR', 'NIFTY_IT', 'CrudeOil',
    'overnight_return', 'hl_range'
]

LAGS = 10

# ==============================
# LOAD SAVED MODELS
# ==============================
xgb_model = joblib.load("xgb_model.pkl")
meta_model = joblib.load("meta_model.pkl")   # ✅ NEW
garch = joblib.load("garch.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("features.pkl")

# ==============================
# LOAD DATA
# ==============================
df_raw = pd.read_csv("nifty50_dataset.csv")

dp = DataPipeline()
df = dp.preprocess(df_raw)

# ==============================
# GARCH OUTPUT
# ==============================
returns = df['returns'] * 100
exog = df[FEATURES] * 100

garch.fit(returns, exog)
vol_dict = garch.get_volatility()

egarch_vol = pd.Series(vol_dict["egarch_vol"], index=df.index)
gjr_vol = pd.Series(vol_dict["gjr_vol"], index=df.index)
exo_vol = pd.Series(vol_dict["exo_vol"], index=df.index)

# FIX 1: Scale GARCH back to raw scale (trained on returns*100)
egarch_vol = egarch_vol / 100
gjr_vol = gjr_vol / 100
exo_vol = exo_vol / 100

# ==============================
# FEATURE ENGINEERING
# ==============================
fe = FeatureEngineer()
df_feat = fe.create_features(df, vol_dict, lags=LAGS)

df_feat['overnight_return'] = df['overnight_return']
df_feat['hl_range'] = df['hl_range']
df_feat['intraday_return'] = df['intraday_return']

df_feat = df_feat.dropna()

# ==============================
# ML DATA
# ==============================
y = df_feat['rolling_std']
X = df_feat.drop(columns=['rolling_std', 'returns'])

X_scaled = scaler.transform(X)

# ==============================
# TRAIN-TEST SPLIT
# ==============================
split = int(0.8 * len(X_scaled))

X_test = X_scaled[split:]
y_test = y[split:]

# ==============================
# BASE MODEL PREDICTIONS
# ==============================
xgb_pred = xgb_model.predict(X_test)
xgb_pred = pd.Series(xgb_pred, index=y_test.index)

# Align GARCH outputs
egarch_vol = egarch_vol.reindex(y_test.index)
gjr_vol = gjr_vol.reindex(y_test.index)
exo_vol = exo_vol.reindex(y_test.index)

# ==============================
# STACK MODEL PREDICTION
# ==============================
garch_stack = (
    0.4 * egarch_vol +
    0.3 * gjr_vol +
    0.3 * exo_vol
)

# Market signal: absolute returns
abs_return = np.abs(df_feat['returns'].loc[y_test.index])

stack_X = pd.DataFrame({
    "garch": garch_stack,
    "egarch": egarch_vol,
    "gjr": gjr_vol,
    "exo": exo_vol,
    "xgb": xgb_pred,
    "abs_return": abs_return
}, index=y_test.index)

stack_pred = meta_model.predict(stack_X)
stack_pred = pd.Series(stack_pred, index=y_test.index)

actual_vol = y_test

# ==============================
# RMSE FUNCTION
# ==============================
def compute_rmse(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred))

# ==============================
# P-VALUE FUNCTION (Paired Test)
# ==============================
def compute_p_value(actual, pred1, pred2):
    err1 = (actual - pred1) ** 2
    err2 = (actual - pred2) ** 2
    stat, p_value = ttest_rel(err1, err2)
    return p_value

# ==============================
# RMSE CALCULATION
# ==============================
rmse_egarch = compute_rmse(actual_vol, egarch_vol)
rmse_gjr = compute_rmse(actual_vol, gjr_vol)
rmse_exo = compute_rmse(actual_vol, exo_vol)
rmse_xgb = compute_rmse(actual_vol, xgb_pred)
rmse_stack = compute_rmse(actual_vol, stack_pred)

# ==============================
# P-VALUE CALCULATION
# ==============================
p_xgb_vs_stack = compute_p_value(actual_vol, xgb_pred, stack_pred)
p_egarch_vs_stack = compute_p_value(actual_vol, egarch_vol, stack_pred)
p_exo_vs_stack = compute_p_value(actual_vol, exo_vol, stack_pred)

# ==============================
# PRINT RESULTS
# ==============================
print("\n===== RMSE COMPARISON =====")
print(f"EGARCH RMSE     : {rmse_egarch:.6f}")
print(f"GJR-GARCH RMSE  : {rmse_gjr:.6f}")
print(f"EGARCH-X RMSE   : {rmse_exo:.6f}")
print(f"XGBoost RMSE    : {rmse_xgb:.6f}")
print(f"STACK MODEL RMSE: {rmse_stack:.6f}")

print("\n===== P-VALUE TEST =====")
print("(* lower than 0.05 = statistically significant improvement *)")
print(f"Stack vs XGBoost p-value : {p_xgb_vs_stack:.6f}")
print(f"Stack vs EGARCH p-value  : {p_egarch_vs_stack:.6f}")
print(f"Stack vs EGARCH-X p-value: {p_exo_vs_stack:.6f}")