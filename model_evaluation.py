import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_squared_error

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
# LOAD MODELS
# ==============================
xgb_model = joblib.load("xgb_model.pkl")
meta_model = joblib.load("meta_model.pkl")
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
# GARCH FIT + VOLATILITY
# ==============================
returns = df['returns'] * 100
exog = df[FEATURES] * 100

garch.fit(returns, exog)
vol_dict = garch.get_volatility()

egarch_vol = pd.Series(vol_dict["egarch_vol"], index=df.index) / 100
gjr_vol = pd.Series(vol_dict["gjr_vol"], index=df.index) / 100
exo_vol = pd.Series(vol_dict["exo_vol"], index=df.index) / 100

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
# SPLIT
# ==============================
split = int(0.8 * len(X_scaled))

X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

train_index = y_train.index
test_index = y_test.index

# ==============================
# GARCH ALIGN
# ==============================
egarch_train = egarch_vol.loc[train_index]
gjr_train = gjr_vol.loc[train_index]
exo_train = exo_vol.loc[train_index]

egarch_test = egarch_vol.loc[test_index]
gjr_test = gjr_vol.loc[test_index]
exo_test = exo_vol.loc[test_index]

# ==============================
# STACK MODEL
# ==============================
xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

# Combine GARCH
garch_train_stack = 0.4*egarch_train + 0.3*gjr_train + 0.3*exo_train
garch_test_stack = 0.4*egarch_test + 0.3*gjr_test + 0.3*exo_test

# Additional feature
abs_return_train = abs(df_feat.loc[train_index, 'returns'])
abs_return_test = abs(df_feat.loc[test_index, 'returns'])

# Stack input
stack_train = pd.DataFrame({
    "garch": garch_train_stack,
    "egarch": egarch_train,
    "gjr": gjr_train,
    "exo": exo_train,
    "xgb": xgb_train_pred,
    "abs_return": abs_return_train
}, index=train_index)

stack_test = pd.DataFrame({
    "garch": garch_test_stack,
    "egarch": egarch_test,
    "gjr": gjr_test,
    "exo": exo_test,
    "xgb": xgb_test_pred,
    "abs_return": abs_return_test
}, index=test_index)

# Predictions
stack_train_pred = meta_model.predict(stack_train)
stack_test_pred = meta_model.predict(stack_test)

# ==============================
# RMSE FUNCTION
# ==============================
def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))

# ==============================
# RMSE RESULTS
# ==============================
print("\n===== GARCH MODELS PERFORMANCE =====")

print("\nTraining Data RMSE")
print("GARCH (EGARCH)    :", rmse(y_train, egarch_train))
print("GJR-GARCH         :", rmse(y_train, gjr_train))
print("EGARCH-X          :", rmse(y_train, exo_train))

print("\nTesting Data RMSE")
print("GARCH (EGARCH)    :", rmse(y_test, egarch_test))
print("GJR-GARCH         :", rmse(y_test, gjr_test))
print("EGARCH-X          :", rmse(y_test, exo_test))

print("\n===== STACK MODEL PERFORMANCE =====")

print("\nTraining Data RMSE")
print("STACK MODEL       :", rmse(y_train, stack_train_pred))

print("\nTesting Data RMSE")
print("STACK MODEL       :", rmse(y_test, stack_test_pred))

# ==============================
# BIC SCORES
# ==============================
print("\n===== BIC SCORES =====")

print("EGARCH   :", garch.egarch.bic)
print("GJR-GARCH:", garch.gjr.bic)
print("EGARCH-X :", garch.egarch_x.bic)