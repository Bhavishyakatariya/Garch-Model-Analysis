import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from data_pipeline import DataPipeline
from garch_models import GarchModels
from feature_engineering import FeatureEngineer
from xgboost_model import XGBModel

from xgboost import XGBRegressor

# ==============================
# CONFIG
# ==============================
FEATURES = [
    'Gold', 'USDINR', 'NIFTY_IT', 'CrudeOil',
    'overnight_return', 'hl_range'
]

LAGS = 10

params = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9
}

# ==============================
# LOAD DATA
# ==============================
df_raw = pd.read_csv("nifty50_dataset.csv")

# ==============================
# PREPROCESS
# ==============================
dp = DataPipeline()
df = dp.preprocess(df_raw)

# ==============================
# GARCH MODELS
# ==============================
garch = GarchModels()

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

# 🔥 GARCH STACK FEATURE
garch_stack = 0.4*egarch_vol + 0.3*gjr_vol + 0.3*exo_vol

# ==============================
# FEATURE ENGINEERING
# ==============================
fe = FeatureEngineer()
df_feat = fe.create_features(df, vol_dict, lags=LAGS)

df_feat['overnight_return'] = df['overnight_return']
df_feat['hl_range'] = df['hl_range']
df_feat['intraday_return'] = df['intraday_return']

df_feat = df_feat.dropna()

# Align GARCH
garch_stack = garch_stack.loc[df_feat.index]

# ==============================
# ML DATA
# ==============================
y = df_feat['rolling_std']
X = df_feat.drop(columns=['rolling_std', 'returns'])

feature_cols = X.columns.tolist()

# ==============================
# SCALING
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# TRAIN-TEST SPLIT
# ==============================
split = int(0.8 * len(X_scaled))

X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# ==============================
# BASE MODEL: XGBOOST
# ==============================
xgb = XGBModel(params)
xgb.train(X_train, y_train)

xgb_pred = xgb.predict(X_test)

# ==============================
# STACKING DATASET - META MODEL TRAINING
# ==============================
# FIX 2: Train meta model on TRAIN data only (avoid data leakage)
# Generate predictions on TRAIN data ONLY
xgb_pred_train = xgb.predict(X_train)

# Extract GARCH train portions
garch_train = garch_stack.loc[y_train.index]
egarch_train = egarch_vol.loc[y_train.index]
gjr_train = gjr_vol.loc[y_train.index]
exo_train = exo_vol.loc[y_train.index]

# Market signal: absolute returns
abs_return_train = np.abs(df_feat['returns'].loc[y_train.index])

# Create stacking training dataset with enhanced features
stack_X_train = pd.DataFrame({
    "garch": garch_train,
    "egarch": egarch_train,
    "gjr": gjr_train,
    "exo": exo_train,
    "xgb": xgb_pred_train,
    "abs_return": abs_return_train
}, index=y_train.index)

stack_y_train = y_train

# ==============================
# META MODEL (XGBOOST) - TRAIN ON TRAIN DATA ONLY
# ==============================
meta_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9
)

meta_model.fit(stack_X_train, stack_y_train)

# ==============================
# STACKING DATASET - META MODEL TESTING
# ==============================
garch_test = garch_stack.loc[y_test.index]
egarch_test = egarch_vol.loc[y_test.index]
gjr_test = gjr_vol.loc[y_test.index]
exo_test = exo_vol.loc[y_test.index]

# Market signal: absolute returns
abs_return_test = np.abs(df_feat['returns'].loc[y_test.index])

stack_X = pd.DataFrame({
    "garch": garch_test,
    "egarch": egarch_test,
    "gjr": gjr_test,
    "exo": exo_test,
    "xgb": xgb_pred,
    "abs_return": abs_return_test
}, index=y_test.index)

stack_y = y_test

meta_pred = meta_model.predict(stack_X)

# ==============================
# EVALUATION
# ==============================
rmse = np.sqrt(mean_squared_error(y_test, meta_pred))

print("\n===== STACKED MODEL PERFORMANCE =====")
print("RMSE:", rmse)

# ==============================
# SAVE MODELS
# ==============================
joblib.dump(xgb.model, "xgb_model.pkl")
joblib.dump(meta_model, "meta_model.pkl")
joblib.dump(garch, "garch.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_cols, "features.pkl")

print("\n✅ Stacking Training Complete & Saved")