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
# GARCH OUTPUT
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
# ALIGN GARCH
# ==============================
egarch_test = egarch_vol.loc[test_index]
gjr_test = gjr_vol.loc[test_index]
exo_test = exo_vol.loc[test_index]

# ==============================
# STACK MODEL
# ==============================
xgb_test_pred = xgb_model.predict(X_test)

garch_test_stack = 0.4*egarch_test + 0.3*gjr_test + 0.3*exo_test

abs_return_test = abs(df_feat.loc[test_index, 'returns'])

stack_test = pd.DataFrame({
    "garch": garch_test_stack,
    "egarch": egarch_test,
    "gjr": gjr_test,
    "exo": exo_test,
    "xgb": xgb_test_pred,
    "abs_return": abs_return_test
}, index=test_index)

stack_pred = meta_model.predict(stack_test)

# ==============================
# STANDARDIZED RMSE FUNCTION
# ==============================
def compute_srmse(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    std_dev = np.std(actual)
    
    if std_dev == 0:
        return 0
    
    return rmse / std_dev

# ==============================
# CALCULATE SRMSE
# ==============================
print("\n===== STANDARDIZED RMSE (SRMSE) =====")

print(f"EGARCH SRMSE     : {compute_srmse(y_test, egarch_test):.6f}")
print(f"GJR-GARCH SRMSE  : {compute_srmse(y_test, gjr_test):.6f}")
print(f"EGARCH-X SRMSE   : {compute_srmse(y_test, exo_test):.6f}")
print(f"XGBoost SRMSE    : {compute_srmse(y_test, xgb_test_pred):.6f}")
print(f"STACK MODEL SRMSE: {compute_srmse(y_test, stack_pred):.6f}")