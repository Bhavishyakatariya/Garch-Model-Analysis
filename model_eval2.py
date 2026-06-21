import numpy as np
import pandas as pd
import joblib
from scipy.stats import t

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

test_index = y_test.index

# ==============================
# GARCH ALIGN (TEST DATA ONLY)
# ==============================
egarch_test = egarch_vol.loc[test_index]
gjr_test = gjr_vol.loc[test_index]
exo_test = exo_vol.loc[test_index]

# ==============================
# PREDICTIONS
# ==============================
xgb_pred = xgb_model.predict(X_test)

# Stack model inputs
garch_stack = 0.4*egarch_test + 0.3*gjr_test + 0.3*exo_test
abs_return_test = abs(df_feat.loc[test_index, 'returns'])

stack_input = pd.DataFrame({
    "garch": garch_stack,
    "egarch": egarch_test,
    "gjr": gjr_test,
    "exo": exo_test,
    "xgb": xgb_pred,
    "abs_return": abs_return_test
}, index=test_index)

stack_pred = meta_model.predict(stack_input)

# ==============================
# ACTUAL VOLATILITY
# ==============================
actual_vol = y_test

# ==============================
# SAFETY: ALIGN + CLEAN DATA
# ==============================
def align_series(*series_list):
    df = pd.concat(series_list, axis=1).dropna()
    return [df.iloc[:, i] for i in range(df.shape[1])]

actual_vol, egarch_test, gjr_test, exo_test, xgb_pred, stack_pred = align_series(
    actual_vol, egarch_test, gjr_test, exo_test, xgb_pred, stack_pred
)

egarch_vol = egarch_test
gjr_vol = gjr_test
exo_vol = exo_test

# ==============================
# QLIKE FUNCTION
# ==============================
def compute_qlike(actual, pred):
    actual = np.array(actual)
    pred = np.array(pred)

    # numerical stability
    pred = np.clip(pred, 1e-8, None)
    actual = np.clip(actual, 1e-8, None)

    return np.mean(np.log(pred) + (actual / pred))


# ==============================
# DIEBOLD-MARIANO TEST
# ==============================
def diebold_mariano(actual, pred1, pred2):
    """
    pred1 = baseline model
    pred2 = new model (stack)
    """

    actual = np.array(actual)

    # squared error loss
    e1 = (actual - pred1) ** 2
    e2 = (actual - pred2) ** 2

    d = e1 - e2  # loss difference

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    # avoid divide-by-zero
    if var_d == 0:
        return 0, 1.0

    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - t.cdf(abs(dm_stat), df=len(d) - 1))

    return dm_stat, p_value


# ==============================
# QLIKE CALCULATION
# ==============================
print("\n===== QLIKE COMPARISON =====")

qlike_egarch = compute_qlike(actual_vol, egarch_vol)
qlike_gjr = compute_qlike(actual_vol, gjr_vol)
qlike_exo = compute_qlike(actual_vol, exo_vol)
qlike_xgb = compute_qlike(actual_vol, xgb_pred)
qlike_stack = compute_qlike(actual_vol, stack_pred)

print(f"EGARCH QLIKE     : {qlike_egarch:.6f}")
print(f"GJR-GARCH QLIKE  : {qlike_gjr:.6f}")
print(f"EGARCH-X QLIKE   : {qlike_exo:.6f}")
print(f"XGBoost QLIKE    : {qlike_xgb:.6f}")
print(f"STACK MODEL QLIKE: {qlike_stack:.6f}")


# ==============================
# DIEBOLD-MARIANO TEST
# ==============================
print("\n===== DIEBOLD-MARIANO TEST =====")
print("(Stack vs Other Models)")

dm_xgb, p_xgb = diebold_mariano(actual_vol, xgb_pred, stack_pred)
dm_egarch, p_egarch = diebold_mariano(actual_vol, egarch_vol, stack_pred)
dm_exo, p_exo = diebold_mariano(actual_vol, exo_vol, stack_pred)

print(f"\nStack vs XGBoost  -> DM Stat: {dm_xgb:.4f}, p-value: {p_xgb:.6f}")
print(f"Stack vs EGARCH   -> DM Stat: {dm_egarch:.4f}, p-value: {p_egarch:.6f}")
print(f"Stack vs EGARCH-X -> DM Stat: {dm_exo:.4f}, p-value: {p_exo:.6f}")