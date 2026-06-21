import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from data_pipeline import DataPipeline
from garch_models import GarchModels
from feature_engineering import FeatureEngineer
from xgboost_model import XGBModel

# ==============================
# CONFIG
# ==============================

FEATURES = [
    'Gold', 'USDINR', 'NIFTY_IT', 'CrudeOil',
    'overnight_return', 'hl_range'
]

LAGS = 10
TRAIN_WINDOW = 1500
TEST_WINDOW = 200

# ==============================
# LOAD DATA
# ==============================

df_raw = pd.read_csv("nifty50_dataset.csv")

dp = DataPipeline()
df = dp.preprocess(df_raw)

# ==============================
# WALK-FORWARD LOOP
# ==============================

predictions = []
actuals = []

start = TRAIN_WINDOW

while start + TEST_WINDOW < len(df):

    train_df = df.iloc[start-TRAIN_WINDOW:start]
    test_df = df.iloc[start:start+TEST_WINDOW]

    # ==========================
    # GARCH TRAIN
    # ==========================
    garch = GarchModels()

    returns = train_df['returns'] * 10
    exog = train_df[FEATURES] * 10

    garch.fit(returns, exog)
    vol_train = garch.get_volatility()

    # ==========================
    # FEATURE ENGINEERING (TRAIN)
    # ==========================
    fe = FeatureEngineer()
    train_feat = fe.create_features(train_df, vol_train, lags=LAGS)

    train_feat['overnight_return'] = train_df['overnight_return']
    train_feat['hl_range'] = train_df['hl_range']
    train_feat['intraday_return'] = train_df['intraday_return']

    train_feat = train_feat.dropna()

    y_train = train_feat['rolling_std']
    X_train = train_feat.drop(columns=['rolling_std', 'returns'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # ==========================
    # TRAIN XGBOOST
    # ==========================
    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05
    }

    model = XGBModel(params)
    model.train(X_train_scaled, y_train)

    # ==========================
    # TEST DATA PROCESSING
    # ==========================
    returns_test = test_df['returns'] * 10
    exog_test = test_df[FEATURES] * 10

    # IMPORTANT: refit GARCH on combined data
    garch.fit(
        pd.concat([train_df['returns'], test_df['returns']]) * 10,
        pd.concat([train_df[FEATURES], test_df[FEATURES]]) * 10
    )

    vol_test = garch.get_volatility()

    test_feat = fe.create_features(test_df, vol_test, lags=LAGS)

    test_feat['overnight_return'] = test_df['overnight_return']
    test_feat['hl_range'] = test_df['hl_range']
    test_feat['intraday_return'] = test_df['intraday_return']

    test_feat = test_feat.dropna()

    # Align columns
    X_test = test_feat[X_train.columns]
    X_test_scaled = scaler.transform(X_test)

    y_test = test_feat['rolling_std']

    # ==========================
    # PREDICT
    # ==========================
    preds = model.predict(X_test_scaled)

    predictions.extend(preds)
    actuals.extend(y_test.values)

    start += TEST_WINDOW

# ==============================
# FINAL EVALUATION
# ==============================

rmse = np.sqrt(mean_squared_error(actuals, predictions))

print("\n===== WALK-FORWARD RESULT =====")
print("RMSE:", rmse)

# ==============================
# CORRELATION
# ==============================

corr = np.corrcoef(actuals, predictions)[0,1]

print("Correlation:", corr)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(actuals, label="Actual Volatility")
plt.plot(predictions, label="Predicted Volatility")
plt.legend()
plt.title("Walk-Forward Prediction")
plt.show()