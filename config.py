FEATURES = [
    'Gold', 'USDINR', 'NIFTY_IT', 'CrudeOil',
    'overnight_return', 'hl_range'
]

TARGET = 'rolling_std'

LAGS = 5

XGB_PARAMS_V1 = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8
}

XGB_PARAMS_V2 = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.03
}