from xgboost import XGBRegressor

class XGBModel:

    def __init__(self, params):
        self.model = XGBRegressor(**params)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)