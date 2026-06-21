from arch import arch_model

class GarchModels:

    def fit(self, returns, exog):
        
        # EGARCH
        self.egarch = arch_model(
            returns, vol='EGARCH', p=1, q=1,
            mean='ARX', lags=1,rescale = False,dist = 't'
        ).fit(disp="off")

        # GJR
        self.gjr = arch_model(
            returns, vol='GARCH', p=1, o=1, q=1,
            mean='ARX', lags=1,rescale=False,dist = 't'
        ).fit(disp="off")

        # EGARCH-X
        self.egarch_x = arch_model(
            returns, x=exog,
            vol='EGARCH', p=2, q=2,
            mean='ARX', lags=1,rescale=False,dist = 't'
        ).fit(disp="off")

    def get_volatility(self):
        return {
            "egarch_vol": self.egarch.conditional_volatility,
            "gjr_vol": self.gjr.conditional_volatility,
            "exo_vol": self.egarch_x.conditional_volatility
        }