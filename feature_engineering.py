class FeatureEngineer:

    def create_features(self, df, vol_dict, lags=10):

        df_feat = df.copy()

        # Add GARCH outputs
        for key, val in vol_dict.items():
            df_feat[key] = val

        # Lag features
        for i in range(1, lags + 1):
            df_feat[f'lag_{i}'] = df_feat['returns'].shift(i)

        # Rolling stats
        df_feat['rolling_mean'] = df_feat['returns'].rolling(5).mean()
        df_feat['rolling_std'] = df_feat['returns'].rolling(5).std()

        # 🔥 Important features
        df_feat['abs_return'] = abs(df_feat['returns'])

        return df_feat.dropna()