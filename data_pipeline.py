import pandas as pd
import numpy as np

class DataPipeline:

    def preprocess(self, df):
        df = df.copy()

        # Date handling
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').set_index('Date')

        price_cols = ['Close', 'Gold', 'USDINR', 'NIFTY_IT', 'CrudeOil']

        for col in price_cols:
            if col in df.columns:

                # 🔥 Step 1: Remove invalid values
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Replace 0 or negative
                df[col] = df[col].apply(lambda x: np.nan if x <= 0 else x)

                # 🔥 Step 2: Fill missing properly
                df[col] = df[col].ffill().bfill()

                # 🔥 Step 3: Log return
                df[col] = np.log(df[col] / df[col].shift(1))

        # Rename target
        df.rename(columns={'Close': 'returns'}, inplace=True)

        return df.dropna()