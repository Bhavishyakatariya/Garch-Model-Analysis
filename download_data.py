import yfinance as yf
import pandas as pd
import os

def fetch_nifty50_data():
    # Nifty 50 has great data quality even back to 2001
    start_date = "2001-01-01"
    end_date = "2024-12-31"

    print("Downloading Nifty 50 (^NSEI) data...")
    # multi_level_index=False is vital for Python 3.13 / latest yfinance
    nifty = yf.download("^NSEI", start=start_date, end=end_date, multi_level_index=False)

    if nifty.empty:
        print("❌ Error: Could not download Nifty 50 data.")
        return

    print("Downloading macro indicators...")
    it = yf.download("^CNXIT", start=start_date, end=end_date, multi_level_index=False)
    crude = yf.download("CL=F", start=start_date, end=end_date, multi_level_index=False)
    gold = yf.download("GC=F", start=start_date, end=end_date, multi_level_index=False)
    usd = yf.download("INR=X", start=start_date, end=end_date, multi_level_index=False)

    # 1. Build base dataframe
    df = nifty[['Open', 'High', 'Low', 'Close']].copy()

    # 2. Add features (Aligned by Date)
    df["NIFTY_IT"] = it["Close"]
    df["CrudeOil"] = crude["Close"]
    df["Gold"] = gold["Close"]
    df["USDINR"] = usd["Close"]

    # 3. Feature Engineering
    df["intraday_return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["overnight_return"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]

    # 4. Modern Cleaning (ffill() replaces the deprecated fillna method)
    df = df.ffill().dropna()

    # 5. Save
    output_path = "data/processed/nifty50_dataset.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    
    print(f"✅ Success! Saved {len(df)} rows to {output_path}")
    return df

if __name__ == "__main__":
    fetch_nifty50_data()