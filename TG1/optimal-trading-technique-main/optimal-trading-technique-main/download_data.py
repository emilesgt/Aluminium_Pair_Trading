import os
import pandas as pd
import yfinance as yf

tickers = [
    "ALI=F",
    "XLB", "PICK", "DBB",
    "AA", "CENX", "KALU", "RIO", "NHYDY", "ACH",
    "CSTM", "S32.AX", "HINDALCO.NS", "1211.HK"
]

START = "2018-01-01"
END   = "2024-12-31"

out_dir = "data_ohlcv"
os.makedirs(out_dir, exist_ok=True)

all_prices = []

for t in tickers:
    print("Downloading", t)
    df = yf.download(t, start=START, end=END, auto_adjust=False, progress=False)

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Keep only Date + Close
    df = df[["Date", "Close"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])

    # Rename Close column with ticker name
    col_name = t.replace("=", "_")
    df = df.rename(columns={"Close": col_name})

    all_prices.append(df)

# Merge all tickers on Date
merged_df = all_prices[0]
for df in all_prices[1:]:
    merged_df = pd.merge(merged_df, df, on="Date", how="outer")

merged_df = merged_df.sort_values("Date").reset_index(drop=True)

# Optional: keep only dates where all assets exist
merged_df = merged_df.dropna()

merged_df.to_csv("all_prices.csv", index=False)

print("DONE - saved to all_prices.csv")