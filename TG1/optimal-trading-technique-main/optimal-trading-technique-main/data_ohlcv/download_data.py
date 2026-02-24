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

for t in tickers:
    print("Downloading", t)
    df = yf.download(t, start=START, end=END, auto_adjust=False, progress=False)

    # Flatten columns if MultiIndex (safety)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Keep standard columns
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Force numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop empty rows
    df = df.dropna(subset=["Date", "Close"])

    fname = t.replace("=", "_") + ".csv"
    df.to_csv(os.path.join(out_dir, fname), index=False)

print("DONE")