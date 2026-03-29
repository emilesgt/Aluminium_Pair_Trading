import os
import pandas as pd
import yfinance as yf

tickers = [
    "ALI=F",
    "XLB", "PICK", "DBB",
    "AA", "CENX", "KALU", "RIO", "NHYDY", "ACH",
    "CSTM", "S32.AX", "HINDALCO.NS", "1211.HK",
    "601600.SS", "1378.HK",
]

fx_tickers = {
    "USDCNY=X": "CNY",
    "USDHKD=X": "HKD",
    "USDINR=X": "INR",
}

ASSET_CURRENCY = {
    "601600.SS":   "CNY",
    "1211.HK":     "HKD",
    "1378.HK":     "HKD",
    "HINDALCO.NS": "INR",
}

START = "2019-01-01"
END   = "2024-12-31"

all_prices = []

# ── Download asset prices ──────────────────────────────────────────────
for t in tickers:
    print("Downloading", t)
    df = yf.download(t, start=START, end=END, auto_adjust=False, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df = df[["Date", "Close"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    col_name = t.replace("=", "_")
    df = df.rename(columns={"Close": col_name})
    all_prices.append(df)

# ── Download FX rates ──────────────────────────────────────────────────
fx_rates = {}
for fx_ticker, currency in fx_tickers.items():
    print("Downloading FX:", fx_ticker)
    df = yf.download(fx_ticker, start=START, end=END, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Close"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.set_index("Date")["Close"]
    fx_rates[currency] = df

# ── Merge all asset prices on Date ────────────────────────────────────
merged_df = all_prices[0]
for df in all_prices[1:]:
    merged_df = pd.merge(merged_df, df, on="Date", how="outer")

merged_df = merged_df.sort_values("Date").reset_index(drop=True)
merged_df["Date"] = pd.to_datetime(merged_df["Date"])
merged_df = merged_df.set_index("Date")

# ── FX-convert non-USD columns to USD ─────────────────────────────────
for asset, currency in ASSET_CURRENCY.items():
    col = asset.replace("=", "_")
    if col not in merged_df.columns:
        continue
    rate = fx_rates[currency].reindex(merged_df.index, method="ffill")
    merged_df[col] = merged_df[col] / rate

# ── Drop rows with any NaN and save ───────────────────────────────────
merged_df = merged_df.reset_index()
merged_df = merged_df.dropna()

print(f"Rows after dropna: {len(merged_df)}")
print(merged_df.head())

out_path = r"C:\Users\Emile\Desktop\cours\ESILV\A4\S2\SPECIALISATION_IF\CM&M\TD1\TG1\optimal-trading-technique-main\optimal-trading-technique-main\all_prices.csv"
merged_df.to_csv(out_path, index=False)
print(f"DONE — saved to {out_path}")