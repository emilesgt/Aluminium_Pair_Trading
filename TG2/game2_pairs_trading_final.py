"""
Trading Game #2 - Student version (improved)
Pairs trading between the anchor (ALI_F) and the most cointegrated equity
using the CSV files stored in ../TG1/data_ohlcv/

Run from TG2 with:
    python game2_pairs_trading_final.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# =========================
# 1) PARAMETERS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.normpath(os.path.join(BASE_DIR, "..", "TG1", "data_ohlcv"))
output_folder = os.path.join(BASE_DIR, "results_game2")
os.makedirs(output_folder, exist_ok=True)

anchor = "ALI_F"
equities = [
    "AA", "ACH", "CENX", "CSTM", "HINDALCO.NS",
    "KALU", "NHYDY", "RIO", "S32.AX", "1211.HK"
]

lookbacks = [20, 30, 40, 60, 90, 120]
train_ratio = 0.75
entry_z = 1.5
exit_z = 0.5
vol_lookback = 30
vol_threshold = 1.5
trailing_stop = 0.025


# =========================
# 2) LOAD ALL CSV FILES
# =========================

def load_price_data(folder, tickers):
    all_data = []

    for ticker in tickers:
        file_path = os.path.join(folder, f"{ticker}.csv")
        temp = pd.read_csv(file_path)

        temp["Date"] = pd.to_datetime(temp["Date"])
        temp = temp[["Date", "Close"]].copy()
        temp = temp.rename(columns={"Close": ticker})

        all_data.append(temp)

    df = all_data[0]
    for temp in all_data[1:]:
        df = pd.merge(df, temp, on="Date", how="inner")

    df = df.sort_values("Date").reset_index(drop=True)
    return df


# =========================
# 3) COINTEGRATION TEST
# =========================

def test_pair_cointegration(data, anchor_col, equity_col, train_ratio=0.75):
    pair_df = data[["Date", anchor_col, equity_col]].dropna().copy()

    split = int(len(pair_df) * train_ratio)
    train = pair_df.iloc[:split].copy()

    x = np.log(train[anchor_col])
    y = np.log(train[equity_col])

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    alpha = model.params["const"]
    beta = model.params[anchor_col]

    spread = y - (alpha + beta * x)
    adf_result = adfuller(spread.dropna(), regression="c", autolag="AIC")
    p_value = adf_result[1]

    return {
        "equity": equity_col,
        "p_value": p_value,
        "beta": beta,
        "alpha": alpha
    }


# =========================
# 4) Z-SCORE FUNCTION
# =========================

def compute_zscore(series, lookback):
    rolling_mean = series.rolling(lookback).mean()
    rolling_std = series.rolling(lookback).std()
    zscore = (series - rolling_mean) / rolling_std
    return zscore


# =========================
# 5) BACKTEST FUNCTION
# =========================

def backtest_strategy(data, anchor_col, equity_col, beta, lookback,
                      entry_z=1.5, exit_z=0.5,
                      vol_lookback=30, vol_threshold=1.5,
                      trailing_stop=0.025):
    df_bt = data.copy()

    df_bt["zscore"] = compute_zscore(df_bt["spread"], lookback)
    df_bt["spread_ret"] = df_bt["spread"].diff()
    df_bt["vol"] = df_bt["spread_ret"].rolling(vol_lookback).std()

    median_vol = df_bt["vol"].median()
    df_bt["vol_ok"] = df_bt["vol"] < vol_threshold * median_vol

    df_bt["asset_ret"] = df_bt[equity_col].pct_change()
    df_bt["anchor_ret"] = df_bt[anchor_col].pct_change()
    df_bt["pair_ret"] = df_bt["asset_ret"] - beta * df_bt["anchor_ret"]

    position = 0
    positions = []
    strategy_returns = []
    entries = []
    exits = []

    trade_pnl = 0
    best_trade_pnl = 0

    for i in range(len(df_bt)):
        if i == 0:
            positions.append(0)
            strategy_returns.append(0)
            entries.append(0)
            exits.append(0)
            continue

        z = df_bt["zscore"].iloc[i - 1]
        vol_ok = df_bt["vol_ok"].iloc[i - 1]
        pair_ret = df_bt["pair_ret"].iloc[i]

        daily_ret = position * pair_ret
        entry_flag = 0
        exit_flag = 0

        if position != 0:
            trade_pnl += daily_ret
            if trade_pnl > best_trade_pnl:
                best_trade_pnl = trade_pnl

        # trailing stop
        if position != 0 and (best_trade_pnl - trade_pnl > trailing_stop):
            position = 0
            trade_pnl = 0
            best_trade_pnl = 0
            exit_flag = 1

        # normal exit
        elif position != 0 and abs(z) < exit_z:
            position = 0
            trade_pnl = 0
            best_trade_pnl = 0
            exit_flag = 1

        # entries
        elif position == 0 and vol_ok:
            if z < -entry_z:
                position = 1
                trade_pnl = 0
                best_trade_pnl = 0
                entry_flag = 1
            elif z > entry_z:
                position = -1
                trade_pnl = 0
                best_trade_pnl = 0
                entry_flag = 1

        positions.append(position)
        strategy_returns.append(daily_ret)
        entries.append(entry_flag)
        exits.append(exit_flag)

    df_bt["position"] = positions
    df_bt["strategy_ret"] = strategy_returns
    df_bt["entry_flag"] = entries
    df_bt["exit_flag"] = exits
    return df_bt


# =========================
# 6) PERFORMANCE METRICS
# =========================

def performance_metrics(returns):
    returns = returns.fillna(0)

    equity_curve = (1 + returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(252)

    if annual_vol != 0:
        sharpe = annual_return / annual_vol
    else:
        sharpe = np.nan

    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_drawdown = drawdown.min()

    if max_drawdown != 0:
        calmar = annual_return / abs(max_drawdown)
    else:
        calmar = np.nan

    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Calmar": calmar
    }


def print_metrics(title, metrics_dict):
    print(f"\n=== {title} ===")
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


# =========================
# 7) MAIN PART
# =========================

def main():
    print("Data folder:", data_folder)
    print("Output folder:", output_folder)

    tickers = [anchor] + equities
    df = load_price_data(data_folder, tickers)

    for col in tickers:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    # cointegration screening
    results = []
    for eq in equities:
        try:
            res = test_pair_cointegration(df, anchor, eq, train_ratio=train_ratio)
            results.append(res)
        except Exception as e:
            print(f"Problem with {eq}: {e}")

    screening = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    screening.to_csv(os.path.join(output_folder, "tg2_cointegration_screening.csv"), index=False)

    print("\n=== COINTEGRATION SCREENING ===")
    print(screening)

    best_equity = screening.loc[0, "equity"]
    best_beta = screening.loc[0, "beta"]
    best_alpha = screening.loc[0, "alpha"]

    print("\nBest equity:", best_equity)
    print("Best beta:", round(best_beta, 4))
    print("Best alpha:", round(best_alpha, 4))

    # build spread for the best pair
    pair_df = df[["Date", anchor, best_equity]].dropna().copy()
    pair_df["log_anchor"] = np.log(pair_df[anchor])
    pair_df["log_equity"] = np.log(pair_df[best_equity])
    pair_df["spread"] = pair_df["log_equity"] - (best_alpha + best_beta * pair_df["log_anchor"])

    split = int(len(pair_df) * train_ratio)
    train_df = pair_df.iloc[:split].copy()
    test_df = pair_df.iloc[split:].copy()

    # grid search on lookback window
    grid_results = []

    for lb in lookbacks:
        bt_train = backtest_strategy(
            train_df,
            anchor_col=anchor,
            equity_col=best_equity,
            beta=best_beta,
            lookback=lb,
            entry_z=entry_z,
            exit_z=exit_z,
            vol_lookback=vol_lookback,
            vol_threshold=vol_threshold,
            trailing_stop=trailing_stop
        )

        ret = bt_train["strategy_ret"].fillna(0)
        mean_ret = ret.mean()
        vol_ret = ret.std()

        if vol_ret != 0:
            sharpe = (mean_ret / vol_ret) * np.sqrt(252)
        else:
            sharpe = np.nan

        total_return = (1 + ret).cumprod().iloc[-1] - 1

        grid_results.append({
            "lookback": lb,
            "train_sharpe": sharpe,
            "train_total_return": total_return
        })

    grid_df = pd.DataFrame(grid_results).sort_values("train_sharpe", ascending=False).reset_index(drop=True)
    grid_df.to_csv(os.path.join(output_folder, "tg2_lookback_grid.csv"), index=False)

    print("\n=== LOOKBACK GRID SEARCH ===")
    print(grid_df)

    best_lookback = int(grid_df.loc[0, "lookback"])
    print("\nBest lookback:", best_lookback)

    # final backtest on test sample
    bt_test = backtest_strategy(
        test_df,
        anchor_col=anchor,
        equity_col=best_equity,
        beta=best_beta,
        lookback=best_lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        vol_lookback=vol_lookback,
        vol_threshold=vol_threshold,
        trailing_stop=trailing_stop
    )

    strategy_metrics = performance_metrics(bt_test["strategy_ret"])
    bh_returns = test_df[best_equity].pct_change().fillna(0)
    bh_metrics = performance_metrics(bh_returns)

    n_entries = int(bt_test["entry_flag"].sum())
    n_exits = int(bt_test["exit_flag"].sum())

    print_metrics("STRATEGY METRICS", strategy_metrics)
    print_metrics("BUY AND HOLD METRICS", bh_metrics)
    print(f"\nNumber of entries: {n_entries}")
    print(f"Number of exits: {n_exits}")

    summary_df = pd.DataFrame([
        {
            "Model": "Strategy",
            **strategy_metrics,
            "Chosen Pair": f"{anchor} vs {best_equity}",
            "Best Lookback": best_lookback,
            "Entries": n_entries,
            "Exits": n_exits
        },
        {
            "Model": f"Buy&Hold {best_equity}",
            **bh_metrics,
            "Chosen Pair": f"{anchor} vs {best_equity}",
            "Best Lookback": best_lookback,
            "Entries": np.nan,
            "Exits": np.nan
        }
    ])

    summary_df.to_csv(os.path.join(output_folder, "tg2_performance_summary.csv"), index=False)
    bt_test.to_csv(os.path.join(output_folder, "tg2_backtest_test_sample.csv"), index=False)

    # plots
    bt_test = bt_test.copy()
    test_df = test_df.copy()

    bt_test["strategy_equity"] = (1 + bt_test["strategy_ret"].fillna(0)).cumprod()
    test_df["bh_equity"] = (1 + bh_returns).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(bt_test["Date"], bt_test["strategy_equity"], label="Strategy")
    plt.plot(test_df["Date"], test_df["bh_equity"], label=f"Buy & Hold {best_equity}")
    plt.title("Out-of-sample performance")
    plt.xlabel("Date")
    plt.ylabel("Cumulative performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "tg2_performance.png"), dpi=300, bbox_inches="tight")
    plt.show()

    pair_df["zscore"] = compute_zscore(pair_df["spread"], best_lookback)

    plt.figure(figsize=(12, 6))
    plt.plot(pair_df["Date"], pair_df["zscore"], label="Z-score")
    plt.axhline(entry_z, linestyle="--")
    plt.axhline(-entry_z, linestyle="--")
    plt.axhline(exit_z, linestyle="--")
    plt.axhline(-exit_z, linestyle="--")
    plt.title(f"Z-score of spread: {anchor} vs {best_equity}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "tg2_zscore.png"), dpi=300, bbox_inches="tight")
    plt.show()

    print("\nFiles saved in results_game2:")
    for file_name in os.listdir(output_folder):
        print("-", file_name)


if __name__ == "__main__":
    main()
