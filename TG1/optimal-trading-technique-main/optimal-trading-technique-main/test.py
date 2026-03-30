import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import os

# =============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================
path_attached = '/mnt/data/all_prices.csv'
path_local = 'all_prices.csv'
path_dl = os.path.join(os.path.expanduser('~'), 'Downloads', 'all_prices.csv')

if os.path.exists(path_attached):
    df = pd.read_csv(path_attached, parse_dates=['Date'])
    print("Fichier attaché chargé")
elif os.path.exists(path_dl):
    df = pd.read_csv(path_dl, parse_dates=['Date'])
    print("Dossier Downloads")
elif os.path.exists(path_local):
    df = pd.read_csv(path_local, parse_dates=['Date'])
    print("Dossier local")
else:
    raise FileNotFoundError("Le fichier all_prices.csv est introuvable.")

df = df.set_index('Date')
df = df.dropna(how='all')
df = df.sort_index()

print("\n" + "="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"Shape              : {df.shape[0]} lignes x {df.shape[1]} colonnes")
print(f"Date range         : {df.index.min().date()}  ->  {df.index.max().date()}")
print(f"Assets disponibles : {', '.join(df.columns)}")

df.to_csv('Donnees_Preparees_Final.csv', index=True)

# =============================================================================
# SÉLECTION DES CANDIDATS (COINTÉGRATION)
# =============================================================================
anchor_name = 'ALI_F'
market_proxy = 'XLB'

excluded_assets = [anchor_name, market_proxy, 'PICK', 'DBB']
candidates = [c for c in df.columns if c not in excluded_assets]

results = []

for stock in candidates:
    pair_data = df[[stock, anchor_name, market_proxy]].dropna()

    if pair_data.empty:
        continue

    if (pair_data[[stock, anchor_name]] <= 0).any().any():
        continue

    stock_log = np.log(pair_data[stock])
    anc_log = np.log(pair_data[anchor_name])

    score, pvalue, _ = coint(stock_log, anc_log)

    stock_ret = pair_data[stock].pct_change().dropna()
    market_ret = pair_data[market_proxy].pct_change().dropna()
    idx = stock_ret.index.intersection(market_ret.index)

    if len(idx) > 1:
        model_beta = sm.OLS(stock_ret.loc[idx], sm.add_constant(market_ret.loc[idx])).fit()
        beta = model_beta.params.iloc[1]
    else:
        beta = np.nan

    corr_val = pair_data[stock].corr(pair_data[anchor_name])

    results.append({
        'Stock': stock,
        'P-Value': pvalue,
        'Beta': beta,
        'Corr': corr_val
    })

df_res = pd.DataFrame(results).sort_values(by='P-Value').reset_index(drop=True)
best_candidates = df_res.head(5)['Stock'].tolist()

print("\n" + "="*70)
print("BASKET SELECTION")
print("="*70)
print(f"Anchor asset       : {anchor_name}")
print(f"Benchmark / Hedge  : {market_proxy}")
print("\nTop 5 cointegrated stocks selected:\n")

selection_display = df_res.head(5).copy()
selection_display['P-Value'] = selection_display['P-Value'].map(lambda x: f"{x:.4f}")
selection_display['Beta'] = selection_display['Beta'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
selection_display['Corr'] = selection_display['Corr'].map(lambda x: f"{x:.2f}")

print(selection_display.to_string(index=False))
print(f"\nFinal basket       : {best_candidates}")

# =============================================================================
# GRAPHE 1 : PRIX NORMALISÉS
# =============================================================================
assets_to_plot = [anchor_name] + best_candidates + [market_proxy]
prices_plot = df[assets_to_plot].dropna()
normalized_prices = prices_plot / prices_plot.iloc[0] * 100

plt.figure(figsize=(14, 7))
for col in normalized_prices.columns:
    plt.plot(normalized_prices.index, normalized_prices[col], label=col)

plt.title("Évolution des prix normalisés des actifs sélectionnés")
plt.ylabel("Base 100")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# OPTIMISATION GUROBI
# =============================================================================
optimization_tickers = best_candidates + [market_proxy]
data_opt = df[optimization_tickers].pct_change().dropna()

Sigma = data_opt.cov().values
cov_market = data_opt.cov()[market_proxy]
var_market = data_opt[market_proxy].var()
betas_opt = (cov_market / var_market).values
n_assets = len(optimization_tickers)

m = gp.Model("Market_Neutral")
m.setParam('OutputFlag', 0)

w = m.addVars(n_assets, lb=-10.0, ub=10.0, name="w")

port_variance = gp.QuadExpr()
for i in range(n_assets):
    for j in range(n_assets):
        port_variance += w[i] * w[j] * Sigma[i, j]
m.setObjective(port_variance, GRB.MINIMIZE)

m.addConstr(gp.quicksum(w[i] for i in range(n_assets - 1)) == 1.0, "Budget_Long")
m.addConstr(gp.quicksum(w[i] * betas_opt[i] for i in range(n_assets)) == 0, "Beta_Zero")
for i in range(n_assets - 1):
    m.addConstr(w[i] >= 0)

m.optimize()

# =============================================================================
# BACKTEST ET ANALYSE
# =============================================================================
if m.status == GRB.OPTIMAL:
    weights = np.array([w[i].X for i in range(n_assets)])

    print("\n" + "="*70)
    print("OPTIMAL ALLOCATION")
    print("="*70)

    alloc_df = pd.DataFrame({
        'Asset': optimization_tickers,
        'Weight (%)': np.round(weights * 100, 2),
        'Beta': np.round(betas_opt, 2),
        'Role': ['ALPHA' if t != market_proxy else 'HEDGE' for t in optimization_tickers]
    })
    print(alloc_df.to_string(index=False))

    capital = 10000.0
    trans_cost_bps = 0.0010

    strat_returns = data_opt.dot(weights)
    daily_pnl = strat_returns * capital
    wealth_curve = capital + daily_pnl.cumsum()

    gross_exposure_entry = np.sum(np.abs(weights)) * capital
    entry_cost = gross_exposure_entry * trans_cost_bps

    wealth_curve_net = wealth_curve - entry_cost

    final_wealth = wealth_curve_net.iloc[-1]
    exit_cost = abs(final_wealth * (gross_exposure_entry / capital)) * trans_cost_bps
    final_net_wealth = final_wealth - exit_cost
    wealth_curve_net.iloc[-1] = final_net_wealth

    total_profit = final_net_wealth - capital
    total_return = total_profit / capital

    net_daily_ret = wealth_curve_net.pct_change().dropna()
    sharpe = (net_daily_ret.mean() / net_daily_ret.std()) * np.sqrt(252)
    rolling_max = wealth_curve_net.cummax()
    drawdown = (wealth_curve_net - rolling_max) / rolling_max
    max_dd = drawdown.min()

# =============================================================================
# COMPARATIF AVEC BENCHMARK
# =============================================================================
bench_ret = df[market_proxy].pct_change().dropna()
bench_ret = bench_ret.loc[wealth_curve_net.index]
bench_curve = capital * (1 + bench_ret).cumprod()

bench_profit = bench_curve.iloc[-1] - capital
bench_total_ret = (bench_curve.iloc[-1] / capital) - 1
bench_sharpe = (bench_ret.mean() / bench_ret.std()) * np.sqrt(252)
bench_dd = ((bench_curve / bench_curve.cummax()) - 1).min()

print("\n" + "="*70)
print("STRATEGY RESULTS VS BENCHMARK")
print("="*70)

summary_df = pd.DataFrame({
    'Metric': ['Final Value ($)', 'Total Profit ($)', 'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)'],
    'Portfolio': [
        f"{final_net_wealth:,.2f}",
        f"{total_profit:,.2f}",
        f"{total_return*100:.2f}",
        f"{sharpe:.2f}",
        f"{max_dd*100:.2f}"
    ],
    f'Benchmark ({market_proxy})': [
        f"{bench_curve.iloc[-1]:,.2f}",
        f"{bench_profit:,.2f}",
        f"{bench_total_ret*100:.2f}",
        f"{bench_sharpe:.2f}",
        f"{bench_dd*100:.2f}"
    ]
})

print(summary_df.to_string(index=False))

print("\nTransaction costs:")
print(f"Entry cost         : ${entry_cost:,.2f}")
print(f"Exit cost          : ${exit_cost:,.2f}")
print(f"Total trading costs: ${entry_cost + exit_cost:,.2f}")

# =============================================================================
# GRAPHE 2 : PORTEFEUILLE VS BENCHMARK
# =============================================================================
plt.figure(figsize=(12, 6))
plt.plot(wealth_curve_net.index, wealth_curve_net.values, linewidth=2, label='Portfolio')
plt.plot(bench_curve.index, bench_curve.values, linestyle='--', alpha=0.8, label=f'Benchmark {market_proxy}')
plt.title("Évolution de la valeur du portefeuille vs benchmark")
plt.ylabel("Valeur du portefeuille ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()