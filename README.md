# Aluminium Pair Trading

A quantitative pairs trading project focused on aluminium-related assets. This project implements a full systematic pipeline — from universe screening through to out-of-sample backtesting — using Python, Jupyter Notebooks, and R.

---

## Overview

Pairs trading is a market-neutral strategy that exploits mean-reverting relationships between two correlated assets. This project applies that framework to the aluminium commodity space, identifying tradeable pairs and backtesting a spread/z-score entry-exit strategy.

The project is structured across two tutorial groups (`TG1` and `TG2`), likely corresponding to progressive stages of the analysis.

---

## Repository Structure

```
Aluminium_Pair_Trading/
│
├── TG1/                          # Tutorial Group 1 — initial analysis
├── TG2/                          # Tutorial Group 2 — extended/advanced analysis
├── ESILV_COMMO_Tutorial_1.pdf    # Assignment brief / tutorial instructions
└── README.md
```

---

## Methodology

The project follows a standard quantitative pairs trading workflow:

### 1. Universe Selection
Identify a set of aluminium-related assets (e.g. aluminium futures, aluminium producers, ETFs, mining equities) to form the candidate universe.

### 2. Correlation Screening
Compute pairwise correlations across the universe to shortlist candidate pairs exhibiting strong historical co-movement.

### 3. Cointegration Testing
Apply formal cointegration tests (e.g. Engle-Granger or Johansen) to verify that shortlisted pairs share a long-run equilibrium relationship — a stronger condition than correlation alone.

### 4. Spread & Z-Score Construction
For cointegrated pairs, compute the spread (residual of the linear regression between the two price series) and normalise it as a z-score:

```
z = (spread - mean(spread)) / std(spread)
```

### 5. Trading Strategy
Define entry and exit rules based on z-score thresholds:
- **Enter long/short** when z-score exceeds ±1 or ±2 standard deviations
- **Exit** when z-score reverts toward zero (mean reversion)

### 6. Out-of-Sample Backtesting
Evaluate strategy performance on a held-out test period to assess real-world viability. Metrics may include cumulative return, Sharpe ratio, and max drawdown.

---

## Tech Stack

| Language | Usage |
|---|---|
| Python | Core data processing, strategy logic, backtesting |
| Jupyter Notebook | Interactive analysis and visualisation |
| R | Statistical testing (cointegration, stationarity) |

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib statsmodels yfinance scipy
```

For R components, ensure the following packages are installed:
```r
install.packages(c("tseries", "urca", "ggplot2"))
```

### Running the Notebooks

1. Clone the repository:
```bash
git clone https://github.com/emilesgt/Aluminium_Pair_Trading.git
cd Aluminium_Pair_Trading
```

2. Launch Jupyter:
```bash
jupyter notebook
```

3. Open the notebooks in `TG1/` first, then `TG2/` for the full sequential workflow.

---

## Academic Context

This project was developed as part of a commodities trading course at **ESILV** (École Supérieure d'Ingénieurs Léonard de Vinci). The tutorial brief is included as `ESILV_COMMO_Tutorial_1.pdf`.

---

## License

This repository is for educational purposes. No explicit license has been applied — please contact the author before reusing or redistributing the code.

---

## Author

**emilesgt** — [GitHub Profile](https://github.com/emilesgt)
