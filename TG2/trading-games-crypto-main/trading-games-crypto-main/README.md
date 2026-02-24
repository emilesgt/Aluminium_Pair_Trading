# Trading Games: Beating Passive Strategies in the Bullish Crypto Market

[![DOI](https://img.shields.io/badge/DOI-10.1002/fut.70018-blue)](https://doi.org/10.1002/fut.70018)
[![Journal](https://img.shields.io/badge/Journal-Journal%20of%20Futures%20Markets-green)](https://onlinelibrary.wiley.com/journal/10969934)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Data](https://img.shields.io/badge/Data-Mendeley-orange)](https://doi.org/10.17632/2kky7c6xkn.1)

## Overview

This repository contains the R code for the optimized pairs-trading strategy presented in:

> **Palazzi, R. B. (2025).** Trading Games: Beating Passive Strategies in the Bullish Crypto Market. *Journal of Futures Markets*. https://doi.org/10.1002/fut.70018

The methodology introduces systematic parameter optimization within a cointegration-based pairs-trading framework, incorporating:

- **Dynamic lookback period optimization** via grid-search
- **Adaptive trailing stop-loss** mechanisms
- **Volatility filtering** to reduce downside risk
- **Minimum holding period** constraints

## Key Results

- Annualized Sharpe ratio ≈ 2.0
- Annualized return ≈ 71%
- Superior risk-adjusted performance vs. buy-and-hold and momentum strategies
- Positive returns in both bull and bear market regimes

## Requirements

```r
install.packages(c("xts", "zoo", "PerformanceAnalytics"))
```

## Usage

```r
# Load required libraries
library(xts)
library(zoo)
library(PerformanceAnalytics)

# Source the strategy
source("pairs_trading_strategy.R")

# Prepare your price data as xts object with two columns (asset prices)
# Y <- log(prices)

# Run the strategy
results <- analyze_pairs_trading(
  Y,
  split_ratio = 0.75,
  threshold_value = 0.7,
  transaction_cost = 0.002,
  trailing_stop_factor = 0.025,
  min_holding_period = 5,
  vol_lookback = 30,
  vol_threshold = 1.5
)

# Access results
results$oos_sharpe      # Out-of-sample Sharpe ratio
results$oos_cumret      # Cumulative returns
results$best_lookback   # Optimal lookback period
results$returns         # Return series
```

## ShnyApp 
[![ShinyApps](https://img.shields.io/badge/Shiny-shinyapps.io-blue?logo=r)](https://rafaelpalazzi.shinyapps.io/trading-games-crypto/)

## Data

The cryptocurrency dataset used in the paper is available at Mendeley Data:

> Palazzi, R. B. (2024). Trading Games: Beating Passive Strategies in the Bullish Crypto Market [Dataset]. Mendeley Data. https://doi.org/10.17632/2kky7c6xkn.1

## Citation

**If you use this code in your research, please cite the paper:**

```bibtex
@article{palazzi2025trading,
  title={Trading Games: Beating Passive Strategies in the Bullish Crypto Market},
  author={Palazzi, Rafael Baptista},
  journal={Journal of Futures Markets},
  year={2025},
  publisher={Wiley},
  doi={10.1002/fut.70018}
}
```

You can also use GitHub's "Cite this repository" button on the right sidebar.

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](LICENSE). You are free to use, share, and adapt this code, provided you give appropriate credit by citing the paper above.

## Author

**Rafael Baptista Palazzi, Ph.D.**  
Faculdade de Economia, Administração e Contabilidade (FEA)  
Universidade de São Paulo (USP)  
📧 palazzi@usp.br

## Acknowledgments

This study was supported by the Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq), Process 152052/2022-4.
