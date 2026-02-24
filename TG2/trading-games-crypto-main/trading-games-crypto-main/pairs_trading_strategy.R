#' Optimized Pairs Trading Strategy for Cryptocurrency Markets
#' 
#' @description
#' Implementation of the cointegration-based pairs trading strategy with 
#' dynamic lookback optimization, volatility filtering, and adaptive 
#' trailing stop-loss mechanisms.
#' 
#' @references
#' Palazzi, R. B. (2025). Trading Games: Beating Passive Strategies in the 
#' Bullish Crypto Market. Journal of Futures Markets. 
#' https://doi.org/10.1002/fut.70018
#' 
#' @author Rafael Baptista Palazzi <palazzi@usp.br>
#' @license CC BY 4.0

# Required packages
library(xts)
library(zoo)
library(PerformanceAnalytics)

#' Generate Trading Signals Based on Z-Score
#' 
#' @param Z_score Numeric vector of z-scores
#' @param threshold_long Threshold for long entry (negative value)
#' @param threshold_short Threshold for short entry (positive value)
#' @return Vector of signals: 1 (long), -1 (short), 0 (neutral)
generate_signal <- function(Z_score, threshold_long, threshold_short) {
  signal <- ifelse(Z_score <= threshold_long, 1,
                   ifelse(Z_score >= threshold_short, -1, 0))
  return(signal)
}

#' Analyze Pairs Trading Strategy
#' 
#' @description
#' Main function implementing the optimized pairs trading strategy with
#' lookback period optimization, volatility filtering, and trailing stop-loss.
#' 
#' @param Y xts object with log prices of two assets (columns)
#' @param split_ratio Train/test split ratio (default: 0.75)
#' @param threshold_value Z-score threshold for signal generation (default: 0.7)
#' @param transaction_cost Transaction cost per trade (default: 0.002)
#' @param trailing_stop_factor Base trailing stop percentage (default: 0.025)
#' @param min_holding_period Minimum days before position change (default: 5)
#' @param vol_lookback Lookback for volatility calculation (default: 30)
#' @param vol_threshold Volatility filter multiplier (default: 1.5)
#' 
#' @return List containing:
#'   \item{best_lookback}{Optimal lookback period from grid search}
#'   \item{oos_sharpe}{Out-of-sample annualized Sharpe ratio}
#'   \item{oos_cumret}{Out-of-sample cumulative return}
#'   \item{signals}{Trading signals time series}
#'   \item{returns}{Strategy returns time series}
#'   \item{z_scores}{Z-score time series}
#'   \item{train_index}{Index of last training observation}
#'   \item{threshold}{Threshold value used}
#'   \item{stop_loss_hits}{Number of stop-loss triggers}
#'   \item{volatility}{Spread volatility time series}
#' 
#' @examples
#' \dontrun{
#' # Load price data as xts with two columns
#' prices <- ... # your price data
#' Y <- log(prices)
#' 
#' results <- analyze_pairs_trading(Y)
#' print(results$oos_sharpe)
#' }
#' 
#' @export
analyze_pairs_trading <- function(Y, 
                                  split_ratio = 0.75, 
                                  threshold_value = 0.7,
                                  transaction_cost = 0.002,
                                  trailing_stop_factor = 0.025,
                                  min_holding_period = 5,
                                  vol_lookback = 30,
                                  vol_threshold = 1.5) {
  
  # Convert to matrix for safer operations
  Y_mat <- as.matrix(Y)
  Y_dates <- index(Y)
  n_obs <- nrow(Y_mat)
  
  # Data preparation - handle NAs
  if (any(is.na(Y_mat))) {
    Y_mat[, 1] <- na.approx(Y_mat[, 1], na.rm = FALSE)
    Y_mat[, 2] <- na.approx(Y_mat[, 2], na.rm = FALSE)
    # Fill any remaining NAs at edges
    Y_mat[is.na(Y_mat)] <- 0
  }
  
  # Split data
  T_trn <- round(split_ratio * n_obs)
  train_indices <- 1:T_trn
  
  # Cointegration estimation using training data
  ls_coeffs <- coef(lm(Y_mat[train_indices, 1] ~ Y_mat[train_indices, 2]))
  mu <- ls_coeffs[1]
  gamma <- ls_coeffs[2]
  
  # Create normalized portfolio weights
  w_ref <- c(1, -gamma) / (1 + abs(gamma))
  
  # Calculate spread (as numeric vector)
  spread_vec <- Y_mat[, 1] * w_ref[1] + Y_mat[, 2] * w_ref[2]
  
  # Calculate spread returns
  spread_return_vec <- c(NA, diff(spread_vec))
  
  # Calculate rolling volatility
  spread_vol_vec <- rep(NA, n_obs)
  for (i in (vol_lookback + 1):n_obs) {
    spread_vol_vec[i] <- sd(spread_return_vec[(i - vol_lookback + 1):i], na.rm = TRUE)
  }
  
  # Average volatility (excluding NAs)
  avg_vol <- mean(spread_vol_vec, na.rm = TRUE)
  if (is.na(avg_vol) || avg_vol == 0) avg_vol <- 0.01
  
  # Volatility filter
  vol_filter_vec <- spread_vol_vec <= (vol_threshold * avg_vol)
  vol_filter_vec[is.na(vol_filter_vec)] <- TRUE
  
  # Optimize lookback period on training data
  lookback_periods <- seq(5, min(360, T_trn - 10), by = 5)
  best_sharpe <- -Inf
  best_lookback <- 30  # default
  
  for (lookback in lookback_periods) {
    # Calculate rolling mean and sd for training period
    roll_mean <- rep(NA, T_trn)
    roll_sd <- rep(NA, T_trn)
    
    for (i in (lookback + 1):T_trn) {
      window_data <- spread_vec[(i - lookback + 1):i]
      roll_mean[i] <- mean(window_data, na.rm = TRUE)
      roll_sd[i] <- sd(window_data, na.rm = TRUE)
    }
    
    # Calculate Z-scores
    Z_score_train <- (spread_vec[train_indices] - roll_mean) / roll_sd
    Z_score_train[is.na(Z_score_train) | is.infinite(Z_score_train)] <- 0
    
    # Generate signals
    signal_train <- generate_signal(Z_score_train, -threshold_value, threshold_value)
    
    # Calculate returns
    signal_lagged <- c(0, signal_train[-length(signal_train)])
    traded_return <- spread_return_vec[train_indices] * signal_lagged
    traded_return <- traded_return[!is.na(traded_return)]
    
    if (length(traded_return) > 50) {
      traded_return_xts <- xts(traded_return, order.by = Y_dates[train_indices[(length(train_indices) - length(traded_return) + 1):length(train_indices)]])
      sharpe <- as.numeric(SharpeRatio.annualized(traded_return_xts, Rf = 0.0001587302))
      if (!is.na(sharpe) && is.finite(sharpe) && sharpe > best_sharpe) {
        best_sharpe <- sharpe
        best_lookback <- lookback
      }
    }
  }
  
  # Calculate rolling statistics with best lookback (full sample)
  roll_mean_all <- rep(NA, n_obs)
  roll_sd_all <- rep(NA, n_obs)
  
  for (i in (best_lookback + 1):n_obs) {
    window_data <- spread_vec[(i - best_lookback + 1):i]
    roll_mean_all[i] <- mean(window_data, na.rm = TRUE)
    roll_sd_all[i] <- sd(window_data, na.rm = TRUE)
  }
  
  # Calculate Z-scores for full sample
  Z_score_all <- (spread_vec - roll_mean_all) / roll_sd_all
  Z_score_all[is.na(Z_score_all) | is.infinite(Z_score_all)] <- 0
  
  # Generate initial signals
  initial_signal <- generate_signal(Z_score_all, -threshold_value, threshold_value)
  
  # Apply volatility filter
  initial_signal[!vol_filter_vec] <- 0
  
  # Initialize final vectors
  final_signal <- initial_signal
  final_returns <- spread_return_vec * c(0, final_signal[-n_obs])  # lagged signal
  
  # Trading simulation with stop-loss
  position_active <- FALSE
  stop_loss_hits <- 0
  holding_days <- 0
  current_position <- 0
  highest_since_entry <- NA
  lowest_since_entry <- NA
  
  # Start from where we have valid data
  start_idx <- max(best_lookback + 1, vol_lookback + 1)
  
  for (i in start_idx:n_obs) {
    sig_today <- final_signal[i]
    current_spread <- spread_vec[i]
    current_vol <- spread_vol_vec[i]
    
    if (is.na(current_vol)) current_vol <- avg_vol
    
    # Update holding period
    if (position_active) {
      holding_days <- holding_days + 1
    }
    
    # Check minimum holding period
    can_change_position <- !position_active || holding_days >= min_holding_period
    
    # Dynamic stop loss
    dynamic_stop <- trailing_stop_factor * max(current_vol / avg_vol, 1)
    
    if (position_active && can_change_position) {
      if (current_position == 1) {
        highest_since_entry <- max(highest_since_entry, current_spread, na.rm = TRUE)
        trailing_stop <- highest_since_entry * (1 - dynamic_stop)
        
        if (current_spread < trailing_stop) {
          stop_loss_hits <- stop_loss_hits + 1
          prev_spread <- spread_vec[i - 1]
          if (!is.na(prev_spread) && prev_spread != 0) {
            final_returns[i] <- (trailing_stop - prev_spread) / abs(prev_spread) - transaction_cost
          }
          final_signal[i] <- 0
          position_active <- FALSE
          holding_days <- 0
          current_position <- 0
        }
      } else if (current_position == -1) {
        lowest_since_entry <- min(lowest_since_entry, current_spread, na.rm = TRUE)
        trailing_stop <- lowest_since_entry * (1 + dynamic_stop)
        
        if (current_spread > trailing_stop) {
          stop_loss_hits <- stop_loss_hits + 1
          prev_spread <- spread_vec[i - 1]
          if (!is.na(prev_spread) && prev_spread != 0) {
            final_returns[i] <- -(trailing_stop - prev_spread) / abs(prev_spread) - transaction_cost
          }
          final_signal[i] <- 0
          position_active <- FALSE
          holding_days <- 0
          current_position <- 0
        }
      }
    }
    
    # New position entry
    if (!position_active && sig_today != 0) {
      position_active <- TRUE
      highest_since_entry <- current_spread
      lowest_since_entry <- current_spread
      current_position <- sig_today
      holding_days <- 0
      final_returns[i] <- final_returns[i] - transaction_cost
    }
  }
  
  # Handle NAs in returns
  final_returns[is.na(final_returns)] <- 0
  
  # Create xts objects for output
  returns_xts <- xts(final_returns, order.by = Y_dates)
  signal_xts <- xts(final_signal, order.by = Y_dates)
  zscore_xts <- xts(Z_score_all, order.by = Y_dates)
  vol_xts <- xts(spread_vol_vec, order.by = Y_dates)
  
  # Calculate out-of-sample metrics
  oos_returns <- returns_xts[(T_trn + 1):n_obs]
  oos_returns <- oos_returns[!is.na(oos_returns) & is.finite(coredata(oos_returns))]
  
  if (length(oos_returns) > 0) {
    oos_sharpe <- SharpeRatio.annualized(oos_returns, Rf = 0.0001587302)
    oos_cumret <- Return.cumulative(oos_returns)
  } else {
    oos_sharpe <- NA
    oos_cumret <- NA
  }
  
  # Print diagnostics
  cat("\nStrategy Parameters:\n")
  cat("Best Lookback Period:", best_lookback, "days\n")
  cat("Min Holding Period:", min_holding_period, "days\n")
  cat("Volatility Lookback:", vol_lookback, "days\n")
  cat("Volatility Threshold:", vol_threshold, "x average\n")
  cat("\nPerformance Metrics:\n")
  cat("Stop Loss Hits:", stop_loss_hits, "\n")
  cat("Average Volatility:", round(avg_vol, 6), "\n")
  cat("Out-of-Sample Sharpe:", round(as.numeric(oos_sharpe), 4), "\n")
  cat("Out-of-Sample Cumulative Return:", round(as.numeric(oos_cumret) * 100, 2), "%\n")
  
  return(list(
    best_lookback = best_lookback,
    oos_sharpe = oos_sharpe,
    oos_cumret = oos_cumret,
    signals = signal_xts,
    returns = returns_xts,
    z_scores = zscore_xts,
    train_index = T_trn,
    threshold = threshold_value,
    stop_loss_hits = stop_loss_hits,
    volatility = vol_xts
  ))
}
