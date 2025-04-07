"""
Technical Indicators Feature Module

This module provides functions for calculating technical indicators like RSI,
true range, and autocorrelation.
"""

import pandas as pd
import numpy as np
import logging
import numba as nb
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from market_data.feature.registry import register_feature

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "indicators"

# Numba-accelerated indicator calculations
@nb.njit(cache=True)
def _calculate_rsi_numba(prices, period):
    """
    Calculate Relative Strength Index (RSI) using Numba for performance.
    
    Args:
        prices: Array of price values
        period: Period for RSI calculation
        
    Returns:
        Array of RSI values (0-100 range)
    """
    n = len(prices)
    rsi = np.full(n, np.nan)
    
    # Calculate price changes
    deltas = np.zeros(n)
    for i in range(1, n):
        deltas[i] = prices[i] - prices[i-1]
    
    # Separate gains and losses
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(n):
        if deltas[i] > 0:
            gains[i] = deltas[i]
        elif deltas[i] < 0:
            losses[i] = -deltas[i]
    
    # Calculate RSI
    for i in range(period, n):
        avg_gain = 0.0
        avg_loss = 0.0
        
        # Calculate average gain and loss over period
        for j in range(i-period+1, i+1):
            avg_gain += gains[j]
            avg_loss += losses[j]
        
        avg_gain /= period
        avg_loss /= period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

@nb.njit(cache=True)
def _calculate_true_range_numba(high, low, close):
    """
    Calculate True Range using Numba for performance.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        
    Returns:
        Array of true range values
    """
    n = len(high)
    tr = np.full(n, np.nan)
    
    tr[0] = high[0] - low[0]  # First value is just the high-low range
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        
        tr[i] = max(hl, hc, lc)
    
    return tr

@nb.njit(cache=True)
def _calculate_open_close_ratio_numba(open_prices, close_prices):
    """
    Calculate open/close ratio using Numba.
    
    Args:
        open_prices: Array of open prices
        close_prices: Array of close prices
        
    Returns:
        Array of open/close ratio values
    """
    n = len(open_prices)
    ratio = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(open_prices[i]) or np.isnan(close_prices[i])) and close_prices[i] > 0:
            ratio[i] = open_prices[i] / close_prices[i]
    
    return ratio

@nb.njit(cache=True)
def _calculate_rolling_correlation_numba(x, y, window):
    """
    Calculate rolling correlation using Numba for performance.
    
    Args:
        x: First array of values
        y: Second array of values
        window: Window size for rolling calculation
        
    Returns:
        Array of correlation values
    """
    n = len(x)
    corr = np.full(n, np.nan)
    
    for i in range(window, n):
        x_window = x[i-window:i]
        y_window = y[i-window:i]
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))
        x_valid = x_window[valid_mask]
        y_valid = y_window[valid_mask]
        
        if len(x_valid) >= 5:  # Need at least 5 valid points for meaningful correlation
            x_mean = np.mean(x_valid)
            y_mean = np.mean(y_valid)
            x_var = np.var(x_valid)
            y_var = np.var(y_valid)
            
            if x_var > 0 and y_var > 0:
                cov = np.mean((x_valid - x_mean) * (y_valid - y_mean))
                corr[i] = cov / (np.sqrt(x_var) * np.sqrt(y_var))
    
    return corr

@nb.njit(cache=True)
def _calculate_lagged_series_numba(values, lag):
    """
    Create a lagged version of a series using Numba.
    
    Args:
        values: Array of input values
        lag: Number of periods to lag
        
    Returns:
        Array of lagged values
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(lag, n):
        result[i] = values[i-lag]
    
    return result

@nb.njit(cache=True)
def _calculate_rolling_mean_numba(values, window):
    """
    Calculate rolling mean using Numba for performance.
    
    Args:
        values: Array of input values
        window: Window size for calculation
        
    Returns:
        Array of mean values
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        window_values = values[i-window:i]
        valid_values = window_values[~np.isnan(window_values)]
        
        if len(valid_values) >= window // 2:  # Require at least half of the window to be valid
            result[i] = np.mean(valid_values)
    
    return result

@nb.njit(cache=True)
def _calculate_rolling_std_numba(values, window):
    """
    Calculate rolling standard deviation using Numba for performance.
    
    Args:
        values: Array of input values
        window: Window size for calculation
        
    Returns:
        Array of standard deviation values
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        window_values = values[i-window:i]
        valid_values = window_values[~np.isnan(window_values)]
        
        if len(valid_values) >= window // 2:  # Require at least half of the window to be valid
            result[i] = np.std(valid_values)
    
    return result

@nb.njit(cache=True)
def _calculate_zscore_numba(values, mean, std):
    """
    Calculate z-score using Numba.
    
    Args:
        values: Array of input values
        mean: Array of mean values
        std: Array of standard deviation values
        
    Returns:
        Array of z-score values
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(values[i]) or np.isnan(mean[i]) or np.isnan(std[i])) and std[i] > 0:
            result[i] = (values[i] - mean[i]) / std[i]
    
    return result

@nb.njit(cache=True)
def _calculate_rolling_min_max_numba(values, window):
    """
    Calculate rolling min and max using Numba for performance.
    
    Args:
        values: Array of input values
        window: Window size for calculation
        
    Returns:
        Tuple of (rolling_min, rolling_max) arrays
    """
    n = len(values)
    roll_min = np.full(n, np.nan)
    roll_max = np.full(n, np.nan)
    
    for i in range(window, n):
        window_values = values[i-window:i]
        valid_values = window_values[~np.isnan(window_values)]
        
        if len(valid_values) >= window // 2:  # Require at least half of the window to be valid
            roll_min[i] = np.min(valid_values)
            roll_max[i] = np.max(valid_values)
    
    return roll_min, roll_max

@nb.njit(cache=True)
def _calculate_minmax_scaling_numba(values, roll_min, roll_max):
    """
    Calculate min-max scaling using Numba.
    
    Args:
        values: Array of input values
        roll_min: Array of rolling minimum values
        roll_max: Array of rolling maximum values
        
    Returns:
        Array of min-max scaled values (0-1 range)
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(n):
        if (not np.isnan(values[i]) and not np.isnan(roll_min[i]) and 
                not np.isnan(roll_max[i]) and roll_max[i] > roll_min[i]):
            result[i] = (values[i] - roll_min[i]) / (roll_max[i] - roll_min[i])
    
    return result

@nb.njit(cache=True)
def _calculate_hl_range_pct_numba(high, low, close):
    """
    Calculate high-low range as percentage of price using Numba.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        
    Returns:
        Array of high-low range percentage values
    """
    n = len(high)
    result = np.full(n, np.nan)
    
    for i in range(n):
        if (not np.isnan(high[i]) and not np.isnan(low[i]) and 
                not np.isnan(close[i]) and close[i] > 0):
            result[i] = (high[i] - low[i]) / close[i]
    
    return result

@dataclass
class IndicatorsParams:
    """Parameters for technical indicators calculations."""
    rsi_period: int = 14
    autocorr_lag: int = 1
    autocorr_window: int = 14
    zscore_window: int = 20
    minmax_window: int = 20
    price_col: str = "close"
    
    def get_params_dir(self) -> str:
        """
        Generate a directory name string from parameters.
        
        Returns:
            Directory name string for caching
        """
        from market_data.util.cache.path import params_to_dir_name
        
        params_dict = {
            'rsi_period': self.rsi_period,
            'autocorr_lag': self.autocorr_lag,
            'autocorr_window': self.autocorr_window,
            'zscore_window': self.zscore_window,
            'minmax_window': self.minmax_window,
            'price': self.price_col
        }
        return params_to_dir_name(params_dict)
        
    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on the maximum window
        from all indicator parameters.
        
        Returns:
            int: Recommended number of warm-up days
        """
        import math
        
        # Find the maximum window from all parameters
        max_window = max(
            self.rsi_period,
            self.autocorr_window,
            self.zscore_window,
            self.minmax_window
        )
        
        # Convert to days (assuming periods are in minutes for 24/7 markets)
        # Add 1 day as buffer for safety
        days_needed = math.ceil(max_window / (24 * 60)) + 1
        
        return max(1, days_needed)  # At least 1 day

@register_feature(FEATURE_LABEL)
class IndicatorsFeature:
    """Technical indicators feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[IndicatorsParams] = None) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for indicators calculation
            
        Returns:
            DataFrame with calculated indicators
        """
        if params is None:
            params = IndicatorsParams()
            
        logger.info(f"Calculating technical indicators with RSI period={params.rsi_period}")
        
        # Ensure we have the required columns
        if params.price_col not in df.columns:
            raise ValueError(f"Price column '{params.price_col}' not found in DataFrame")
        
        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Extract price data as numpy arrays for Numba functions
        close = df[params.price_col].values
        
        # Calculate RSI using Numba
        rsi = _calculate_rsi_numba(close, params.rsi_period)
        result['rsi'] = rsi
        
        # Calculate true range if high/low are available
        if all(col in df.columns for col in ['high', 'low']):
            high = df['high'].values
            low = df['low'].values
            
            true_range = _calculate_true_range_numba(high, low, close)
            result['true_range'] = true_range
            
            # Calculate high-low range as percentage of price
            hl_range_pct = _calculate_hl_range_pct_numba(high, low, close)
            result['hl_range_pct'] = hl_range_pct
        
        # Calculate open/close ratio if open is available
        if 'open' in df.columns:
            open_prices = df['open'].values
            open_close_ratio = _calculate_open_close_ratio_numba(open_prices, close)
            result['open_close_ratio'] = open_close_ratio
        
        # Calculate autocorrelation
        lagged = _calculate_lagged_series_numba(close, params.autocorr_lag)
        autocorr = _calculate_rolling_correlation_numba(
            lagged, close, params.autocorr_window
        )
        result['autocorr_lag1'] = autocorr
        
        # Calculate z-score normalization
        mean = _calculate_rolling_mean_numba(close, params.zscore_window)
        std = _calculate_rolling_std_numba(close, params.zscore_window)
        zscore = _calculate_zscore_numba(close, mean, std)
        result['close_zscore'] = zscore
        
        # Calculate min-max scaling
        roll_min, roll_max = _calculate_rolling_min_max_numba(close, params.minmax_window)
        minmax = _calculate_minmax_scaling_numba(close, roll_min, roll_max)
        result['close_minmax'] = minmax
        
        return result 