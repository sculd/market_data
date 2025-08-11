"""
Technical Indicators Feature Module

This module provides functions for calculating technical indicators like RSI,
true range, and autocorrelation.
"""

import pandas as pd
import numpy as np
import logging
import datetime
import numba as nb
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from market_data.feature.registry import register_feature
from market_data.feature.param import FeatureParam
from market_data.feature.impl.common_calc import _calculate_rolling_std_numba, _calculate_rolling_mean_numba

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

@nb.njit(cache=True)
def _calculate_zscore_numba(values, mean, std):
    """
    Calculate z-score using Numba.
    
    Args:
        values: Array of input values
        mean: Array of mean values
        std: Array of standard deviation values
        
    Returns:
        Array of z-score values capped to the range [-10, 10]
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    # Add a minimum threshold for standard deviation to avoid division by near-zero
    min_std = 1e-8
    
    for i in range(n):
        if not (np.isnan(values[i]) or np.isnan(mean[i]) or np.isnan(std[i])):
            # Use max(std[i], min_std) to avoid division by very small numbers
            effective_std = max(std[i], min_std)
            # Calculate z-score
            zscore = (values[i] - mean[i]) / effective_std
            # Cap z-score to the range [-10, 10]
            if zscore > 10:
                zscore = 10
            elif zscore < -10:
                zscore = -10
            result[i] = zscore
    
    return result

@dataclass
class IndicatorsParams(FeatureParam):
    """Parameters for technical indicators calculations."""
    rsi_period: int = 14
    autocorr_lag: int = 1
    autocorr_window: int = 14
    zscore_window: int = 20
    minmax_window: int = 20
    price_col: str = "close"
    
    def get_warm_up_period(self) -> datetime.timedelta:
        max_window = max(
            self.rsi_period,
            self.autocorr_window,
            self.zscore_window,
            self.minmax_window
        )
        return datetime.timedelta(minutes=max_window)

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
        days_needed = math.ceil(max_window / (24 * 60))
        
        return max(1, days_needed)  # At least 1 day
    
    def to_str(self) -> str:
        """Convert parameters to string format: rsi_period:14,autocorr_lag:1,autocorr_window:14,zscore_window:20,minmax_window:20,price_col:close"""
        return f"rsi_period:{self.rsi_period},autocorr_lag:{self.autocorr_lag},autocorr_window:{self.autocorr_window},zscore_window:{self.zscore_window},minmax_window:{self.minmax_window},price_col:{self.price_col}"
    
    @classmethod
    def from_str(cls, feature_label_str: str) -> 'IndicatorsParams':
        """Parse Indicators parameters from JSON-like format: rsi_period:14,autocorr_lag:1,autocorr_window:14,zscore_window:20,minmax_window:20,price_col:close"""
        params = {}
        for pair in feature_label_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'rsi_period':
                    params['rsi_period'] = int(value)
                elif key == 'autocorr_lag':
                    params['autocorr_lag'] = int(value)
                elif key == 'autocorr_window':
                    params['autocorr_window'] = int(value)
                elif key == 'zscore_window':
                    params['zscore_window'] = int(value)
                elif key == 'minmax_window':
                    params['minmax_window'] = int(value)
                elif key == 'price_col':
                    params['price_col'] = value
        return cls(**params)

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
        if 'open' not in df.columns:
            raise ValueError("'open' column not found in DataFrame")
        if 'high' not in df.columns:
            raise ValueError("'high' column not found in DataFrame")
        if 'low' not in df.columns:
            raise ValueError("'low' column not found in DataFrame")
        if 'close' not in df.columns:
            raise ValueError("'close' column not found in DataFrame")
        
        # Check if 'symbol' column exists
        has_symbol = 'symbol' in df.columns
        if not has_symbol:
            logger.warning("DataFrame does not have a 'symbol' column, will use a single default symbol")
            df = df.copy()
            df['symbol'] = 'default'
        
        # Create list to store DataFrames for each symbol
        results = []
        
        # Process each symbol separately
        for symbol, group_df in df.groupby('symbol'):
            # Create result DataFrame for this symbol
            symbol_result = pd.DataFrame(index=group_df.index)
            
            # Add symbol column
            symbol_result['symbol'] = symbol
            
            # Extract arrays for Numba functions
            open_prices = group_df['open'].values
            high = group_df['high'].values
            low = group_df['low'].values
            close = group_df['close'].values
            
            # Calculate RSI
            rsi = _calculate_rsi_numba(close, params.rsi_period)
            symbol_result['rsi'] = rsi
            
            # Calculate open/close ratio
            oc_ratio = _calculate_open_close_ratio_numba(open_prices, close)
            symbol_result['open_close_ratio'] = oc_ratio
            
            # Calculate autocorrelation with lag
            lagged = _calculate_lagged_series_numba(close, params.autocorr_lag)
            autocorr = _calculate_rolling_correlation_numba(close, lagged, params.autocorr_window)
            symbol_result['autocorr_lag1'] = autocorr
            
            # Calculate high-low range percentage
            hl_range_pct = _calculate_hl_range_pct_numba(high, low, close)
            symbol_result['hl_range_pct'] = hl_range_pct
            
            # Calculate z-score
            mean = _calculate_rolling_mean_numba(close, params.zscore_window)
            std = _calculate_rolling_std_numba(close, params.zscore_window)
            zscore = _calculate_zscore_numba(close, mean, std)
            symbol_result['close_zscore'] = zscore
            
            # Calculate min-max scaling
            roll_min, roll_max = _calculate_rolling_min_max_numba(close, params.minmax_window)
            minmax = _calculate_minmax_scaling_numba(close, roll_min, roll_max)
            symbol_result['close_minmax'] = minmax
            
            # Add to results list
            results.append(symbol_result)
        
        # Combine all symbol results
        result = pd.concat(results).reset_index().set_index(['timestamp', 'symbol'])
        
        return result 