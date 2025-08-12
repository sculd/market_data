"""
Market Regime Feature Module

This module provides functions for calculating market regime indicators
based on volatility patterns.
"""

import datetime
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numba as nb
import numpy as np
import pandas as pd

from market_data.feature.impl.common_calc import (
    _calculate_rolling_mean_numba, _calculate_rolling_std_numba)
from market_data.feature.impl.indicators import _calculate_zscore_numba
from market_data.feature.impl.returns import _calculate_simple_returns_numba
from market_data.feature.param import FeatureParam
from market_data.feature.registry import register_feature

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "market_regime"

# Numba-accelerated calculations
@nb.njit(cache=True)
def _initialize_returns_numba(returns):
    """
    Replace initial NaN values in returns with zeros.
    
    Args:
        returns: Array of return values
        
    Returns:
        Array of initialized returns
    """
    n = len(returns)
    result = returns.copy()
    
    i = 0
    while i < n and np.isnan(result[i]):
        result[i] = 0.0
        i += 1
    
    return result

@nb.njit(cache=True)
def _calculate_garch_volatility_numba(returns, omega, alpha, beta):
    """
    Calculate GARCH(1,1) volatility using Numba for performance.
    
    Args:
        returns: Array of return values (with NaNs replaced)
        omega: GARCH intercept parameter
        alpha: GARCH parameter for squared returns
        beta: GARCH parameter for lagged volatility
        
    Returns:
        Array of volatility values
    """
    n = len(returns)
    sigma2 = np.zeros(n)
    
    # Initialize with variance of non-NaN returns
    valid_returns = returns[~np.isnan(returns)]
    if len(valid_returns) > 0:
        sigma2[0] = np.var(valid_returns)
    else:
        sigma2[0] = omega  # Default value if all returns are NaN
    
    # Pre-calculate squared returns
    squared_returns = np.zeros(n)
    for i in range(n):
        if not np.isnan(returns[i]):
            squared_returns[i] = returns[i] * returns[i]
        else:
            squared_returns[i] = 0.0
    
    # Calculate GARCH volatility
    for i in range(1, n):
        if np.isnan(returns[i-1]):
            sigma2[i] = sigma2[i-1]  # Keep last value for NaN returns
        else:
            sigma2[i] = omega + alpha * squared_returns[i-1] + beta * sigma2[i-1]
    
    # Convert variance to volatility (standard deviation)
    volatility = np.sqrt(sigma2)
    
    return volatility

@nb.njit(cache=True)
def _calculate_percentile_numba(values, percentage):
    """
    Calculate percentile of values array using Numba.
    
    Args:
        values: Array of values
        percentage: Percentile (0-100) to calculate
        
    Returns:
        Percentile value
    """
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return np.nan
    
    # Sort values
    sorted_values = np.sort(valid_values)
    
    # Calculate index
    index = (percentage / 100.0) * (len(sorted_values) - 1)
    
    # Get lower and upper indices
    lower_idx = int(np.floor(index))
    upper_idx = int(np.ceil(index))
    
    # If indices are the same, return the value
    if lower_idx == upper_idx:
        return sorted_values[lower_idx]
    
    # Interpolate between lower and upper values
    fraction = index - lower_idx
    return sorted_values[lower_idx] + fraction * (sorted_values[upper_idx] - sorted_values[lower_idx])

@nb.njit(cache=True)
def _calculate_volatility_regime_numba(volatility, window):
    """
    Calculate volatility regime classification using Numba.
    
    Args:
        volatility: Array of volatility values
        window: Window size for percentile calculation
        
    Returns:
        Array of regime classifications:
        -1: Low volatility
        0: Medium volatility
        1: High volatility
    """
    n = len(volatility)
    regime = np.zeros(n)
    
    for i in range(window, n):
        # Get the window of volatility values
        window_vol = volatility[i-window:i]
        valid_window = window_vol[~np.isnan(window_vol)]
        
        if len(valid_window) < window // 4:
            continue
            
        # Calculate percentiles for this window
        low_threshold = _calculate_percentile_numba(valid_window, 33)
        high_threshold = _calculate_percentile_numba(valid_window, 66)
        
        # Classify current point
        if volatility[i] > high_threshold:
            regime[i] = 1  # High volatility
        elif volatility[i] < low_threshold:
            regime[i] = -1  # Low volatility
    
    return regime

@nb.njit(cache=True)
def _calculate_rolling_variance_numba(values, window, mean=None):
    """
    Calculate rolling variance using Numba for performance.
    
    Args:
        values: Array of values
        window: Window size for calculation
        mean: Optional pre-calculated mean array
        
    Returns:
        Array of variance values
    """
    n = len(values)
    variance = np.full(n, np.nan)
    
    for i in range(window, n):
        # Get window of values
        window_values = values[i-window:i]
        valid_values = window_values[~np.isnan(window_values)]
        
        if len(valid_values) < window // 4:  # Require at least 25% of window to be valid
            continue
            
        # Calculate mean if not provided
        if mean is None:
            window_mean = np.mean(valid_values)
        else:
            window_mean = mean[i]
            
        # Calculate variance
        squared_diff = np.sum((valid_values - window_mean) ** 2)
        variance[i] = squared_diff / len(valid_values)
    
    return variance

@nb.njit(cache=True)
def _calculate_rolling_skew_numba(values, window, mean=None, variance=None):
    """
    Calculate rolling skewness using Numba for performance.
    
    Args:
        values: Array of values
        window: Window size for calculation
        mean: Optional pre-calculated mean array
        variance: Optional pre-calculated variance array
        
    Returns:
        Array of skewness values
    """
    n = len(values)
    skewness = np.full(n, np.nan)
    
    for i in range(window, n):
        # Get window of values
        window_values = values[i-window:i]
        valid_values = window_values[~np.isnan(window_values)]
        
        if len(valid_values) < window // 4:  # Require at least 25% of window to be valid
            continue
            
        # Calculate mean if not provided
        if mean is None:
            window_mean = np.mean(valid_values)
        else:
            window_mean = mean[i]
            
        # Calculate variance if not provided
        if variance is None:
            window_variance = np.var(valid_values)
        else:
            window_variance = variance[i]
            
        if window_variance == 0:
            continue
            
        # Calculate skewness
        cubed_diff = np.sum((valid_values - window_mean) ** 3)
        n_valid = len(valid_values)
        skewness[i] = (cubed_diff / n_valid) / (window_variance ** 1.5)
    
    return skewness

@nb.njit(cache=True)
def _calculate_rolling_kurtosis_numba(values, window, mean=None, variance=None):
    """
    Calculate rolling kurtosis using Numba for performance.
    
    Args:
        values: Array of values
        window: Window size for calculation
        mean: Optional pre-calculated mean array
        variance: Optional pre-calculated variance array
        
    Returns:
        Array of excess kurtosis values (kurtosis - 3)
    """
    n = len(values)
    kurtosis = np.full(n, np.nan)
    
    for i in range(window, n):
        # Get window of values
        window_values = values[i-window:i]
        valid_values = window_values[~np.isnan(window_values)]
        
        if len(valid_values) < window // 4:  # Require at least 25% of window to be valid
            continue
            
        # Calculate mean if not provided
        if mean is None:
            window_mean = np.mean(valid_values)
        else:
            window_mean = mean[i]
            
        # Calculate variance if not provided
        if variance is None:
            window_variance = np.var(valid_values)
        else:
            window_variance = variance[i]
            
        if window_variance == 0:
            continue
            
        # Calculate kurtosis
        fourth_diff = np.sum((valid_values - window_mean) ** 4)
        n_valid = len(valid_values)
        kurtosis[i] = (fourth_diff / n_valid) / (window_variance ** 2) - 3  # Return excess kurtosis
    
    return kurtosis

@dataclass
class MarketRegimeParams(FeatureParam):
    """Parameters for market regime calculations."""
    volatility_windows: List[int] = field(default_factory=lambda: [240, 1440, 4320])  # 4h, 1d, 3d
    price_col: str = "close"
    include_mean: bool = True
    include_variance: bool = True
    include_skewness: bool = True
    include_kurtosis: bool = True
    
    def get_warm_up_period(self) -> datetime.timedelta:
        max_window = max(self.volatility_windows) if self.volatility_windows else 240
        return datetime.timedelta(minutes=max_window)

    
    def to_str(self) -> str:
        """Convert parameters to string format: volatility_windows:[240,1440],price_col:close,include_mean:true,include_variance:true,include_skewness:true,include_kurtosis:true"""
        windows_str = '[' + ','.join(str(w) for w in self.volatility_windows) + ']'
        return f"volatility_windows:{windows_str},price_col:{self.price_col},include_mean:{str(self.include_mean).lower()},include_variance:{str(self.include_variance).lower()},include_skewness:{str(self.include_skewness).lower()},include_kurtosis:{str(self.include_kurtosis).lower()}"
    
    @classmethod
    def from_str(cls, feature_label_str: str) -> 'MarketRegimeParams':
        """Parse Market Regime parameters from JSON-like format: volatility_windows:[240,1440],price_col:close,include_mean:true,include_variance:true"""
        params = {}
        for pair in feature_label_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'volatility_windows':
                    # Parse [240,1440] format
                    if value.startswith('[') and value.endswith(']'):
                        windows_str = value[1:-1]
                        params['volatility_windows'] = [int(w.strip()) for w in windows_str.split(',') if w.strip()]
                elif key == 'price_col':
                    params['price_col'] = value
                elif key == 'include_mean':
                    params['include_mean'] = value.lower() == 'true'
                elif key == 'include_variance':
                    params['include_variance'] = value.lower() == 'true'
                elif key == 'include_skewness':
                    params['include_skewness'] = value.lower() == 'true'
                elif key == 'include_kurtosis':
                    params['include_kurtosis'] = value.lower() == 'true'
        return cls(**params)

# disable for now
#@register_feature(FEATURE_LABEL)
class MarketRegimeFeature:
    """Market regime feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[MarketRegimeParams] = None) -> pd.DataFrame:
        """
        Calculate market regime indicators.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for market regime calculation
            
        Returns:
            DataFrame with calculated market regime indicators
        """
        if params is None:
            params = MarketRegimeParams()
            
        logger.info(f"Calculating market regime indicators with windows {params.volatility_windows}")
        
        # Ensure we have the price column
        if params.price_col not in df.columns:
            raise ValueError(f"Price column '{params.price_col}' not found in DataFrame")
        
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
            
            # Extract prices as numpy array for Numba functions
            prices = group_df[params.price_col].values
            
            # Calculate returns using function from returns module
            returns = _calculate_simple_returns_numba(prices, 1)
            
            # Use a standard window for the basic statistics (use the smallest window or 20 by default)
            stat_window = min(params.volatility_windows) if params.volatility_windows else 20
            
            # Calculate rolling statistics as needed
            if params.include_mean:
                mean = _calculate_rolling_mean_numba(returns, stat_window)
                symbol_result['return_mean'] = mean
            else:
                mean = None
                
            if params.include_variance:
                variance = _calculate_rolling_variance_numba(returns, stat_window, mean)
                symbol_result['return_variance'] = variance
                # Also include volatility (standard deviation)
                symbol_result['return_volatility'] = np.sqrt(np.maximum(0, variance))  # Ensure non-negative
            else:
                variance = None
                
            if params.include_skewness:
                skewness = _calculate_rolling_skew_numba(returns, stat_window, mean, variance)
                symbol_result['return_skewness'] = skewness
                
            if params.include_kurtosis:
                kurtosis = _calculate_rolling_kurtosis_numba(returns, stat_window, mean, variance)
                symbol_result['return_excess_kurtosis'] = kurtosis
            
            # Calculate volatility regimes for each window
            for window in params.volatility_windows:
                # Calculate volatility regime
                volatility = np.sqrt(np.maximum(0, _calculate_rolling_variance_numba(returns, window)))
                regime = _calculate_volatility_regime_numba(volatility, window)
                symbol_result[f'volatility_regime_{window}'] = regime
                
                # Calculate volatility z-score
                mean = _calculate_rolling_mean_numba(volatility, window)
                std = _calculate_rolling_std_numba(volatility, window)
                zscore = _calculate_zscore_numba(volatility, mean, std)
                symbol_result[f'volatility_zscore_{window}'] = zscore
            
            # Add to results list
            results.append(symbol_result)
        
        # Combine all symbol results
        result = pd.concat(results).reset_index().set_index(['timestamp', 'symbol'])
        
        return result 