"""
Market Regime Feature Module

This module provides functions for calculating market regime indicators
based on volatility patterns.
"""

import pandas as pd
import numpy as np
import logging
import numba as nb
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from market_data.feature.registry import register_feature
from market_data.feature.impl.returns import _calculate_simple_returns_numba
from market_data.feature.impl.common_calc import _calculate_rolling_std_numba, _calculate_rolling_mean_numba

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

@dataclass
class MarketRegimeParams:
    """Parameters for market regime calculations."""
    volatility_windows: List[int] = field(default_factory=lambda: [240, 1440, 4320])  # 4h, 1d, 3d
    price_col: str = "close"
    garch_omega: float = 0.1
    garch_alpha: float = 0.1
    garch_beta: float = 0.8
    
    def get_params_dir(self) -> str:
        """
        Generate a directory name string from parameters.
        
        Returns:
            Directory name string for caching
        """
        from market_data.util.cache.path import params_to_dir_name
        
        windows_str = '_'.join(str(w) for w in self.volatility_windows)
        
        params_dict = {
            'windows': self.volatility_windows,
            'price': self.price_col,
            'garch_params': [self.garch_omega, self.garch_alpha, self.garch_beta]
        }
        return params_to_dir_name(params_dict)
        
    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on the maximum volatility window
        and GARCH parameters.
        
        Returns:
            int: Recommended number of warm-up days
        """
        import math
        
        if not self.volatility_windows:
            # Default GARCH warm-up if no windows specified
            return 3
            
        # Find the maximum window
        max_window = max(self.volatility_windows)
        
        # Convert to days (assuming windows are in minutes for 24/7 markets)
        # Add 3 days for GARCH convergence as additional safety
        days_needed = math.ceil(max_window / (24 * 60)) + 3
        
        return max(3, days_needed)  # At least 3 days for GARCH

@register_feature(FEATURE_LABEL)
class MarketRegimeFeature:
    """Market regime feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[MarketRegimeParams] = None) -> pd.DataFrame:
        """
        Calculate market regime features based on volatility.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for market regime calculation
            
        Returns:
            DataFrame with calculated market regime features
        """
        if params is None:
            params = MarketRegimeParams()
            
        logger.info(f"Calculating market regime features for windows {params.volatility_windows}")
        
        # Ensure we have the price column
        if params.price_col not in df.columns:
            raise ValueError(f"Price column '{params.price_col}' not found in DataFrame")
        
        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Extract prices as numpy array for Numba functions
        prices = df[params.price_col].values
        
        # Calculate returns using _calculate_simple_returns_numba from returns module
        returns = _calculate_simple_returns_numba(prices, 1)
        
        # Initialize returns (replace NaN values at beginning)
        returns_init = _initialize_returns_numba(returns)
        
        # Calculate GARCH volatility
        volatility = _calculate_garch_volatility_numba(
            returns_init, 
            params.garch_omega, 
            params.garch_alpha, 
            params.garch_beta
        )
        
        result['garch_volatility'] = volatility
        
        # Calculate rolling regime and zscore for different time horizons
        for window in params.volatility_windows:
            # Calculate volatility regime
            regime = _calculate_volatility_regime_numba(volatility, window)
            result[f'volatility_regime_{window}'] = regime
            
            # Calculate volatility z-score
            mean = _calculate_rolling_mean_numba(volatility, window)
            std = _calculate_rolling_std_numba(volatility, window)
            zscore = _calculate_zscore_numba(volatility, mean, std)
            result[f'volatility_zscore_{window}'] = zscore
        
        return result 