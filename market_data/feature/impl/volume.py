"""
Volume Feature Module

This module provides functions for calculating volume-based indicators like
On-Balance Volume (OBV) and volume ratios.
"""

import pandas as pd
import numpy as np
import logging
import numba as nb
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from market_data.feature.registry import register_feature
from market_data.feature.impl.common_calc import _calculate_rolling_std_numba, _calculate_rolling_mean_numba
from market_data.feature.impl.indicators import _calculate_zscore_numba

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "volume"

# Numba-accelerated volume calculations
@nb.njit(cache=True)
def _calculate_obv_numba(prices, volumes):
    """
    Calculate On-Balance Volume (OBV) using Numba for performance.
    
    Args:
        prices: Array of price values
        volumes: Array of volume values
        
    Returns:
        Array of OBV values
    """
    n = len(prices)
    obv = np.full(n, np.nan)
    
    # Find first valid index
    start_idx = 0
    while start_idx < n and (np.isnan(prices[start_idx]) or np.isnan(volumes[start_idx])):
        start_idx += 1
    
    if start_idx >= n:
        return obv  # All values are NaN
    
    # Initialize OBV with first valid volume
    obv[start_idx] = volumes[start_idx]
    
    # Calculate OBV
    for i in range(start_idx + 1, n):
        if np.isnan(prices[i]) or np.isnan(volumes[i]) or np.isnan(prices[i-1]):
            if i > 0 and not np.isnan(obv[i-1]):
                obv[i] = obv[i-1]  # Keep last value
        else:
            if prices[i] > prices[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif prices[i] < prices[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]  # Price unchanged
    
    return obv

@nb.njit(cache=True)
def _calculate_ratio_numba(values, mean):
    """
    Calculate ratio of values to mean using Numba.
    
    Args:
        values: Array of input values
        mean: Array of mean values
        
    Returns:
        Array of ratio values
    """
    n = len(values)
    ratio = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(values[i]) or np.isnan(mean[i])) and mean[i] > 0:
            ratio[i] = values[i] / mean[i]
    
    return ratio

@nb.njit(cache=True)
def _calculate_pct_change_numba(values):
    """
    Calculate percentage change using Numba.
    
    Args:
        values: Array of input values
        
    Returns:
        Array of percentage change values
    """
    n = len(values)
    pct_change = np.full(n, np.nan)
    
    for i in range(1, n):
        if not (np.isnan(values[i]) or np.isnan(values[i-1])) and values[i-1] != 0:
            pct_change[i] = (values[i] - values[i-1]) / abs(values[i-1])
    
    return pct_change

@dataclass
class VolumeParams:
    """Parameters for volume indicator calculations."""
    ratio_periods: List[int] = field(default_factory=lambda: [20, 50, 100])
    obv_zscore_window: int = 20
    price_col: str = "close"
    volume_col: str = "volume"
    
    def get_params_dir(self) -> str:
        """
        Generate a directory name string from parameters.
        
        Returns:
            Directory name string for caching
        """
        from market_data.util.cache.path import params_to_dir_name
        
        periods_str = '_'.join(str(p) for p in self.ratio_periods)
        
        params_dict = {
            'ratio_periods': self.ratio_periods,
            'obv_zscore': self.obv_zscore_window,
            'price': self.price_col,
            'volume': self.volume_col
        }
        return params_to_dir_name(params_dict)
        
    def get_warm_up_period(self) -> datetime.timedelta:
        max_window = self.obv_zscore_window
        
        if self.ratio_periods:
            max_period = max(self.ratio_periods)
            max_window = max(max_window, max_period)
        
        return datetime.timedelta(minutes=max_window)

    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on the maximum volume ratio period
        and OBV zscore window.
        
        Returns:
            int: Recommended number of warm-up days
        """
        import math
        
        # Find the maximum window from all parameters
        max_window = self.obv_zscore_window
        
        if self.ratio_periods:
            max_period = max(self.ratio_periods)
            max_window = max(max_window, max_period)
        
        # Convert to days (assuming periods are in minutes for 24/7 markets)
        days_needed = math.ceil(max_window / (24 * 60))
        
        return max(1, days_needed)  # At least 1 day

@register_feature(FEATURE_LABEL)
class VolumeFeature:
    """Volume indicators feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[VolumeParams] = None) -> pd.DataFrame:
        """
        Calculate volume-based indicators.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for volume indicator calculation
            
        Returns:
            DataFrame with calculated volume indicators
        """
        if params is None:
            params = VolumeParams()
            
        logger.info(f"Calculating volume indicators")
        
        # Ensure we have the required columns
        if params.price_col not in df.columns:
            raise ValueError(f"Price column '{params.price_col}' not found in DataFrame")
        if params.volume_col not in df.columns:
            raise ValueError(f"Volume column '{params.volume_col}' not found in DataFrame")
        
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
            prices = group_df[params.price_col].values
            volumes = group_df[params.volume_col].values
            
            # Calculate On-Balance Volume using Numba (used for derived metrics, not stored directly)
            obv = _calculate_obv_numba(prices, volumes)
            
            # Calculate OBV percentage change
            obv_pct_change = _calculate_pct_change_numba(obv)
            symbol_result['obv_pct_change'] = obv_pct_change
            
            # Calculate OBV Z-score
            obv_mean = _calculate_rolling_mean_numba(obv, params.obv_zscore_window)
            obv_std = _calculate_rolling_std_numba(obv, params.obv_zscore_window)
            obv_zscore = _calculate_zscore_numba(obv, obv_mean, obv_std)
            symbol_result['obv_zscore'] = obv_zscore
            
            # Calculate volume ratios for each period
            for period in params.ratio_periods:
                # Calculate rolling mean of volume
                avg_volume = _calculate_rolling_mean_numba(volumes, period)
                
                # Calculate volume to average ratio
                volume_ratio = _calculate_ratio_numba(volumes, avg_volume)
                symbol_result[f'volume_ratio_{period}'] = volume_ratio
            
            # Add to results list
            results.append(symbol_result)
        
        # Combine all symbol results
        result = pd.concat(results).reset_index().set_index(['timestamp', 'symbol'])
        
        return result 