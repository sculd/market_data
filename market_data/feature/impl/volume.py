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
from market_data.feature.param import FeatureParam
from market_data.feature.fractional_difference import ZscoredFFDParams as BaseZscoredFFDParams, get_zscored_ffd_series
from market_data.feature.impl.common_calc import _calculate_rolling_mean_numba

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "volume"
_volume_col = "volume"

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
    n = len(values)
    pct_change = np.full(n, np.nan)
    
    for i in range(1, n):
        if not (np.isnan(values[i]) or np.isnan(values[i-1])) and values[i-1] != 0:
            pct_change[i] = (values[i] - values[i-1]) / abs(values[i-1])
    
    return pct_change

@nb.njit(cache=True)
def _calculate_log_numba(values):
    """
    Apply log transformation to reduce explosiveness, using log(1 + abs(x)) * sign(x).
    This preserves the sign while reducing the magnitude of extreme values.
    """
    n = len(values)
    log_values = np.full(n, np.nan)
    
    for i in range(n):
        if not np.isnan(values[i]):
            if values[i] == 0:
                log_values[i] = 0.0
            else:
                # Use log(1 + abs(x)) * sign(x) to preserve sign and reduce explosiveness
                sign = 1.0 if values[i] > 0 else -1.0
                log_values[i] = np.log1p(abs(values[i])) * sign
    
    return log_values


@dataclass
class VolumeParams(FeatureParam):
    """Parameters for volume indicator calculations."""
    ratio_period: int = 100
    price_col: str = "close"
    zscored_ffd_params: BaseZscoredFFDParams = field(default_factory=BaseZscoredFFDParams)
    
    def get_params_dir(self) -> str:
        """
        Generate a directory name string from parameters.
        
        Returns:
            Directory name string for caching
        """
        from market_data.util.cache.path import params_to_dir_name
        
        params_dict = {
            'ratio_period': self.ratio_period,
            'price': self.price_col,
            'd': self.zscored_ffd_params.ffd_params.d,
            'threshold': self.zscored_ffd_params.ffd_params.threshold,
            'zscore_window': self.zscored_ffd_params.zscore_window
        }
        return params_to_dir_name(params_dict)
        
    def _get_warm_up_len(self) -> int:
        max_window = self.zscored_ffd_params.zscore_window + 100  # 100 is the warm-up period for ffd
        return max(max_window, self.ratio_period)
        
    def get_warm_up_period(self) -> datetime.timedelta:
        return datetime.timedelta(minutes=self._get_warm_up_len())

    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on the maximum volume ratio period
        and OBV zscore window.
        
        Returns:
            int: Recommended number of warm-up days
        """
        import math
        
        max_window = self._get_warm_up_len()
        
        # Convert to days (assuming periods are in minutes for 24/7 markets)
        days_needed = math.ceil(max_window / (24 * 60))
        
        return max(1, days_needed)  # At least 1 day
    
    def to_str(self) -> str:
        """Convert parameters to string format: ratio_period:100,price_col:close,d:0.5,threshold:0.01,zscore_window:100"""
        return f"ratio_period:{self.ratio_period},price_col:{self.price_col},d:{self.zscored_ffd_params.ffd_params.d},threshold:{self.zscored_ffd_params.ffd_params.threshold},zscore_window:{self.zscored_ffd_params.zscore_window}"
    
    @classmethod
    def from_str(cls, feature_label_str: str) -> 'VolumeParams':
        """Parse Volume parameters from JSON-like format: ratio_period:100,price_col:close,d:0.5,threshold:0.01,zscore_window:100"""
        params = {}
        base_params = BaseZscoredFFDParams()
        
        for pair in feature_label_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'ratio_period':
                    params['ratio_period'] = int(value)
                elif key == 'price_col':
                    params['price_col'] = value
                elif key == 'd':
                    base_params.ffd_params.d = float(value)
                elif key == 'threshold':
                    base_params.ffd_params.threshold = float(value)
                elif key == 'zscore_window':
                    base_params.zscore_window = int(value)
        
        params['zscored_ffd_params'] = base_params
        return cls(**params)

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
        if _volume_col not in df.columns:
            raise ValueError(f"Volume column '{_volume_col}' not found in DataFrame")
        
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
            volumes = group_df[_volume_col].values
            
            # Calculate On-Balance Volume using Numba (used for derived metrics, not stored directly)
            obv = _calculate_obv_numba(prices, volumes)
            
            # Convert obv numpy array to pandas Series for zscored_ffd_series function
            obv_series = pd.Series(obv, index=group_df.index)
            
            ffd_obv = get_zscored_ffd_series(
                obv_series, 
                zscored_params=params.zscored_ffd_params
            )
            symbol_result[f'obv_ffd_zscore'] = ffd_obv

            # Calculate OBV percentage change
            obv_pct_change = _calculate_pct_change_numba(obv)
            # Apply log transformation to reduce explosiveness for ML models
            obv_pct_change_log = _calculate_log_numba(obv_pct_change)
            symbol_result['obv_pct_change_log'] = obv_pct_change_log
            
            # Calculate rolling mean of volume
            avg_volume = _calculate_rolling_mean_numba(volumes, params.ratio_period)
            
            # Calculate volume to average ratio
            volume_ratio = _calculate_ratio_numba(volumes, avg_volume)
            volume_ratio_log = _calculate_log_numba(volume_ratio)
            symbol_result[f'volume_ratio_log'] = volume_ratio_log
        
            # Add to results list
            results.append(symbol_result)
        
        # Combine all symbol results
        result = pd.concat(results).reset_index().set_index(['timestamp', 'symbol'])
        
        return result 