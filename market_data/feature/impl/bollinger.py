"""
Bollinger Bands Feature Module

This module provides functions for calculating Bollinger Bands features.
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
FEATURE_LABEL = "bollinger"

# Numba-accelerated Bollinger Bands calculation
@nb.njit(cache=True)
def _calculate_bollinger_bands_numba(prices, period, std_dev):
    """
    Calculate Bollinger Bands using Numba for performance.
    
    Args:
        prices: Array of price values
        period: Period for moving average
        std_dev: Number of standard deviations for upper/lower bands
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    n = len(prices)
    upper_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(period, n):
        # Get window of valid prices
        window = prices[i-period:i]
        valid_window = window[~np.isnan(window)]
        
        if len(valid_window) >= period // 2:  # Require at least half of the window to be valid
            mean = np.mean(valid_window)
            std = np.std(valid_window)
            
            middle_band[i] = mean
            upper_band[i] = mean + (std_dev * std)
            lower_band[i] = mean - (std_dev * std)
    
    return upper_band, middle_band, lower_band

@nb.njit(cache=True)
def _calculate_bb_position_numba(prices, lower_band, upper_band):
    """
    Calculate position within Bollinger Bands using Numba.
    
    Args:
        prices: Array of price values
        lower_band: Lower Bollinger Band values
        upper_band: Upper Bollinger Band values
        
    Returns:
        Array of position values (0-1 range)
    """
    n = len(prices)
    position = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(lower_band[i]) or np.isnan(upper_band[i])):
            band_width = upper_band[i] - lower_band[i]
            if band_width > 0:
                position[i] = (prices[i] - lower_band[i]) / band_width
    
    return position

@nb.njit(cache=True)
def _calculate_bb_width_numba(upper_band, middle_band, lower_band):
    """
    Calculate Bollinger Bands width using Numba.
    
    Args:
        upper_band: Upper Bollinger Band values
        middle_band: Middle Bollinger Band values 
        lower_band: Lower Bollinger Band values
        
    Returns:
        Array of width values (normalized)
    """
    n = len(upper_band)
    width = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(upper_band[i]) or np.isnan(middle_band[i]) or np.isnan(lower_band[i])):
            if middle_band[i] > 0:
                width[i] = (upper_band[i] - lower_band[i]) / middle_band[i]
    
    return width

@dataclass
class BollingerParams:
    """Parameters for Bollinger Bands calculations."""
    period: int = 20
    std_dev: float = 2.0
    price_col: str = "close"
    
    def get_params_dir(self) -> str:
        """
        Generate a directory name string from parameters.
        
        Returns:
            Directory name string for caching
        """
        from market_data.util.cache.path import params_to_dir_name
        
        params_dict = {
            'period': self.period,
            'std_dev': self.std_dev,
            'price': self.price_col
        }
        return params_to_dir_name(params_dict)
        
    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on the Bollinger Bands period.
        
        Returns:
            int: Recommended number of warm-up days
        """
        import math
        
        # Convert period to days (assuming periods are in minutes for 24/7 markets)
        # Add 1 day as buffer for safety
        days_needed = math.ceil(self.period / (24 * 60)) + 1
        
        return max(1, days_needed)  # At least 1 day

@register_feature(FEATURE_LABEL)
class BollingerFeature:
    """Bollinger Bands feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[BollingerParams] = None) -> pd.DataFrame:
        """
        Calculate Bollinger Bands features.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for Bollinger Bands calculation
            
        Returns:
            DataFrame with calculated Bollinger Bands features
        """
        if params is None:
            params = BollingerParams()
            
        logger.info(f"Calculating Bollinger Bands with period={params.period}, std_dev={params.std_dev}")
        
        # Ensure we have the price column
        if params.price_col not in df.columns:
            raise ValueError(f"Price column '{params.price_col}' not found in DataFrame")
        
        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Extract prices as numpy array for Numba functions
        prices = df[params.price_col].values
        
        # Calculate Bollinger Bands using Numba
        upper, middle, lower = _calculate_bollinger_bands_numba(
            prices, params.period, params.std_dev
        )
        
        # Add bands to result
        result['bb_upper'] = upper
        result['bb_middle'] = middle
        result['bb_lower'] = lower
        
        # Calculate position within bands (normalized to 0-1 range)
        result['bb_position'] = _calculate_bb_position_numba(prices, lower, upper)
            
        # Calculate bandwidth (normalized)
        result['bb_width'] = _calculate_bb_width_numba(upper, middle, lower)
        
        return result 