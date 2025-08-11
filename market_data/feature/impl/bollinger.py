"""
Bollinger Bands Feature Module

This module provides functions for calculating Bollinger Bands features.
"""

import pandas as pd
import numpy as np
import logging
import numba as nb
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from market_data.feature.registry import register_feature
from market_data.feature.param import FeatureParam

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "bollinger"

# Numba-accelerated Bollinger Bands calculation
@nb.njit(cache=True)
def _calculate_bollinger_bands_numba(values, period, std_dev):
    """
    Calculate Bollinger Bands using Numba for performance.
    
    Args:
        values: Array of price values
        period: Period for moving average
        std_dev: Number of standard deviations for upper/lower bands
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    n = len(values)
    upper_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(period, n):
        # Get window of valid values
        window = values[i-period:i]
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
        Array of position values (0-1 range, capped)
    """
    n = len(prices)
    position = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(lower_band[i]) or np.isnan(upper_band[i])):
            band_width = upper_band[i] - lower_band[i]
            if band_width > 0:
                # Calculate raw position
                raw_position = (prices[i] - lower_band[i]) / band_width
                # Cap between 0 and 1
                position[i] = max(0.0, min(1.0, raw_position))
    
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
class BollingerParams(FeatureParam):
    """Parameters for Bollinger Bands calculations."""
    period: int = 20
    std_dev: float = 2.0
    price_col: str = "close"
    
    
    def get_warm_up_period(self) -> datetime.timedelta:
        return datetime.timedelta(minutes=self.period)

    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on the Bollinger Bands period.

        It assumes that the period is in minutes and the data is 24/7.
        
        Returns:
            int: Recommended number of warm-up days
        """
        import math
        
        # Convert period to days (assuming periods are in minutes for 24/7 markets)
        days_needed = math.ceil(self.period / (24 * 60))
        
        return max(1, days_needed)  # At least 1 day
    
    def to_str(self) -> str:
        """Convert parameters to string format: period:20,std_dev:2.0,price_col:close"""
        return f"period:{self.period},std_dev:{self.std_dev},price_col:{self.price_col}"
    
    @classmethod
    def from_str(cls, feature_label_str: str) -> 'BollingerParams':
        """Parse Bollinger parameters from JSON-like format: period:20,std_dev:2.0,price_col:close"""
        params = {}
        for pair in feature_label_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'period':
                    params['period'] = int(value)
                elif key == 'std_dev':
                    params['std_dev'] = float(value)
                elif key == 'price_col':
                    params['price_col'] = value
        return cls(**params)

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
            
            # Calculate cumulative percentage changes instead of using raw prices
            # First get the percentage changes and fill NaN values (first row)
            pct_changes = group_df[params.price_col].pct_change().fillna(0)
            
            # Then calculate the cumulative sum
            cum_pct_changes = pct_changes.cumsum().values
            
            # Calculate Bollinger Bands using Numba on the cumulative percentage changes
            upper, middle, lower = _calculate_bollinger_bands_numba(
                cum_pct_changes, params.period, params.std_dev
            )
            
            # Add bands to result
            symbol_result['bb_upper'] = upper
            symbol_result['bb_middle'] = middle
            symbol_result['bb_lower'] = lower
            
            # Calculate position within bands (capped between 0-1)
            symbol_result['bb_position'] = _calculate_bb_position_numba(cum_pct_changes, lower, upper)
                
            # Calculate bandwidth (normalized)
            symbol_result['bb_width'] = _calculate_bb_width_numba(upper, middle, lower)
            
            # Add to results list
            results.append(symbol_result)
        
        # Combine all symbol results
        result = pd.concat(results).reset_index().set_index(['timestamp', 'symbol'])
        
        return result 