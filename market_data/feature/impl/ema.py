"""
EMA Feature Module

This module provides functions for calculating Exponential Moving Averages (EMA)
and price relatives to EMA.
"""

import pandas as pd
import numpy as np
import logging
import datetime
import numba as nb
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from market_data.feature.registry import register_feature
from market_data.feature.label import FeatureParam

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "ema"

# Numba-accelerated EMA calculations
@nb.njit(cache=True)
def _calculate_ema_numba(prices, span):
    """
    Calculate EMA using Numba for performance.
    
    Args:
        prices: Array of price values
        span: Span parameter for EMA (similar to period)
        
    Returns:
        Array of EMA values
    """
    n = len(prices)
    ema = np.full(n, np.nan)
    
    # Find first valid price
    start_idx = 0
    while start_idx < n and np.isnan(prices[start_idx]):
        start_idx += 1
    
    if start_idx >= n:
        return ema  # All prices are NaN
    
    # Initialize EMA with first valid price
    ema[start_idx] = prices[start_idx]
    
    # Calculate alpha (smoothing factor)
    alpha = 2.0 / (span + 1.0)
    
    # Calculate EMA
    for i in range(start_idx + 1, n):
        if not np.isnan(prices[i]):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        else:
            ema[i] = ema[i-1]  # Maintain last value for NaN prices
    
    return ema

@nb.njit(cache=True)
def _calculate_price_to_ema_ratio_numba(prices, ema):
    """
    Calculate price to EMA ratio using Numba.
    
    Args:
        prices: Array of price values
        ema: Array of EMA values
        
    Returns:
        Array of price/EMA ratios
    """
    n = len(prices)
    ratio = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(prices[i]) or np.isnan(ema[i])) and ema[i] > 0:
            ratio[i] = prices[i] / ema[i]
    
    return ratio

@dataclass
class EMAParams(FeatureParam):
    """Parameters for EMA feature calculations."""
    periods: List[int] = field(default_factory=lambda: [5, 15, 30, 60, 120])
    price_col: str = "close"
    include_price_relatives: bool = True
    
    def get_params_dir(self) -> str:
        """
        Generate a directory name string from parameters.
        
        Returns:
            Directory name string for caching
        """
        from market_data.util.cache.path import params_to_dir_name
        
        periods_str = '_'.join(str(p) for p in self.periods)
        
        params_dict = {
            'periods': self.periods,
            'price': self.price_col,
            'rel': self.include_price_relatives
        }
        return params_to_dir_name(params_dict)
        
    def get_warm_up_period(self) -> datetime.timedelta:
        return datetime.timedelta(minutes=max(self.periods))

    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on the maximum EMA period.
        
        For EMA calculation, warm-up should be at least 2-3 times the period
        to allow for proper convergence.
        
        Returns:
            int: Recommended number of warm-up days
        """
        import math
        
        if not self.periods:
            return 0
            
        # Find the maximum period
        max_period = max(self.periods)
        
        # For proper EMA convergence, we typically need 2-3 times the period length
        # Convert to days (assuming periods are in minutes for 24/7 markets)
        days_needed = math.ceil(3 * max_period / (24 * 60))
        
        return max(1, days_needed)  # At least 1 day
    
    def to_str(self) -> str:
        """Convert parameters to string format: periods:[5,15,30],price_col:close,include_price_relatives:true"""
        periods_str = '[' + ','.join(str(p) for p in self.periods) + ']'
        return f"periods:{periods_str},price_col:{self.price_col},include_price_relatives:{str(self.include_price_relatives).lower()}"
    
    @classmethod
    def from_str(cls, feature_label_str: str) -> 'EMAParams':
        """Parse EMA parameters from JSON-like format: periods:[5,15,30],price_col:close,include_price_relatives:true"""
        params = {}
        for pair in feature_label_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'periods':
                    # Parse [5,15,30] format
                    if value.startswith('[') and value.endswith(']'):
                        periods_str = value[1:-1]
                        params['periods'] = [int(p.strip()) for p in periods_str.split(',') if p.strip()]
                elif key == 'price_col':
                    params['price_col'] = value
                elif key == 'include_price_relatives':
                    params['include_price_relatives'] = value.lower() == 'true'
        return cls(**params)

@register_feature(FEATURE_LABEL)
class EMAFeature:
    """EMA feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[EMAParams] = None) -> pd.DataFrame:
        """
        Calculate EMA features for given periods.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for EMA calculation
            
        Returns:
            DataFrame with calculated EMA features
        """
        if params is None:
            params = EMAParams()
            
        logger.info(f"Calculating EMAs for periods {params.periods}")
        
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
            
            # Calculate EMA for each period using Numba
            for period in params.periods:
                # Calculate EMA
                ema = _calculate_ema_numba(prices, period)
                symbol_result[f'ema_{period}'] = ema

                # Calculate price relative to EMA if requested
                if params.include_price_relatives:
                    price_to_ema = _calculate_price_to_ema_ratio_numba(prices, ema)
                    symbol_result[f'ema_rel_{period}'] = price_to_ema
            
            # Add to results list
            results.append(symbol_result)
        
        # Combine all symbol results
        result = pd.concat(results).reset_index().set_index(['timestamp', 'symbol'])
        
        return result 