"""
Returns Feature Module

This module provides functions for calculating return-based features.
"""

import pandas as pd
import numpy as np
import logging
import numba as nb
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from market_data.feature.registry import register_feature

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "returns"

# Numba-accelerated return calculations
@nb.njit(cache=True)
def _calculate_simple_returns_numba(prices, period):
    """
    Calculate simple returns using Numba for performance.
    
    Args:
        prices: Array of price values
        period: Period for return calculation
        
    Returns:
        Array of simple returns
    """
    n = len(prices)
    returns = np.full(n, np.nan)
    
    for i in range(period, n):
        if prices[i-period] > 0:  # Avoid division by zero
            returns[i] = (prices[i] / prices[i-period]) - 1.0
    
    return returns

@nb.njit(cache=True)
def _calculate_log_returns_numba(prices, period):
    """
    Calculate logarithmic returns using Numba for performance.
    
    Args:
        prices: Array of price values
        period: Period for return calculation
        
    Returns:
        Array of logarithmic returns
    """
    n = len(prices)
    returns = np.full(n, np.nan)
    
    for i in range(period, n):
        if prices[i-period] > 0:  # Avoid division by zero
            returns[i] = np.log(prices[i] / prices[i-period])
    
    return returns

@dataclass
class ReturnParams:
    """Parameters for return feature calculations."""
    periods: List[int] = field(default_factory=lambda: [1, 5, 15, 30, 60, 120])
    price_col: str = "close"
    log_returns: bool = False
    
    def get_params_dir(self) -> str:
        """
        Generate a directory name string from parameters.
        
        Returns:
            Directory name string for caching
        """
        from market_data.util.cache.path import params_to_dir_name
        
        periods_str = '_'.join(str(p) for p in self.periods)
        log_str = 'log' if self.log_returns else 'simple'
        
        params_dict = {
            'periods': self.periods,
            'price': self.price_col,
            'type': log_str
        }
        return params_to_dir_name(params_dict)
        
    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on the maximum return period.
        
        Returns:
            int: Recommended number of warm-up days
        """
        import math
        
        if not self.periods:
            return 0
            
        # Find the maximum period
        max_period = max(self.periods)
        
        # Convert to days (assuming periods are in minutes for 24/7 markets)
        # Add 1 day as buffer for safety
        days_needed = math.ceil(max_period / (24 * 60)) + 1
        
        return max(1, days_needed)  # At least 1 day

@register_feature(FEATURE_LABEL)
class ReturnsFeature:
    """Returns feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[ReturnParams] = None) -> pd.DataFrame:
        """
        Calculate return features for given periods.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for return calculation
            
        Returns:
            DataFrame with calculated return features
        """
        if params is None:
            params = ReturnParams()
            
        logger.info(f"Calculating returns for periods {params.periods}")
        
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
            
            # Calculate returns for each period using Numba
            for period in params.periods:
                if params.log_returns:
                    # Log returns with Numba
                    returns = _calculate_log_returns_numba(prices, period)
                else:
                    # Simple returns with Numba
                    returns = _calculate_simple_returns_numba(prices, period)
                    
                # Add to result DataFrame
                symbol_result[f'return_{period}'] = returns
            
            # Add to results list
            results.append(symbol_result)
        
        # Combine all symbol results
        result = pd.concat(results).reset_index().set_index(['timestamp', 'symbol'])
        
        return result 