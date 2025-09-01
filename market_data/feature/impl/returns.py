"""
Returns Feature Module

This module provides functions for calculating return-based features.
"""

import datetime
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numba as nb
import numpy as np
import pandas as pd

from market_data.feature.param import FeatureParam
from market_data.feature.common import Feature
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
class ReturnParams(FeatureParam):
    """Parameters for return feature calculations."""
    periods: List[int] = field(default_factory=lambda: [1, 5, 15, 30, 60, 120])
    price_col: str = "close"
    log_returns: bool = False
    
    def get_warm_up_period(self) -> datetime.timedelta:
        max_period = 1 if not self.periods else max(self.periods)
        return datetime.timedelta(minutes=max_period)

    
    def to_str(self) -> str:
        """Convert parameters to string format: periods:[1,5,15],price_col:close,log_returns:false"""
        periods_str = '[' + ','.join(str(p) for p in self.periods) + ']'
        return f"periods:{periods_str},price_col:{self.price_col},log_returns:{str(self.log_returns).lower()}"
    
    @classmethod
    def from_str(cls, feature_label_str: str) -> 'ReturnParams':
        """Parse Return parameters from JSON-like format: periods:[1,5,15],price_col:close,log_returns:false"""
        params = {}
        for pair in feature_label_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'periods':
                    # Parse [1,5,15] format
                    if value.startswith('[') and value.endswith(']'):
                        periods_str = value[1:-1]
                        params['periods'] = [int(p.strip()) for p in periods_str.split(',') if p.strip()]
                elif key == 'price_col':
                    params['price_col'] = value
                elif key == 'log_returns':
                    params['log_returns'] = value.lower() == 'true'
        return cls(**params)

@register_feature(FEATURE_LABEL)
class ReturnsFeature(Feature):
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