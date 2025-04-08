"""
Volatility Feature Module

This module provides functions for calculating volatility-based features.
"""

import pandas as pd
import numpy as np
import logging
import numba as nb
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from market_data.feature.registry import register_feature
from market_data.feature.impl.returns import _calculate_simple_returns_numba
from market_data.feature.impl.common_calc import _calculate_rolling_std_numba

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "volatility"

@nb.njit(cache=True)
def _annualize_volatility_numba(volatility, trading_periods):
    """
    Annualize volatility values using Numba.
    
    Args:
        volatility: Array of volatility values
        trading_periods: Annualization factor (trading periods per year)
        
    Returns:
        Array of annualized volatility values
    """
    annualization_factor = np.sqrt(trading_periods)
    n = len(volatility)
    result = np.full(n, np.nan)
    
    for i in range(n):
        if not np.isnan(volatility[i]):
            result[i] = volatility[i] * annualization_factor
    
    return result

@dataclass
class VolatilityParams:
    """Parameters for volatility feature calculations."""
    windows: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 60])
    price_col: str = "close"
    annualize: bool = True
    trading_periods_per_year: int = 252 * 24 * 60  # Minutes in a trading year for crypto (24/7)
    
    def get_params_dir(self) -> str:
        """
        Generate a directory name string from parameters.
        
        Returns:
            Directory name string for caching
        """
        from market_data.util.cache.path import params_to_dir_name
        
        windows_str = '_'.join(str(w) for w in self.windows)
        annualize_str = 'ann' if self.annualize else 'raw'
        
        params_dict = {
            'windows': self.windows,
            'price': self.price_col,
            'type': annualize_str
        }
        return params_to_dir_name(params_dict)
    
    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on the maximum volatility window.
        
        Returns:
            int: Recommended number of warm-up days
        """
        import math
        
        if not self.windows:
            return 0
            
        # Find the maximum window
        max_window = max(self.windows)
        
        # Convert to days (assuming windows are in minutes for 24/7 markets)
        # Add 1 day as buffer for safety
        days_needed = math.ceil(max_window / (24 * 60)) + 1
        
        return max(1, days_needed)  # At least 1 day

@register_feature(FEATURE_LABEL)
class VolatilityFeature:
    """Volatility feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[VolatilityParams] = None) -> pd.DataFrame:
        """
        Calculate volatility features for given window sizes.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for volatility calculation
            
        Returns:
            DataFrame with calculated volatility features
        """
        if params is None:
            params = VolatilityParams()
            
        logger.info(f"Calculating volatility for windows {params.windows}")
        
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
            
            # Calculate returns (period-to-period) using function from returns module
            returns = _calculate_simple_returns_numba(prices, 1)
            
            # Calculate volatility for each window using Numba
            for window in params.windows:
                # Calculate rolling standard deviation of returns
                vol = _calculate_rolling_std_numba(returns, window)
                
                # Annualize if requested
                if params.annualize:
                    annualization_factor = np.sqrt(params.trading_periods_per_year / window)
                    vol = _annualize_volatility_numba(vol, params.trading_periods_per_year / window)
                
                symbol_result[f'volatility_{window}'] = vol
            
            # Add to results list
            results.append(symbol_result)
        
        # Combine results from all symbols
        result = pd.concat(results).reset_index().set_index(['timestamp', 'symbol'])
        
        return result 