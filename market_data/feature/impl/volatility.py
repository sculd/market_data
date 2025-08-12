"""
Volatility Feature Module

This module provides functions for calculating volatility-based features.
"""

import datetime
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numba as nb
import numpy as np
import pandas as pd

from market_data.feature.impl.common_calc import _calculate_rolling_std_numba
from market_data.feature.impl.returns import _calculate_simple_returns_numba
from market_data.feature.param import FeatureParam
from market_data.feature.registry import register_feature

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
class VolatilityParams(FeatureParam):
    """Parameters for volatility feature calculations."""
    windows: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 60])
    price_col: str = "close"
    annualize: bool = True
    trading_periods_per_year: int = 252 * 24 * 60  # Minutes in a trading year for crypto (24/7)
    
    def get_warm_up_period(self) -> datetime.timedelta:
        max_period = 1 if not self.windows else max(self.windows)
        return datetime.timedelta(minutes=max_period)

    
    def to_str(self) -> str:
        """Convert parameters to string format: windows:[5,10,20],price_col:close,annualize:true"""
        windows_str = '[' + ','.join(str(w) for w in self.windows) + ']'
        return f"windows:{windows_str},price_col:{self.price_col},annualize:{str(self.annualize).lower()},trading_periods_per_year:{self.trading_periods_per_year}"
    
    @classmethod
    def from_str(cls, feature_label_str: str) -> 'VolatilityParams':
        """Parse Volatility parameters from JSON-like format: windows:[5,10,20],price_col:close,annualize:true"""
        params = {}
        for pair in feature_label_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'windows':
                    # Parse [5,10,20] format
                    if value.startswith('[') and value.endswith(']'):
                        windows_str = value[1:-1]
                        params['windows'] = [int(w.strip()) for w in windows_str.split(',') if w.strip()]
                elif key == 'price_col':
                    params['price_col'] = value
                elif key == 'annualize':
                    params['annualize'] = value.lower() == 'true'
                elif key == 'trading_periods_per_year':
                    params['trading_periods_per_year'] = int(value)
        return cls(**params)

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