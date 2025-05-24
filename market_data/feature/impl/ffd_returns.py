"""
Returns Feature Module

This module provides functions for calculating return-based features.
"""

import pandas as pd
import numpy as np
import logging
import numba as nb
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


import market_data.feature.fractional_difference
from market_data.feature.fractional_difference import ZscoredFFDParams as BaseZscoredFFDParams, get_zscored_ffd_series
from market_data.feature.impl.returns import _calculate_log_returns_numba, _calculate_simple_returns_numba
from market_data.feature.registry import register_feature

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "ffd_returns"

@dataclass
class ZscoredFFDReturnParams:
    """Parameters for FFD returns feature calculations."""
    zscored_ffd_params: BaseZscoredFFDParams = field(default_factory=BaseZscoredFFDParams)
    price_col: str = "close"
    
    def get_params_dir(self) -> str:
        from market_data.util.cache.path import params_to_dir_name
        
        params_dict = {
            'price': self.price_col,
            'd': self.zscored_ffd_params.ffd_params.d,
            'threshold': self.zscored_ffd_params.ffd_params.threshold,
            'zscore_window': self.zscored_ffd_params.zscore_window
        }
        return params_to_dir_name(params_dict)
        
    def get_warm_up_period(self) -> datetime.timedelta:
        warm_up = max(
            self.zscored_ffd_params.zscore_window,
            100  # Conservative estimate for FFD weights convergence
        )
        return datetime.timedelta(minutes=warm_up)

    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on the maximum return period,
        FFD requirements, and zscore window.
        
        Returns:
            int: Recommended number of warm-up days
        """
        import math
        
        # Add requirements for FFD and zscore
        total_minutes = max(
            self.zscored_ffd_params.zscore_window,
            100  # Conservative estimate for FFD weights convergence
        )
        
        # Convert to days (assuming periods are in minutes for 24/7 markets)
        days_needed = math.ceil(total_minutes / (24 * 60))
        
        return max(1, days_needed)  # At least 1 day


@register_feature(FEATURE_LABEL)
class ZscoredFFDReturnsFeature:
    """FFD Returns feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[ZscoredFFDReturnParams] = None) -> pd.DataFrame:
        """
        Calculate FFD returns features.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for FFD returns calculation
            
        Returns:
            DataFrame with calculated FFD returns features
        """
        if params is None:
            params = ZscoredFFDReturnParams()
            
        logger.info(f"Calculating FFD returns for {params}")
        
        # Ensure we have the price column
        if params.price_col not in df.columns:
            raise ValueError(f"Price column '{params.price_col}' not found in DataFrame")
        
        # Check if 'symbol' is in index or columns
        has_symbol = 'symbol' in df.columns
        
        logger.info(f"Original DF index names: {df.index.names}")
        logger.info(f"Original DF columns: {df.columns.tolist()}")
        
        if not has_symbol:
            logger.warning("DataFrame does not have a 'symbol' column or index level, will use a single default symbol")
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
            
            # Extract prices as numpy array for return calculation
            price_series = group_df[params.price_col]
            
            try:
                ffd_prices = get_zscored_ffd_series(
                    price_series, 
                    zscored_params=params.zscored_ffd_params
                )
                
                # Convert FFD prices back to numpy array for return calculations
                ffd_prices_array = ffd_prices.reindex(group_df.index).values
                
                symbol_result[f'ffd_return'] = ffd_prices_array
                    
            except Exception as e:
                logger.warning(f"Failed to calculate FFD for symbol {symbol}: {e}")
                # Fill with NaN if FFD calculation fails
                for period in params.periods:
                    symbol_result[f'ffd_return_{period}'] = np.nan
            
            # Add to results list
            results.append(symbol_result)
        
        # Combine all symbol results
        result = pd.concat(results)
        
        result = result.reset_index().set_index(['timestamp', 'symbol'])
        
        return result