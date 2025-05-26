"""
Fractional Finite Difference Zscore Feature Module

This module provides functions for calculating ffd zscore features.
"""

import pandas as pd
import numpy as np
import logging
import numba as nb
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


from market_data.feature.fractional_difference import ZscoredFFDParams as BaseZscoredFFDParams, get_zscored_ffd_series
from market_data.feature.impl.returns import _calculate_log_returns_numba, _calculate_simple_returns_numba
from market_data.feature.registry import register_feature

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "ffd_zscore"

@dataclass
class ZscoredFFDParams:
    """Parameters for FFD zscore feature calculations."""
    zscored_ffd_params: BaseZscoredFFDParams = field(default_factory=BaseZscoredFFDParams)
    cols: List[str] = field(default_factory=lambda: ["close", "volume"])
    
    def get_params_dir(self) -> str:
        from market_data.util.cache.path import params_to_dir_name
        
        params_dict = {
            'cols': self.cols,
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
        Calculate the recommended warm-up period based on the maximum period,
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
class ZscoredFFDsFeature:
    """FFD zscore feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[ZscoredFFDParams] = None) -> pd.DataFrame:
        """
        Calculate FFD zscore features for multiple columns.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for FFD zscore calculation
            
        Returns:
            DataFrame with calculated FFD zscore features
        """
        if params is None:
            params = ZscoredFFDParams()
            
        logger.info(f"Calculating FFD zscore for {params}")
        
        # Ensure we have all the required columns
        missing_cols = [col for col in params.cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame")
        
        # Check if 'symbol' is in index or columns
        has_symbol = 'symbol' in df.columns
        
        logger.info(f"Original DF index names: {df.index.names}")
        logger.info(f"Original DF columns: {df.columns.tolist()}")
        logger.info(f"Processing columns: {params.cols}")
        
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
            
            # Process each column
            for col in params.cols:
                try:
                    price_series = group_df[col]
                    
                    ffd_prices = get_zscored_ffd_series(
                        price_series, 
                        zscored_params=params.zscored_ffd_params
                    )
                    
                    ffd_prices_array = ffd_prices.reindex(group_df.index).values
                    
                    symbol_result[f'ffd_zscore_{col}'] = ffd_prices_array
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate FFD for column {col}, symbol {symbol}: {e}")
                    # Fill with NaN if FFD calculation fails
                    symbol_result[f'ffd_zscore_{col}'] = np.nan
            
            # Add to results list
            results.append(symbol_result)
        
        # Combine all symbol results
        result = pd.concat(results)
        
        result = result.reset_index().set_index(['timestamp', 'symbol'])
        
        return result