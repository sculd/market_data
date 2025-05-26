"""
Fractional Finite Difference Volatility Zscore Feature Module

This module provides functions for calculating ffd volatility zscore features.
"""

import pandas as pd
import numpy as np
import logging
import numba as nb
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


from market_data.feature.fractional_difference import ZscoredFFDParams as BaseZscoredFFDParams, get_zscored_ffd_series
from market_data.feature.registry import register_feature

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "ffd_volatility_zscore"

@dataclass
class ZscoredFFDVolatilityParams:
    """Parameters for FFD volatility zscore feature calculations."""
    zscored_ffd_params: BaseZscoredFFDParams = field(default_factory=BaseZscoredFFDParams)
    price_col: str = "close"
    volatility_window: int = 20  # Window for volatility calculation
    log_returns: bool = True  # Use log returns vs simple returns
    
    def get_params_dir(self) -> str:
        from market_data.util.cache.path import params_to_dir_name
        
        params_dict = {
            'price_col': self.price_col,
            'volatility_window': self.volatility_window,
            'log_returns': self.log_returns,
            'd': self.zscored_ffd_params.ffd_params.d,
            'threshold': self.zscored_ffd_params.ffd_params.threshold,
            'zscore_window': self.zscored_ffd_params.zscore_window
        }
        return params_to_dir_name(params_dict)
        
    def get_warm_up_period(self) -> datetime.timedelta:
        warm_up = max(
            self.zscored_ffd_params.zscore_window,
            self.volatility_window,
            100  # Conservative estimate for FFD weights convergence
        )
        return datetime.timedelta(minutes=warm_up)

    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on volatility window,
        FFD requirements, and zscore window.
        
        Returns:
            int: Recommended number of warm-up days
        """
        import math
        
        # Add requirements for volatility, FFD and zscore
        total_minutes = max(
            self.zscored_ffd_params.zscore_window,
            self.volatility_window,
            100  # Conservative estimate for FFD weights convergence
        )
        
        # Convert to days (assuming periods are in minutes for 24/7 markets)
        days_needed = math.ceil(total_minutes / (24 * 60))
        
        return max(1, days_needed)  # At least 1 day


@register_feature(FEATURE_LABEL)
class ZscoredFFDVolatilityFeature:
    """FFD volatility zscore feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[ZscoredFFDVolatilityParams] = None) -> pd.DataFrame:
        """
        Calculate FFD volatility zscore features for multiple columns.
        
        Process:
        1. Calculate returns from prices
        2. Calculate rolling volatility (std of returns)
        3. Apply FFD to volatility series
        4. Apply z-score normalization
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for FFD volatility zscore calculation
            
        Returns:
            DataFrame with calculated FFD volatility zscore features
        """
        if params is None:
            params = ZscoredFFDVolatilityParams()
            
        logger.info(f"Calculating FFD volatility zscore for {params}")
        
        # Ensure we have the required column
        if params.price_col not in df.columns:
            raise ValueError(f"Price column '{params.price_col}' not found in DataFrame")
        
        # Check if 'symbol' is in index or columns
        has_symbol = 'symbol' in df.columns
        
        logger.info(f"Original DF index names: {df.index.names}")
        logger.info(f"Original DF columns: {df.columns.tolist()}")
        logger.info(f"Processing price column: {params.price_col}")
        logger.info(f"Volatility window: {params.volatility_window}")
        logger.info(f"Log returns: {params.log_returns}")
        
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
            
            try:
                price_series = group_df[params.price_col]
                
                # Step 1: Calculate returns
                if params.log_returns:
                    returns = np.log(price_series / price_series.shift(1))
                else:
                    returns = price_series.pct_change()
                
                # Step 2: Calculate rolling volatility (standard deviation of returns)
                volatility = returns.rolling(window=params.volatility_window, min_periods=1).std()
                
                # Step 3: Apply FFD to volatility series
                ffd_volatility = get_zscored_ffd_series(
                    volatility, 
                    zscored_params=params.zscored_ffd_params
                )
                
                # Step 4: Align with original index
                ffd_volatility_array = ffd_volatility.reindex(group_df.index).values
                
                # Create output column
                result_col_name = f'ffd_volatility_zscore_{params.price_col}'
                symbol_result[result_col_name] = ffd_volatility_array
                    
            except Exception as e:
                logger.warning(f"Failed to calculate FFD volatility for column {params.price_col}, symbol {symbol}: {e}")
                # Fill with NaN if calculation fails
                result_col_name = f'ffd_volatility_zscore_{params.price_col}'
                symbol_result[result_col_name] = np.nan
            
            # Add to results list
            results.append(symbol_result)
        
        # Combine all symbol results
        result = pd.concat(results)
        
        result = result.reset_index().set_index(['timestamp', 'symbol'])
        
        return result