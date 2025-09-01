"""
Fractional Finite Difference Zscore Feature Module

This module provides functions for calculating ffd zscore features.
"""

import datetime
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numba as nb
import numpy as np
import pandas as pd

from market_data.feature.fractional_difference import ZscoredFFDParams as BaseZscoredFFDParams
from market_data.feature.fractional_difference import get_zscored_ffd_series
from market_data.feature.impl.returns import _calculate_log_returns_numba, _calculate_simple_returns_numba
from market_data.feature.param import FeatureParam
from market_data.feature.common import Feature
from market_data.feature.registry import register_feature

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "ffd_zscore"

@dataclass
class ZscoredFFDParams(FeatureParam):
    """Parameters for FFD zscore feature calculations."""
    zscored_ffd_params: BaseZscoredFFDParams = field(default_factory=BaseZscoredFFDParams)
    cols: List[str] = field(default_factory=lambda: ["close", "volume"])
    
    def get_warm_up_period(self) -> datetime.timedelta:
        warm_up = max(
            self.zscored_ffd_params.zscore_window,
            100  # Conservative estimate for FFD weights convergence
        )
        return datetime.timedelta(minutes=warm_up)

    def to_str(self) -> str:
        """Convert parameters to string format: cols:[close,volume],d:0.5,threshold:0.01,zscore_window:100"""
        cols_str = '[' + ','.join(self.cols) + ']'
        return f"cols:{cols_str},d:{self.zscored_ffd_params.ffd_params.d},threshold:{self.zscored_ffd_params.ffd_params.threshold},zscore_window:{self.zscored_ffd_params.zscore_window}"
    
    @classmethod
    def from_str(cls, feature_label_str: str) -> 'ZscoredFFDParams':
        """Parse FFD zscore parameters from JSON-like format: cols:[close,volume],d:0.5,threshold:0.01,zscore_window:100"""
        params = {}
        base_params = BaseZscoredFFDParams()
        
        for pair in feature_label_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'cols':
                    # Parse [close,volume] format
                    if value.startswith('[') and value.endswith(']'):
                        cols_str = value[1:-1]
                        params['cols'] = [c.strip() for c in cols_str.split(',') if c.strip()]
                elif key == 'd':
                    base_params.ffd_params.d = float(value)
                elif key == 'threshold':
                    base_params.ffd_params.threshold = float(value)
                elif key == 'zscore_window':
                    base_params.zscore_window = int(value)
        
        params['zscored_ffd_params'] = base_params
        return cls(**params)


@register_feature(FEATURE_LABEL)
class ZscoredFFDsFeature(Feature):
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