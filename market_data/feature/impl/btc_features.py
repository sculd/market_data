"""
Bitcoin Features Module

This module provides functions for calculating Bitcoin-specific features
such as BTC returns for cross-referencing with other assets.
"""

import datetime
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from market_data.feature.param import FeatureParam
from market_data.feature.registry import register_feature

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "btc_features"

@dataclass
class BTCParams(FeatureParam):
    """Parameters for Bitcoin feature calculations."""
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 15, 30, 60, 120])
    price_col: str = "close"
    btc_symbol_patterns: List[str] = field(default_factory=lambda: ["BTC", "BTCUSDT", "BTC-USDT"])
    
    def get_warm_up_period(self) -> datetime.timedelta:
        return datetime.timedelta(minutes=max(self.return_periods))

    
    def to_str(self) -> str:
        """Convert parameters to string format: return_periods:[1,5,15],price_col:close"""
        periods_str = '[' + ','.join(str(p) for p in self.return_periods) + ']'
        patterns_str = '[' + ','.join(self.btc_symbol_patterns) + ']'
        return f"return_periods:{periods_str},price_col:{self.price_col},btc_symbol_patterns:{patterns_str}"
    
    @classmethod
    def from_str(cls, feature_label_str: str) -> 'BTCParams':
        """Parse BTC parameters from JSON-like format: return_periods:[1,5,15],price_col:close"""
        params = {}
        for pair in feature_label_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'return_periods':
                    # Parse [1,5,15] format
                    if value.startswith('[') and value.endswith(']'):
                        periods_str = value[1:-1]
                        params['return_periods'] = [int(p.strip()) for p in periods_str.split(',') if p.strip()]
                elif key == 'price_col':
                    params['price_col'] = value
                elif key == 'btc_symbol_patterns':
                    # Parse [BTC,BTCUSDT] format
                    if value.startswith('[') and value.endswith(']'):
                        patterns_str = value[1:-1]
                        params['btc_symbol_patterns'] = [p.strip() for p in patterns_str.split(',') if p.strip()]
        return cls(**params)

@register_feature(FEATURE_LABEL)
class BTCFeature:
    """Bitcoin feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[BTCParams] = None) -> pd.DataFrame:
        """
        Calculate Bitcoin-specific features.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for BTC feature calculation
            
        Returns:
            DataFrame with calculated BTC features
        """
        if params is None:
            params = BTCParams()
            
        logger.info(f"Calculating Bitcoin features for periods {params.return_periods}")
        
        # Ensure we have the price column
        if params.price_col not in df.columns:
            raise ValueError(f"Price column '{params.price_col}' not found in DataFrame")
        
        # Ensure we have symbol column
        if 'symbol' not in df.columns:
            raise ValueError("Symbol column not found in DataFrame")
        
        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        
        # Identify Bitcoin data by symbol
        btc_symbols = [s for s in df['symbol'].unique() 
                       if any(pattern.upper() in s.upper() for pattern in params.btc_symbol_patterns)]
        
        if not btc_symbols:
            logger.warning("No Bitcoin symbol found in DataFrame. Cannot calculate BTC features.")
            # Return empty DataFrame with same index
            return result
        
        # Use the first matching BTC symbol found
        btc_symbol = btc_symbols[0]
        logger.info(f"Using {btc_symbol} as Bitcoin reference for BTC features")
        
        # If multiple BTC symbols exist, log a warning
        if len(btc_symbols) > 1:
            logger.warning(f"Multiple Bitcoin symbols found: {btc_symbols}. Using {btc_symbol}")
            
        # Get the BTC data
        btc_df = df[df['symbol'] == btc_symbol].copy()
        
        if btc_df.empty:
            logger.warning(f"No data found for BTC symbol {btc_symbol}")
            return result
            
        # Calculate BTC returns for each period
        for period in params.return_periods:
            # Calculate return
            btc_return = btc_df[params.price_col].pct_change(period)
            
            # Reindex to match the full index range
            btc_return = btc_return.groupby(btc_return.index).mean().reindex(df.index)
            
            # Add as a feature
            result[f'btc_return_{period}'] = btc_return
            
        # Ensure all rows have symbol from the original dataframe
        result['symbol'] = df['symbol']
        
        # Combine all symbol results
        result = result.reset_index().set_index(['timestamp', 'symbol'])
        
        return result 