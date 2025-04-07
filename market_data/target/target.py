import pandas as pd
import numpy as np
from typing import List, Union, Dict, Tuple, Literal
import logging
import warnings
import numba as nb
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default values for target parameters
DEFAULT_FORWARD_PERIODS = [2, 10]
DEFAULT_TP_VALUES = [0.03]
DEFAULT_SL_VALUES = [0.03]

# Default values for complete target parameters (used in add_target_features)
FULL_FORWARD_PERIODS = [1, 5, 15, 30, 60, 120, 240, 480]
FULL_TP_VALUES = [0.005, 0.01, 0.02, 0.03, 0.05]
FULL_SL_VALUES = [0.005, 0.01, 0.02, 0.03, 0.05]

@dataclass
class TargetParams:
    """
    Encapsulates parameters for target engineering.
    
    This class holds all the parameters needed for target calculation,
    providing a single source of truth for target configuration.
    
    Attributes:
        forward_periods (List[int]): Periods (in minutes) for calculating forward returns
        tp_values (List[float]): Take-profit threshold values (as decimal)
        sl_values (List[float]): Stop-loss threshold values (as decimal)
    """
    forward_periods: List[int] = field(default_factory=lambda: DEFAULT_FORWARD_PERIODS)
    tp_values: List[float] = field(default_factory=lambda: DEFAULT_TP_VALUES)
    sl_values: List[float] = field(default_factory=lambda: DEFAULT_SL_VALUES)
    
    def __repr__(self) -> str:
        """String representation of the parameters."""
        return (
            f"TargetParams(forward_periods={self.forward_periods}, "
            f"tp_values={self.tp_values}, "
            f"sl_values={self.sl_values})"
        )

# Numba-accelerated functions for performance-critical calculations
@nb.njit(cache=True)
def _calculate_tp_sl_labels_numba(closes, highs, lows, period, take_profit, stop_loss, position_type):
    """
    Numba-accelerated take-profit/stop-loss label calculation.
    
    Args:
        closes: Array of close prices
        highs: Array of high prices
        lows: Array of low prices
        period: Forward-looking period
        take_profit: Take-profit threshold as decimal
        stop_loss: Stop-loss threshold as decimal
        position_type: 0 for long, 1 for short
        
    Returns:
        Array with classification labels (1=TP hit, -1=SL hit, 0=neither)
    """
    n = len(closes)
    labels = np.zeros(n)
    
    # Pre-compute TP/SL price levels for all data points
    if position_type == 0:  # long position
        tp_prices = closes * (1 + take_profit)
        sl_prices = closes * (1 - stop_loss)
    else:  # short position
        tp_prices = closes * (1 - take_profit)
        sl_prices = closes * (1 + stop_loss)
    
    # Process each position
    for i in range(n - 1):
        if i + period >= n:
            # Not enough future data
            continue
                
        if position_type == 0:  # long position
            # Check forward window for TP/SL hits
            for j in range(i + 1, min(i + period + 1, n)):
                if highs[j] >= tp_prices[i]:
                    labels[i] = 1
                    break
                elif lows[j] <= sl_prices[i]:
                    labels[i] = -1
                    break
        else:  # short position
            # Check forward window for TP/SL hits
            for j in range(i + 1, min(i + period + 1, n)):
                if lows[j] <= tp_prices[i]:
                    labels[i] = 1
                    break
                elif highs[j] >= sl_prices[i]:
                    labels[i] = -1
                    break
                
    return labels

class TargetEngineer:
    """
    Class for engineering target features from OHLCV market data.
    
    The input DataFrame should have:
    - 'timestamp' as index (with minute-level data)
    - Columns: 'symbol', 'open', 'high', 'low', 'close', 'volume'
    """
    
    def __init__(self, df: pd.DataFrame, params: TargetParams = None):
        """
        Initialize with a DataFrame of market data.
        
        Args:
            df: DataFrame with timestamp index and OHLCV columns
            params: Target parameters. If None, uses default parameters
        """
        self.validate_dataframe(df)
        self.df = df.copy()
        self.params = params or TargetParams()
    
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that the DataFrame has the required structure."""
        required_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not a DatetimeIndex. Converting to datetime.")
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Failed to convert index to datetime: {e}")
    
    def add_target_features(self) -> pd.DataFrame:
        """
        Add target features for machine learning, including forward returns and
        classification labels based on take-profit and stop-loss levels.
        
        Returns:
            DataFrame with target features added
        """
        # Create an empty list to hold all target DataFrames
        all_targets = []
        
        # Group by symbol to calculate per-symbol targets
        for symbol, group in self.df.groupby('symbol'):
            # Create a dictionary to hold all target columns
            target_data = {'symbol': symbol}
            
            # Add forward returns for different periods
            for period in self.params.forward_periods:
                target_data[f'label_forward_return_{period}m'] = self.calculate_forward_return(group, period)
            
            # Add classification labels for different configurations
            for period in self.params.forward_periods:
                for tp in self.params.tp_values:
                    for sl in self.params.sl_values:
                        # Long position labels
                        target_data[f'label_long_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m'] = \
                            self.calculate_tp_sl_labels(group, period, tp, sl, 'long')
                        
                        # Short position labels
                        target_data[f'label_short_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m'] = \
                            self.calculate_tp_sl_labels(group, period, tp, sl, 'short')
            
            # Create a single DataFrame from all targets at once
            symbol_targets = pd.DataFrame(target_data, index=group.index)
            
            # Append to the list of targets
            all_targets.append(symbol_targets)
        
        # Concatenate all symbol targets into a single DataFrame
        if all_targets:
            result_df = pd.concat(all_targets)
            return result_df
        else:
            # Return an empty DataFrame with only the symbol column
            return pd.DataFrame({'symbol': []}, index=self.df.index)
    
    def get_target_column_names(self) -> List[str]:
        """
        Get the names of target columns that would be created by add_target_features.
        
        Returns:
            List of column names for target features
        """
        column_names = []
        
        # Forward return column names
        for period in self.params.forward_periods:
            column_names.append(f'label_forward_return_{period}m')
        
        # Classification label column names
        for period in self.params.forward_periods:
            for tp in self.params.tp_values:
                for sl in self.params.sl_values:
                    # Long position labels
                    column_names.append(f'label_long_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m')
                    # Short position labels
                    column_names.append(f'label_short_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m')
        
        return column_names
    
    @staticmethod
    def calculate_forward_return(df: pd.DataFrame, period: int = 1) -> pd.Series:
        """
        Calculate forward return over specified period in minutes.
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods (minutes) to look forward
            
        Returns:
            Series with forward returns
        """
        # Reverse the shift direction for forward returns
        return df['close'].pct_change(period).shift(-period)
    
    @staticmethod
    def calculate_tp_sl_labels(df: pd.DataFrame, 
                              period: int, 
                              take_profit: float, 
                              stop_loss: float, 
                              position_type: Literal['long', 'short'] = 'long') -> pd.Series:
        """
        Calculate classification labels based on take-profit and stop-loss levels.
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods (minutes) to look forward
            take_profit: Take-profit threshold as decimal (e.g., 0.01 for 1%)
            stop_loss: Stop-loss threshold as decimal (e.g., 0.01 for 1%)
            position_type: 'long' or 'short'
            
        Returns:
            Series with classification labels:
                1: take-profit reached within period
               -1: stop-loss reached within period
                0: neither reached within period
        """
        # Extract arrays for Numba processing
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # Convert position_type to integer for Numba
        pos_type_int = 0 if position_type == 'long' else 1
        
        # Call Numba-accelerated function
        labels = _calculate_tp_sl_labels_numba(
            closes, highs, lows, period, take_profit, stop_loss, pos_type_int
        )
        
        # Convert result back to pandas Series
        return pd.Series(labels, index=df.index)


# Convenience functions
def create_targets(df: pd.DataFrame, params: TargetParams = None) -> pd.DataFrame:
    """
    Convenience function to create ML target features from market data.
    Includes forward returns and classification labels.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        params: Target calculation parameters. If None, uses default parameters.
        
    Returns:
        DataFrame with target features
    """
    # Configure Numba threads
    num_threads = os.cpu_count() or 4
    nb.set_num_threads(min(num_threads, 8))  # Limit to max 8 threads
    
    engineer = TargetEngineer(df, params=params)
    # Create a copy to defragment
    return engineer.add_target_features().copy()


def get_target_columns(params: TargetParams = None) -> List[str]:
    """
    Get the names of all target columns that would be created by create_targets.
    
    Args:
        params: Target calculation parameters. If None, uses default parameters.
        
    Returns:
        List of column names for target features
    """
    # Create a temporary instance with empty DataFrame
    empty_df = pd.DataFrame({'symbol': ['BTC'], 
                             'open': [1], 'high': [1], 'low': [1], 'close': [1], 'volume': [1]},
                           index=[pd.Timestamp.now()])
    engineer = TargetEngineer(empty_df, params=params)
    
    # Get column names
    return engineer.get_target_column_names()


# Predefined target column lists for common configurations
TARGET_COLUMNS_DEFAULT = get_target_columns(TargetParams())
TARGET_COLUMNS_FULL = get_target_columns(TargetParams(
    forward_periods=FULL_FORWARD_PERIODS,
    tp_values=FULL_TP_VALUES,
    sl_values=FULL_SL_VALUES
)) 