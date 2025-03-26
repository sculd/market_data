import pandas as pd
import numpy as np
from typing import List, Union, Dict, Tuple, Literal
import logging
import warnings

logger = logging.getLogger(__name__)

class TargetEngineer:
    """
    Class for engineering target features from OHLCV market data.
    
    The input DataFrame should have:
    - 'timestamp' as index (with minute-level data)
    - Columns: 'symbol', 'open', 'high', 'low', 'close', 'volume'
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame of market data.
        
        Args:
            df: DataFrame with timestamp index and OHLCV columns
        """
        self.validate_dataframe(df)
        self.df = df.copy()
    
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
    
    def add_target_features(self, 
                           forward_periods: List[int] = [1, 5, 15, 30, 60, 120, 240, 480],
                           tp_values: List[float] = [0.005, 0.01, 0.02, 0.03, 0.05],
                           sl_values: List[float] = [0.005, 0.01, 0.02, 0.03, 0.05]) -> pd.DataFrame:
        """
        Add target features for machine learning, including forward returns and
        classification labels based on take-profit and stop-loss levels.
        
        Args:
            forward_periods: Periods (in minutes) for calculating forward returns
            tp_values: Take-profit threshold values (as decimal)
            sl_values: Stop-loss threshold values (as decimal)
            
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
            for period in forward_periods:
                target_data[f'forward_return_{period}m'] = self.calculate_forward_return(group, period)
            
            # Add classification labels for different configurations
            for period in forward_periods:
                for tp in tp_values:
                    for sl in sl_values:
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
    
    def get_target_column_names(self,
                              forward_periods: List[int] = [1, 5, 15, 30, 60, 120, 240, 480],
                              tp_values: List[float] = [0.005, 0.01, 0.02, 0.03, 0.05],
                              sl_values: List[float] = [0.005, 0.01, 0.02, 0.03, 0.05]) -> List[str]:
        """
        Get the names of target columns that would be created by add_target_features.
        
        Args:
            forward_periods: Periods (in minutes) for calculating forward returns
            tp_values: Take-profit threshold values (as decimal)
            sl_values: Stop-loss threshold values (as decimal)
            
        Returns:
            List of column names for target features
        """
        column_names = []
        
        # Forward return column names
        for period in forward_periods:
            column_names.append(f'forward_return_{period}m')
        
        # Classification label column names
        for period in forward_periods:
            for tp in tp_values:
                for sl in sl_values:
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
        # Initialize result series
        result = pd.Series(0, index=df.index)
        
        # Create an array to hold the results
        labels = np.zeros(len(df))
        
        # Setup price levels based on position type
        if position_type == 'long':
            # For long positions
            tp_prices = df['close'] * (1 + take_profit)
            sl_prices = df['close'] * (1 - stop_loss)
        else:
            # For short positions
            tp_prices = df['close'] * (1 - take_profit)
            sl_prices = df['close'] * (1 + stop_loss)
        
        # Vectorized version not possible due to variable-length forward windows
        # Process in chunks to reduce the number of individual updates
        chunk_size = 1000
        for chunk_start in range(0, len(df) - 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(df) - 1)
            
            for i in range(chunk_start, chunk_end):
                if i + period >= len(df):
                    # Not enough future data
                    continue
                
                entry_price = df['close'].iloc[i]
                future_window = df.iloc[i+1:i+period+1]
                
                if position_type == 'long':
                    # Check if take-profit reached (high price exceeds tp level)
                    if (future_window['high'] >= tp_prices.iloc[i]).any():
                        labels[i] = 1
                    # Check if stop-loss reached (low price below sl level)
                    elif (future_window['low'] <= sl_prices.iloc[i]).any():
                        labels[i] = -1
                else:
                    # Check if take-profit reached (low price below tp level)
                    if (future_window['low'] <= tp_prices.iloc[i]).any():
                        labels[i] = 1
                    # Check if stop-loss reached (high price exceeds sl level)
                    elif (future_window['high'] >= sl_prices.iloc[i]).any():
                        labels[i] = -1
        
        # Assign the computed labels to the result series at once
        result = pd.Series(labels, index=df.index)
        return result


# Convenience functions
def create_targets(df: pd.DataFrame, 
                  forward_periods: List[int] = [2, 10],
                  tp_values: List[float] = [0.03],
                  sl_values: List[float] = [0.03]) -> pd.DataFrame:
    """
    Convenience function to create ML target features from market data.
    Includes forward returns and classification labels.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        forward_periods: Periods (in minutes) for calculating forward returns
        tp_values: Take-profit threshold values (as decimal)
        sl_values: Stop-loss threshold values (as decimal)
        
    Returns:
        DataFrame with target features
    """
    engineer = TargetEngineer(df)
    # Create a copy to defragment
    return engineer.add_target_features(forward_periods, tp_values, sl_values).copy()


def get_target_columns(forward_periods: List[int] = [2, 10],
                       tp_values: List[float] = [0.03],
                       sl_values: List[float] = [0.03]) -> List[str]:
    """
    Get the names of all target columns that would be created by create_targets.
    
    Args:
        forward_periods: Periods (in minutes) for calculating forward returns
        tp_values: Take-profit threshold values (as decimal)
        sl_values: Stop-loss threshold values (as decimal)
        
    Returns:
        List of column names for target features
    """
    # Create a temporary instance with empty DataFrame
    empty_df = pd.DataFrame({'symbol': ['BTC'], 
                             'open': [1], 'high': [1], 'low': [1], 'close': [1], 'volume': [1]},
                           index=[pd.Timestamp.now()])
    engineer = TargetEngineer(empty_df)
    
    # Get column names
    return engineer.get_target_column_names(forward_periods, tp_values, sl_values)


# Predefined target column lists for common configurations
TARGET_COLUMNS_DEFAULT = get_target_columns(
    forward_periods=[2, 10],
    tp_values=[0.03],
    sl_values=[0.03]
)

TARGET_COLUMNS_FULL = get_target_columns(
    forward_periods=[1, 10, 30, 60, 120],
    tp_values=[0.005, 0.01, 0.02, 0.03, 0.05],
    sl_values=[0.005, 0.01, 0.02, 0.03, 0.05]
) 