import pandas as pd
import dask.dataframe as dd
from dask.delayed import delayed
import numpy as np
from typing import List, Union, Dict, Tuple, Literal
import logging
import warnings
import time

# Try to import numba for accelerated computations
try:
    import numba as nb
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

logger = logging.getLogger(__name__)

# Numba-accelerated functions for performance-critical calculations
if HAVE_NUMBA:
    @nb.njit(cache=True)
    def calculate_tp_sl_labels_numba(closes, highs, lows, period, take_profit, stop_loss, position_type):
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
        
        for i in range(n - 1):
            if i + period >= n:
                # Not enough future data
                continue
                
            entry_price = closes[i]
            
            if position_type == 0:  # long position
                tp_price = entry_price * (1 + take_profit)
                sl_price = entry_price * (1 - stop_loss)
                
                # Check forward window for TP/SL hits
                for j in range(i + 1, min(i + period + 1, n)):
                    if highs[j] >= tp_price:
                        labels[i] = 1
                        break
                    elif lows[j] <= sl_price:
                        labels[i] = -1
                        break
            else:  # short position
                tp_price = entry_price * (1 - take_profit)
                sl_price = entry_price * (1 + stop_loss)
                
                # Check forward window for TP/SL hits
                for j in range(i + 1, min(i + period + 1, n)):
                    if lows[j] <= tp_price:
                        labels[i] = 1
                        break
                    elif highs[j] >= sl_price:
                        labels[i] = -1
                        break
                
        return labels

class TargetEngineer:
    """
    Class for engineering target features from OHLCV market data, using efficient Dask-based processing.
    
    The input DataFrame should have:
    - 'timestamp' as index (with minute-level data)
    - Columns: 'symbol', 'open', 'high', 'low', 'close', 'volume'
    """
    
    def __init__(self, df: Union[pd.DataFrame, dd.DataFrame]):
        """
        Initialize with a DataFrame of market data.
        
        Args:
            df: DataFrame with timestamp index and OHLCV columns (pandas or Dask)
        """
        # Convert to Dask if input is pandas
        if isinstance(df, pd.DataFrame):
            # Convert to Dask with efficient partitioning
            symbols = df['symbol'].unique()
            npartitions = min(len(symbols), max(4, min(8, len(df) // 50000)))
            logger.info(f"Converting pandas DataFrame to Dask with {npartitions} partitions")

            # For better performance, sort by symbol to ensure clean partitioning
            sorted_df = df.sort_values('symbol')
            self.df = dd.from_pandas(sorted_df, npartitions=npartitions)
        else:
            # Already a Dask DataFrame
            self.df = df
        
        self.validate_dataframe(self.df)
    
    def validate_dataframe(self, df: Union[pd.DataFrame, dd.DataFrame]) -> None:
        """Validate that the DataFrame has the required structure."""
        required_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # Check index for Dask DataFrame
        if isinstance(df, dd.DataFrame):
            if not df.index.name and not df.index._meta.name:
                logger.warning("DataFrame index may not be a DatetimeIndex. Assuming it needs conversion.")
                try:
                    df = df.map_partitions(lambda pdf: pdf.set_index(pd.to_datetime(pdf.index)))
                except Exception as e:
                    raise ValueError(f"Failed to convert index to datetime: {e}")
        else:
            # For pandas DataFrame
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
        start_time = time.time()
        logger.info("Starting target feature engineering using Dask implementation")
        
        # Define function to process each partition
        def process_symbol_partition(partition):
            if len(partition) == 0:
                return pd.DataFrame()
            
            symbol = partition['symbol'].iloc[0]
            
            # Create feature data dictionary with the right column order
            target_data = {'symbol': symbol}
            
            # Add forward returns for different periods
            for period in forward_periods:
                target_data[f'label_forward_return_{period}m'] = self.calculate_forward_return(partition, period)
            
            # Add classification labels for different configurations
            for period in forward_periods:
                for tp in tp_values:
                    for sl in sl_values:
                        # Long position labels
                        target_data[f'label_long_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m'] = \
                            self.calculate_tp_sl_labels(partition, period, tp, sl, 'long')
                        
                        # Short position labels
                        target_data[f'label_short_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m'] = \
                            self.calculate_tp_sl_labels(partition, period, tp, sl, 'short')
            
            return pd.DataFrame(target_data, index=partition.index)
        
        # Define the expected column order for the metadata
        meta_columns = ['symbol']
        meta_columns.extend([f'label_forward_return_{p}m' for p in forward_periods])
        
        # Add classification label columns to metadata
        for period in forward_periods:
            for tp in tp_values:
                for sl in sl_values:
                    meta_columns.append(f'label_long_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m')
                    meta_columns.append(f'label_short_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m')
        
        # Create metadata dictionary
        meta = {col: 'float64' if col != 'symbol' else 'object' for col in meta_columns}
        
        # Apply function to each partition
        logger.info(f"Processing target features using Dask with {self.df.npartitions} partitions")
        result_dask = self.df.map_partitions(process_symbol_partition, meta=meta)
        
        # Compute the result
        logger.info("Computing Dask result using processes scheduler")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = result_dask.compute(scheduler="processes")
        
        elapsed = time.time() - start_time
        logger.info(f"Target feature engineering completed in {elapsed:.2f} seconds")
        return result_df
    
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
            column_names.append(f'label_forward_return_{period}m')
        
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
        # Use Numba-accelerated function if available
        if HAVE_NUMBA:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Convert position type to integer for Numba
            pos_type_int = 0 if position_type == 'long' else 1
            
            # Call Numba function
            labels = calculate_tp_sl_labels_numba(
                closes, highs, lows, period, take_profit, stop_loss, pos_type_int
            )
            
            return pd.Series(labels, index=df.index)
        
        # Fallback implementation
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
def create_targets(df: Union[pd.DataFrame, dd.DataFrame], 
                  forward_periods: List[int] = [2, 10],
                  tp_values: List[float] = [0.03],
                  sl_values: List[float] = [0.03],
                  n_workers: int = None) -> pd.DataFrame:
    """
    Convenience function to create ML target features from market data.
    Includes forward returns and classification labels.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        forward_periods: Periods (in minutes) for calculating forward returns
        tp_values: Take-profit threshold values (as decimal)
        sl_values: Stop-loss threshold values (as decimal)
        n_workers: Number of worker processes to use (default: auto)
        
    Returns:
        DataFrame with target features
    """
    # Configure workers if specified
    if n_workers is not None and n_workers > 0:
        import os
        os.environ["NUMBA_NUM_THREADS"] = str(n_workers)
        os.environ["OMP_NUM_THREADS"] = str(n_workers)
    
    engineer = TargetEngineer(df)
    return engineer.add_target_features(forward_periods, tp_values, sl_values)


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