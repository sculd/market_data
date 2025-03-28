import pandas as pd
import polars as pl
import numpy as np
from typing import List, Union, Dict, Tuple, Literal
import logging
import warnings
import datetime

logger = logging.getLogger(__name__)

class TargetEngineer:
    """
    Class for engineering target features from OHLCV market data.
    
    The input DataFrame should have:
    - 'timestamp' column containing datetime values
    - Columns: 'symbol', 'open', 'high', 'low', 'close', 'volume'
    """
    
    def __init__(self, df: pl.DataFrame):
        """
        Initialize with a DataFrame of market data.
        
        Args:
            df: DataFrame with timestamp column and OHLCV columns
        """
        # First check for timestamp column existence
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column")
            
        # Validate other aspects of the dataframe - this validation is now safer
        self.validate_dataframe(df)
        
        # Sort by timestamp and symbol for consistent processing
        self.df = df.sort(by=['timestamp', 'symbol'])
    
    def validate_dataframe(self, df: pl.DataFrame) -> None:
        """Validate that the DataFrame has the required structure."""
        required_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # Ensure timestamp column has proper type - with safer type checking
        if 'timestamp' in df.columns:
            timestamp_type = df['timestamp'].dtype
            is_datetime = (
                timestamp_type == pl.Datetime or 
                str(timestamp_type).lower().startswith('datetime') or
                'date' in str(timestamp_type).lower()
            )
            
            if not is_datetime:
                logger.warning("DataFrame timestamp column is not a Datetime type. Converting to datetime.")
                try:
                    # Create a new DataFrame with converted timestamp
                    # This is safer than modifying the original
                    timestamp_col = df['timestamp']
                    # Try different approaches to convert to datetime
                    try:
                        # First try direct casting
                        df = df.with_columns(pl.col('timestamp').cast(pl.Datetime))
                    except Exception:
                        # If that fails, try to parse as string
                        df = df.with_columns(pl.col('timestamp').str.to_datetime())
                except Exception as e:
                    logger.warning(f"Failed to convert timestamp to datetime: {e}")
                    # Continue without conversion rather than failing
                    pass
    
    def add_target_features(self, 
                           forward_periods: List[int] = [1, 5, 15, 30, 60, 120, 240, 480],
                           tp_values: List[float] = [0.005, 0.01, 0.02, 0.03, 0.05],
                           sl_values: List[float] = [0.005, 0.01, 0.02, 0.03, 0.05]) -> pl.DataFrame:
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
        
        # Process each symbol
        for symbol in self.df['symbol'].unique():
            # Filter data for this symbol
            symbol_df = self.df.filter(pl.col('symbol') == symbol)
            
            # Start with timestamp and symbol
            target_exprs = [
                pl.col('timestamp'),
                pl.lit(symbol).alias('symbol')
            ]
            
            # Add forward returns for different periods - these are fast as expressions
            for period in forward_periods:
                target_exprs.append(
                    self.calculate_forward_return_expr(period).alias(f'label_forward_return_{period}m')
                )
            
            # Create the basic target DataFrame with timestamp, symbol and forward returns
            symbol_targets = symbol_df.select(target_exprs)
            
            # Using an optimized approach for TP/SL label calculation
            # We'll calculate these in batches per period to minimize redundant processing
            
            # For each forward period, calculate all TP/SL combinations at once
            for period in forward_periods:
                # Create empty dictionaries to collect labels before adding to DataFrame
                long_labels_dict = {}
                short_labels_dict = {}
                
                # Process each TP/SL combination for this period
                for tp in tp_values:
                    for sl in sl_values:
                        # Generate column names
                        long_col_name = f'label_long_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m'
                        short_col_name = f'label_short_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m'
                        
                        # Get the label arrays
                        long_labels = self.calculate_tp_sl_labels(
                            symbol_df, period, tp, sl, 'long'
                        )
                        short_labels = self.calculate_tp_sl_labels(
                            symbol_df, period, tp, sl, 'short'
                        )
                        
                        # Store in dictionaries
                        long_labels_dict[long_col_name] = long_labels
                        short_labels_dict[short_col_name] = short_labels
                
                # Add all labels for this period at once
                columns_to_add = []
                for col_name, label_expr in {**long_labels_dict, **short_labels_dict}.items():
                    columns_to_add.append(label_expr.alias(col_name))
                
                if columns_to_add:
                    symbol_targets = symbol_targets.with_columns(columns_to_add)
            
            # Append to the list of targets
            all_targets.append(symbol_targets)
        
        # Concatenate all symbol targets into a single DataFrame
        if all_targets:
            result_df = pl.concat(all_targets)
            return result_df
        else:
            # Return an empty DataFrame
            return pl.DataFrame({'timestamp': [], 'symbol': []})
    
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
    def calculate_forward_return_expr(period: int = 1) -> pl.Expr:
        """
        Calculate forward return over specified period in minutes as an expression.
        
        Args:
            period: Number of periods (minutes) to look forward
            
        Returns:
            Polars expression for forward return
        """
        # Reverse the shift direction for forward returns
        # In Polars, reverse shift is negative, just like pandas
        return pl.col('close').pct_change(period).shift(-period)
    
    @staticmethod
    def calculate_tp_sl_labels(df: pl.DataFrame, 
                              period: int, 
                              take_profit: float, 
                              stop_loss: float, 
                              position_type: Literal['long', 'short'] = 'long') -> pl.Expr:
        """
        Calculate classification labels based on take-profit and stop-loss levels.
        
        Args:
            df: DataFrame with OHLCV data
            period: Number of periods (minutes) to look forward
            take_profit: Take-profit threshold as decimal (e.g., 0.01 for 1%)
            stop_loss: Stop-loss threshold as decimal (e.g., 0.01 for 1%)
            position_type: 'long' or 'short'
            
        Returns:
            Polars expression with classification labels:
                1: take-profit reached within period
               -1: stop-loss reached within period
                0: neither reached within period
        """
        try:
            # Try to use Numba for high-performance calculation
            import numba
            
            # Define a JIT-compiled function for calculating TP/SL labels
            # Use cache=True to reuse compiled code across calls
            @numba.njit(cache=True)
            def calculate_labels_fast(high_values, low_values, close_values, take_profit, stop_loss, period, n_rows, is_long):
                # Pre-allocate result array
                result = np.zeros(n_rows, dtype=np.int8)
                
                # Process each row
                for i in range(n_rows):
                    # Skip if we don't have enough future data
                    if i + period >= n_rows:
                        continue
                    
                    # Calculate TP/SL prices
                    base_price = close_values[i]
                    if is_long:
                        tp_price = base_price * (1 + take_profit)
                        sl_price = base_price * (1 - stop_loss)
                    else:
                        tp_price = base_price * (1 - take_profit)
                        sl_price = base_price * (1 + stop_loss)
                    
                    # Check future window
                    for j in range(i + 1, min(i + period + 1, n_rows)):
                        if is_long:
                            # Long position
                            if high_values[j] >= tp_price:
                                result[i] = 1
                                break
                            if low_values[j] <= sl_price:
                                result[i] = -1
                                break
                        else:
                            # Short position
                            if low_values[j] <= tp_price:
                                result[i] = 1
                                break
                            if high_values[j] >= sl_price:
                                result[i] = -1
                                break
                
                return result
            
            # Extract necessary arrays from DataFrame for efficient processing
            high_values = df.select('high').to_numpy().flatten()
            low_values = df.select('low').to_numpy().flatten()
            close_values = df.select('close').to_numpy().flatten()
            
            # Calculate labels
            n_rows = len(high_values)
            is_long = position_type == 'long'
            
            # Simple JIT-compiled calculation
            result = calculate_labels_fast(
                high_values, low_values, close_values,
                take_profit, stop_loss, period, n_rows, is_long
            )
            
        except (ImportError, Exception) as e:
            # Fallback if Numba is not available or there's an error
            if isinstance(e, ImportError):
                logger.warning("Numba not available. Using slower implementation for TP/SL calculation.")
            else:
                logger.warning(f"Error in Numba calculation: {e}. Using slower implementation.")
            
            # Create a base DataFrame with price levels
            if position_type == 'long':
                # For long positions
                tp_price_expr = pl.col('close') * (1 + take_profit)
                sl_price_expr = pl.col('close') * (1 - stop_loss)
            else:
                # For short positions
                tp_price_expr = pl.col('close') * (1 - take_profit)
                sl_price_expr = pl.col('close') * (1 + stop_loss)
            
            # Create a base DataFrame with price levels
            base_df = df.select([
                'high',
                'low',
                'close',
                tp_price_expr.alias('tp_price'),
                sl_price_expr.alias('sl_price')
            ])
            
            # Convert to numpy arrays for faster processing
            high_values = base_df.select('high').to_numpy().flatten()
            low_values = base_df.select('low').to_numpy().flatten()
            tp_prices = base_df.select('tp_price').to_numpy().flatten()
            sl_prices = base_df.select('sl_price').to_numpy().flatten()
            
            # Initialize result array
            result = np.zeros(len(high_values), dtype=np.int8)
            
            # Process in chunks to improve performance
            chunk_size = 1000
            for chunk_start in range(0, len(high_values) - 1, chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(high_values) - 1)
                
                for i in range(chunk_start, chunk_end):
                    if i + period >= len(high_values):
                        # Not enough future data
                        continue
                    
                    # Get target prices
                    tp_price = tp_prices[i]
                    sl_price = sl_prices[i]
                    
                    # Process future window
                    hit_tp = False
                    hit_sl = False
                    
                    # Check each future price
                    for j in range(i + 1, min(i + period + 1, len(high_values))):
                        if position_type == 'long':
                            # Long position: high price hits TP, low price hits SL
                            if high_values[j] >= tp_price:
                                hit_tp = True
                                break
                            if low_values[j] <= sl_price:
                                hit_sl = True
                                break
                        else:
                            # Short position: low price hits TP, high price hits SL
                            if low_values[j] <= tp_price:
                                hit_tp = True
                                break
                            if high_values[j] >= sl_price:
                                hit_sl = True
                                break
                    
                    # Set result
                    if hit_tp:
                        result[i] = 1
                    elif hit_sl:
                        result[i] = -1
        
        # Convert results to a Polars expression
        return pl.lit(result.tolist())


# Convenience functions
def create_targets(df: Union[pl.DataFrame, pd.DataFrame], 
                  forward_periods: List[int] = [2, 10],
                  tp_values: List[float] = [0.03],
                  sl_values: List[float] = [0.03]) -> pl.DataFrame:
    """
    Convenience function to create ML target features from market data.
    Includes forward returns and classification labels.
    
    Args:
        df: DataFrame with timestamp column and OHLCV columns
        forward_periods: Periods (in minutes) for calculating forward returns
        tp_values: Take-profit threshold values (as decimal)
        sl_values: Stop-loss threshold values (as decimal)
        
    Returns:
        DataFrame with target features
    """
    # Convert pandas DataFrame to Polars if needed
    if not isinstance(df, pl.DataFrame):
        logger.info("Converting pandas DataFrame to Polars")
        if isinstance(df.index, pd.DatetimeIndex):
            # If timestamp is in the index, move it to a column
            df_reset = df.reset_index()
            df_pl = pl.from_pandas(df_reset)
        else:
            df_pl = pl.from_pandas(df)
    else:
        df_pl = df
    
    engineer = TargetEngineer(df_pl)
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
    # Create a temporary instance with empty DataFrame - use proper datetime
    empty_df = pl.DataFrame({
        'timestamp': [datetime.datetime(2023, 1, 1)],  # Use Python's datetime
        'symbol': ['BTC'], 
        'open': [1.0], 'high': [1.0], 'low': [1.0], 'close': [1.0], 'volume': [1.0]
    })
    
    # Skip validation to avoid casting issues
    engineer = TargetEngineer.__new__(TargetEngineer)
    engineer.df = empty_df  # Set the df attribute directly
    
    # Get column names
    return engineer.get_target_column_names(forward_periods, tp_values, sl_values)


# Predefined target column lists for common configurations
# Initialize these in a way that won't cause import errors
try:
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
except Exception as e:
    # Fallback to empty lists if there's any issue during import
    logger.warning(f"Failed to initialize target columns: {e}")
    TARGET_COLUMNS_DEFAULT = []
    TARGET_COLUMNS_FULL = [] 