import pandas as pd
import polars as pl
import numpy as np
from typing import List, Union, Dict, Tuple, Literal, Optional
import logging
import warnings

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Class for engineering features from OHLCV market data.
    
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
        self.validate_dataframe(df)
        
        # Ensure timestamp is properly set
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column")
        
        # Sort by timestamp
        self.df = df.sort(by=['timestamp', 'symbol'])
        self.btc_df = None
        
        # Identify Bitcoin data by looking for common BTC patterns in symbol names
        # This handles different exchange formats: BTC, BTCUSDT, BTC-USDT-SWAP, BTC/USDT, etc.
        btc_symbols = [s for s in self.df['symbol'].unique() if 'BTC' in s.upper()]
        if btc_symbols:
            # Use the first matching BTC symbol found
            btc_symbol = btc_symbols[0]
            logger.info(f"Using {btc_symbol} as Bitcoin reference for BTC features")
            self.btc_df = self.df.filter(pl.col('symbol') == btc_symbol)
            # If multiple BTC symbols exist, log a warning
            if len(btc_symbols) > 1:
                logger.warning(f"Multiple Bitcoin symbols found: {btc_symbols}. Using {btc_symbol}")
    
    def validate_dataframe(self, df: pl.DataFrame) -> None:
        """Validate that the DataFrame has the required structure."""
        required_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # Ensure timestamp column has proper type
        if 'timestamp' in df.columns and df['timestamp'].dtype != pl.Datetime:
            logger.warning("DataFrame timestamp column is not a Datetime type. Converting to datetime.")
            try:
                # Attempt to cast to datetime
                df = df.with_columns(pl.col('timestamp').cast(pl.Datetime))
            except Exception as e:
                raise ValueError(f"Failed to convert timestamp to datetime: {e}")
    
    def add_all_features(self, return_periods: List[int] = [1, 5, 15, 30, 60, 120], 
                         ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
                         add_btc_features: bool = True) -> pl.DataFrame:
        """
        Add all features to the DataFrame.
        
        Args:
            return_periods: Periods (in minutes) for calculating returns
            ema_periods: Periods (in minutes) for calculating EMAs
            add_btc_features: Whether to add BTC-related features
            
        Returns:
            DataFrame with all features added (excluding raw OHLCV data)
        """
        # List to collect all symbol feature DataFrames
        all_features = []
        
        # Track all column names to ensure consistent structure
        all_column_names = set(['timestamp', 'symbol'])
        
        # First pass: build all features and collect column names
        symbol_features_dict = {}
        
        # Create default "absent" column prototypes with correct types
        # Most numeric features will be float64, so we'll use that as the default
        numeric_null = pl.lit(None).cast(pl.Float64)
        
        # Add all potential columns and their data types to track
        column_types = {
            'timestamp': pl.Datetime,
            'symbol': pl.Utf8,
        }
        
        # Pre-define all return types
        for period in return_periods:
            column_types[f'return_{period}m'] = pl.Float64
            if add_btc_features:
                column_types[f'btc_return_{period}m'] = pl.Float64
                
        # Pre-define all EMA types
        for period in ema_periods:
            column_types[f'ema_{period}m'] = pl.Float64
            column_types[f'ema_rel_{period}m'] = pl.Float64
        
        # Other numeric columns
        numeric_columns = [
            'bb_position', 'bb_width', 'true_range', 'open_close_ratio',
            'rsi', 'autocorr_lag1', 'close_zscore', 'close_minmax',
            'hl_range_pct', 'volume_ratio_20m', 'obv', 'obv_zscore'
        ]
        for col in numeric_columns:
            column_types[col] = pl.Float64
        
        for symbol in self.df['symbol'].unique():
            # Filter data for this symbol
            symbol_df = self.df.filter(pl.col('symbol') == symbol)
            
            # Start with the timestamp and symbol columns
            feature_expressions = [
                pl.col('timestamp'),
                pl.lit(symbol).alias('symbol')
            ]
            
            # Price indicators - Returns
            for period in return_periods:
                feature_expressions.append(
                    self.calculate_return_expr(period).alias(f'return_{period}m')
                )
            
            # EMAs and EMA Ratios
            for period in ema_periods:
                ema_expr = self.calculate_ema_expr(period)
                feature_expressions.append(ema_expr.alias(f'ema_{period}m'))
                feature_expressions.append(
                    (pl.col('close') / ema_expr).alias(f'ema_rel_{period}m')
                )
            
            # Bollinger Bands
            bb_period = 20
            bb_upper_expr, bb_middle_expr, bb_lower_expr = self.calculate_bollinger_bands_expr(bb_period, 2)
            feature_expressions.append(
                ((pl.col('close') - bb_lower_expr) / (bb_upper_expr - bb_lower_expr)).alias('bb_position')
            )
            feature_expressions.append(
                ((bb_upper_expr - bb_lower_expr) / bb_middle_expr).alias('bb_width')
            )
            
            # True Range
            feature_expressions.append(
                self.calculate_true_range_expr().alias('true_range')
            )
            
            # Open/Close ratio
            feature_expressions.append(
                self.calculate_open_close_ratio_expr().alias('open_close_ratio')
            )
            
            # RSI (14 period) - already returns expression
            feature_expressions.append(
                self.calculate_rsi_expr(14).alias('rsi')
            )
            
            # Autocorrelation lag 1
            feature_expressions.append(
                self.calculate_autocorr_expr(5).alias('autocorr_lag1')
            )
            
            # Z-score of close price
            feature_expressions.append(
                self.calculate_zscore_expr('close', 20).alias('close_zscore')
            )
            
            # Min-max scaling
            feature_expressions.append(
                self.calculate_minmax_scale_expr('close', 20).alias('close_minmax')
            )
            
            # High-Low range as percentage of price
            feature_expressions.append(
                self.calculate_hl_range_pct_expr().alias('hl_range_pct')
            )
            
            # Volume ratio
            feature_expressions.append(
                self.calculate_volume_ratio_expr(20).alias('volume_ratio_20m')
            )
            
            # On-Balance Volume (OBV) is challenging to express directly in Polars expressions
            # We'll handle it separately
            obv_df = self.calculate_obv(symbol_df)
            
            # Create the features DataFrame
            symbol_features = symbol_df.select(feature_expressions)
            
            # Add OBV and OBV Z-score by first joining with the OBV DataFrame
            symbol_features = symbol_features.join(
                obv_df,
                on='timestamp',
                how='left'
            )
            
            # Now calculate the OBV Z-score using the now-available 'obv' column
            symbol_features = symbol_features.with_columns([
                self.calculate_zscore_expr('obv', 20).alias('obv_zscore')
            ])
            
            # Add BTC features if requested
            is_btc_symbol = 'BTC' in symbol.upper()
            if add_btc_features and self.btc_df is not None and not is_btc_symbol:
                btc_features_df = self.calculate_btc_features(symbol_df['timestamp'], return_periods)
                symbol_features = symbol_features.join(
                    btc_features_df, 
                    on='timestamp', 
                    how='left'
                )
            
            # Store this symbol's features
            symbol_features_dict[symbol] = symbol_features
            
            # Update the set of all column names
            all_column_names.update(symbol_features.columns)
            
            # Update column types from this DataFrame
            for col in symbol_features.columns:
                if col not in column_types:
                    column_types[col] = symbol_features[col].dtype
        
        # Convert to a sorted list to ensure consistent order
        all_columns_sorted = sorted(list(all_column_names))
        
        # Second pass: ensure all DataFrames have the same columns in the same order
        for symbol, df in symbol_features_dict.items():
            # Create a list of expressions for selecting or creating columns
            select_exprs = []
            
            for col in all_columns_sorted:
                if col in df.columns:
                    # Column exists, just select it
                    select_exprs.append(pl.col(col))
                else:
                    # Column doesn't exist, create a NULL with proper type
                    col_type = column_types.get(col, pl.Float64)  # Default to Float64 if unknown
                    select_exprs.append(pl.lit(None).cast(col_type).alias(col))
            
            # Select or create all columns with proper types
            df_with_all_columns = df.select(select_exprs)
            all_features.append(df_with_all_columns)
        
        # Combine all symbol feature DataFrames
        if all_features:
            result_df = pl.concat(all_features)
            return result_df
        else:
            # Return an empty DataFrame with only the symbol column
            return pl.DataFrame({'timestamp': [], 'symbol': []})
    
    def create_sequence_features(self, sequence_length: int = 60,
                               return_periods: List[int] = [1, 5, 15, 30, 60, 120], 
                               ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
                               add_btc_features: bool = True) -> pl.DataFrame:
        """
        Create a DataFrame with sequence features only.
        Each column contains arrays of past values for each feature.
        Raw OHLCV columns are excluded.
        
        Args:
            sequence_length: Length of the sequence arrays
            return_periods: Periods (in minutes) for calculating returns
            ema_periods: Periods (in minutes) for calculating EMAs
            add_btc_features: Whether to add BTC-related features
            
        Returns:
            DataFrame with sequence features only
        """
        # Replace the Numba-based function with a pure Python implementation
        def create_sequences(values, sequence_length):
            n = len(values)
            sequences = np.zeros((n, sequence_length), dtype=np.float64) * np.nan
            
            for i in range(n):
                start_idx = max(0, i - sequence_length + 1)
                actual_seq_len = i - start_idx + 1
                if actual_seq_len > 0:
                    sequences[i, -actual_seq_len:] = values[start_idx:i+1]
            
            return sequences
        
        # First get the regular features (which already exclude raw OHLCV)
        features_df = self.add_all_features(
            return_periods=return_periods,
            ema_periods=ema_periods,
            add_btc_features=add_btc_features
        )
        
        # Create an empty list to hold all sequence features DataFrames
        all_sequence_features = []
        
        # Get feature columns to create sequences for (exclude 'symbol' and 'timestamp')
        feature_cols = [col for col in features_df.columns if col not in ['symbol', 'timestamp']]
        
        # Process each symbol
        for symbol in features_df['symbol'].unique():
            # Filter for the current symbol
            symbol_df = features_df.filter(pl.col('symbol') == symbol)
            
            # Extract timestamp and create symbol series
            timestamp_series = symbol_df.select('timestamp')['timestamp']
            
            # Get column types before conversion to numpy
            column_types = {col: symbol_df[col].dtype for col in symbol_df.columns}
            
            # Convert to numpy arrays once for all operations
            numpy_df = symbol_df.to_numpy()
            columns = symbol_df.columns
            
            # Create a map of column names to indices for fast access
            col_idx_map = {col: idx for idx, col in enumerate(columns)}
            
            # Pre-allocate a dictionary to hold all sequences
            sequence_dict = {
                'timestamp': timestamp_series,
                'symbol': pl.Series([symbol] * len(timestamp_series))
            }
            
            # Process all features at once, directly with numpy
            for col in feature_cols:
                if col in col_idx_map:
                    # Extract column values as numpy array - zero copy operation
                    values = numpy_df[:, col_idx_map[col]]
                    
                    # Check if the column type is numeric before processing
                    col_type = column_types.get(col)
                    is_numeric = (
                        col_type is not None and 
                        (col_type.is_numeric() or 
                         # Handle special case for Polars Float64/etc
                         'float' in str(col_type).lower() or 
                         'int' in str(col_type).lower())
                    )
                    
                    if is_numeric:
                        # Try to safely convert to a numeric type if not already
                        try:
                            if not np.issubdtype(values.dtype, np.number):
                                values = values.astype(np.float64)
                                
                            # Handle NaN values
                            nan_mask = np.isnan(values)
                            if nan_mask.any():
                                # Replace NaNs with a sentinel value
                                masked_values = values.copy()
                                masked_values[nan_mask] = 0.0
                                
                                # Generate sequences using the standard function
                                sequences = create_sequences(masked_values, sequence_length)
                                
                                # Restore NaNs
                                for i in range(len(values)):
                                    if nan_mask[i]:
                                        sequences[i, -1] = np.nan
                            else:
                                # No NaNs, use values directly
                                sequences = create_sequences(values, sequence_length)
                        except (TypeError, ValueError) as e:
                            # If conversion fails, log warning and skip this column
                            logger.warning(f"Could not process column {col} as numeric: {e}")
                            continue
                    else:
                        # For non-numeric columns, we'll create sequences of None values
                        logger.warning(f"Skipping non-numeric column {col} for sequence generation")
                        # Create empty/NaN sequences for this column
                        sequences = np.zeros((len(symbol_df), sequence_length)) * np.nan
                    
                    # Add to dictionary - convert to list only once at the end
                    sequence_dict[col] = [row for row in sequences]
            
            # Create Polars DataFrame in a single operation
            seq_df = pl.DataFrame(sequence_dict)
            all_sequence_features.append(seq_df)
        
        # Concatenate all sequence DataFrames
        if all_sequence_features:
            result_df = pl.concat(all_sequence_features)
            return result_df
        else:
            # Return empty DataFrame
            return pl.DataFrame({'timestamp': [], 'symbol': []})
    
    @staticmethod
    def calculate_return_expr(period: int = 1) -> pl.Expr:
        """Calculate return over specified period in minutes as an expression."""
        return pl.col('close').pct_change(period)
    
    @staticmethod
    def calculate_ema_expr(period: int = 20) -> pl.Expr:
        """Calculate exponential moving average as an expression."""
        return pl.col('close').ewm_mean(span=period, adjust=False)
    
    @staticmethod
    def calculate_bollinger_bands_expr(period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands over specified period as expressions."""
        middle_band = pl.col('close').rolling_mean(window_size=period)
        std = pl.col('close').rolling_std(window_size=period)
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_true_range_expr() -> pl.Expr:
        """Calculate True Range as an expression."""
        return pl.max_horizontal(
            pl.col('high') - pl.col('low'),
            (pl.col('high') - pl.col('close').shift(1)).abs(),
            (pl.col('low') - pl.col('close').shift(1)).abs()
        )
    
    @staticmethod
    def calculate_open_close_ratio_expr() -> pl.Expr:
        """Calculate Open/Close ratio as an expression."""
        return pl.col('open') / pl.col('close')
    
    @staticmethod
    def calculate_rsi_expr(period: int = 14) -> pl.Expr:
        """
        Create an expression to calculate RSI.
        
        Returns a Polars expression that can be used in select() or with_columns()
        """
        delta = pl.col('close').diff()
        up = pl.when(delta > 0).then(delta).otherwise(0)
        down = pl.when(delta < 0).then(-delta).otherwise(0)
        
        avg_gain = up.rolling_mean(window_size=period)
        avg_loss = down.rolling_mean(window_size=period)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_autocorr_expr(window_size: int = 5) -> pl.Expr:
        """Calculate autocorrelation approximation as an expression."""
        return (pl.col('close').shift(1) * pl.col('close')).rolling_mean(window_size=window_size)
    
    @staticmethod
    def calculate_zscore_expr(column: str, window: int = 20) -> pl.Expr:
        """Calculate Z-score normalization over specified window as an expression."""
        return (
            (pl.col(column) - pl.col(column).rolling_mean(window_size=window)) /
            pl.col(column).rolling_std(window_size=window)
        )
    
    @staticmethod
    def calculate_minmax_scale_expr(column: str, window: int = 20) -> pl.Expr:
        """Calculate Min-Max scaling over specified window as an expression."""
        return (
            (pl.col(column) - pl.col(column).rolling_min(window_size=window)) /
            (pl.col(column).rolling_max(window_size=window) - pl.col(column).rolling_min(window_size=window))
        )
    
    @staticmethod
    def calculate_hl_range_pct_expr() -> pl.Expr:
        """Calculate high-low range as percentage of price as an expression."""
        return (pl.col('high') - pl.col('low')) / pl.col('close')
    
    @staticmethod
    def calculate_volume_ratio_expr(period: int = 20) -> pl.Expr:
        """Calculate volume ratio versus N-minute average as an expression."""
        return pl.col('volume') / pl.col('volume').rolling_mean(window_size=period)
    
    @staticmethod
    def calculate_obv(df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate On-Balance Volume.
        
        Returns a DataFrame with timestamp and obv columns
        """
        # This is hard to express purely in Polars expressions,
        # so we'll use a more explicit approach
        close_diff = df.select(pl.col('close').diff())
        
        # Convert to pandas for OBV calculation
        pd_df = df.select(['timestamp', 'volume', 'close']).to_pandas()
        pd_diff = close_diff.to_pandas()
        
        # Pre-allocate the array
        values = np.zeros(len(pd_df))
        values[0] = pd_df['volume'].iloc[0]
        
        # Process in one pass
        for i in range(1, len(pd_df)):
            if pd_diff['close'].iloc[i] > 0:
                values[i] = values[i-1] + pd_df['volume'].iloc[i]
            elif pd_diff['close'].iloc[i] < 0:
                values[i] = values[i-1] - pd_df['volume'].iloc[i]
            else:
                values[i] = values[i-1]
        
        # Create a Polars DataFrame with the OBV values
        return pl.DataFrame({
            'timestamp': df['timestamp'],
            'obv': values
        })

    def calculate_btc_features(self, timestamps: Union[pl.Series, List], return_periods: List[int]) -> pl.DataFrame:
        """
        Calculate BTC-specific features.
        
        Args:
            timestamps: Series or list of timestamps to align with
            return_periods: Periods in minutes for calculating returns
            
        Returns:
            DataFrame with BTC features
        """
        if self.btc_df is None:
            logger.warning("Bitcoin data not found in DataFrame. Cannot calculate BTC features. "
                         "Ensure your DataFrame includes a symbol containing 'BTC' (e.g., 'BTC-USDT-SWAP', 'BTCUSDT').")
            
            # Return empty DataFrame with timestamp column
            if isinstance(timestamps, pl.Series):
                return pl.DataFrame({'timestamp': timestamps})
            else:
                return pl.DataFrame({'timestamp': timestamps})
        
        # Create a base DataFrame with just the timestamp
        btc_df = self.btc_df.select(['timestamp'])
        
        # Add each BTC return as a column to this base DataFrame
        btc_return_exprs = []
        for period in return_periods:
            btc_return_exprs.append(
                pl.col('close').pct_change(period).alias(f'btc_return_{period}m')
            )
        
        # Add all BTC return columns at once
        if btc_return_exprs:
            btc_df = btc_df.join(
                self.btc_df.select(['timestamp'] + btc_return_exprs),
                on='timestamp', 
                how='left'
            )
        
        # Filter to the requested timestamps
        if isinstance(timestamps, pl.Series):
            timestamps_df = pl.DataFrame({'timestamp': timestamps})
        else:
            timestamps_df = pl.DataFrame({'timestamp': timestamps})
        
        # Join with the timestamps
        result_df = timestamps_df.join(btc_df, on='timestamp', how='left')
        
        return result_df


# Convenience functions
def create_features(df: Union[pl.DataFrame, pd.DataFrame], 
                    return_periods: List[int] = [1, 5, 15, 30, 60, 120],
                    ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
                    add_btc_features: bool = True) -> pl.DataFrame:
    """
    Convenience function to add all features to a market data DataFrame.
    Raw OHLCV data is excluded from the result.
    
    Args:
        df: DataFrame with timestamp column and OHLCV columns
        return_periods: Periods (in minutes) for calculating returns
        ema_periods: Periods (in minutes) for calculating EMAs
        add_btc_features: Whether to add BTC-related features
        
    Returns:
        DataFrame with normalized/relative features only
    """
    # Convert pandas DataFrame to Polars if needed
    if not isinstance(df, pl.DataFrame):
        logger.info("Converting pandas DataFrame to Polars")
        df = pl.from_pandas(df.reset_index())
    
    engineer = FeatureEngineer(df)
    return engineer.add_all_features(
        return_periods=return_periods,
        ema_periods=ema_periods,
        add_btc_features=add_btc_features
    )


def create_sequence_features(df: Union[pl.DataFrame, pd.DataFrame], 
                             sequence_length: int = 60,
                             return_periods: List[int] = [1, 5, 15, 30, 60, 120],
                             ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
                             add_btc_features: bool = True) -> pl.DataFrame:
    """
    Convenience function to create sequence features from market data.
    Each column in the resulting DataFrame contains arrays of the past values.
    Raw OHLCV data is excluded.
    
    Args:
        df: DataFrame with timestamp column and OHLCV columns
        sequence_length: Length of the sequence arrays
        return_periods: Periods (in minutes) for calculating returns
        ema_periods: Periods (in minutes) for calculating EMAs
        add_btc_features: Whether to add BTC-related features
        
    Returns:
        DataFrame with sequence features only
    """
    # Convert pandas DataFrame to Polars if needed
    if not isinstance(df, pl.DataFrame):
        logger.info("Converting pandas DataFrame to Polars")
        df = pl.from_pandas(df.reset_index())
    
    engineer = FeatureEngineer(df)
    return engineer.create_sequence_features(
        sequence_length=sequence_length,
        return_periods=return_periods,
        ema_periods=ema_periods,
        add_btc_features=add_btc_features
    )
