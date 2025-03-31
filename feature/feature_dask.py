import pandas as pd
import dask.dataframe as dd
from dask.delayed import delayed
import numpy as np
from typing import List, Union, Dict, Tuple, Literal
import logging
import warnings
from multiprocessing.pool import ThreadPool
import time

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Class for engineering features from OHLCV market data, using efficient Dask-based processing.
    
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
        
        # Get BTC data for features
        self.btc_df = self._extract_btc_data()
    
    def _extract_btc_data(self):
        """Extract Bitcoin data for reference features."""
        # Identify Bitcoin data 
        btc_symbols = [s for s in self.df['symbol'].unique().compute() if 'BTC' in s.upper()]
        
        if not btc_symbols:
            logger.warning("No Bitcoin symbol found in the data")
            return None
        
        # Use the first matching BTC symbol found
        btc_symbol = btc_symbols[0]
        logger.info(f"Using {btc_symbol} as Bitcoin reference for BTC features")
        
        # Extract BTC data
        btc_df = self.df[self.df['symbol'] == btc_symbol].compute()
        
        # If multiple BTC symbols exist, log a warning
        if len(btc_symbols) > 1:
            logger.warning(f"Multiple Bitcoin symbols found: {btc_symbols}. Using {btc_symbol}")
        
        return btc_df
    
    def validate_dataframe(self, df: dd.DataFrame) -> None:
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
    
    def add_all_features(self, return_periods: List[int] = [1, 5, 15, 30, 60, 120], 
                         ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
                         add_btc_features: bool = True) -> pd.DataFrame:
        """
        Add all features to the DataFrame.
        
        Args:
            return_periods: Periods (in minutes) for calculating returns
            ema_periods: Periods (in minutes) for calculating EMAs
            add_btc_features: Whether to add BTC-related features
            
        Returns:
            DataFrame with all features added (excluding raw OHLCV data)
        """
        start_time = time.time()
        logger.info("Starting feature engineering using Dask implementation")
        
        # Use Dask implementation
        result_df = self._add_features_dask(return_periods, ema_periods, add_btc_features)
        
        elapsed = time.time() - start_time
        logger.info(f"Feature engineering completed in {elapsed:.2f} seconds")
        return result_df
    
    def _add_features_dask(self, return_periods, ema_periods, add_btc_features):
        """Dask implementation of feature engineering."""
        # Define function to process each partition
        def process_symbol_partition(partition):
            if len(partition) == 0:
                return pd.DataFrame()
            
            symbol = partition['symbol'].iloc[0]
            
            # Create feature data dictionary with the right column order
            feature_data = {'symbol': symbol}
            
            # Price indicators - returns
            for period in return_periods:
                feature_data[f'return_{period}m'] = self.calculate_return(partition, period)
            
            # EMAs - first all periods
            ema_values = {}
            for period in ema_periods:
                ema = self.calculate_ema(partition, period)
                feature_data[f'ema_{period}m'] = ema
                ema_values[period] = ema
            
            # Then all ema_rel values
            for period in ema_periods:
                feature_data[f'ema_rel_{period}m'] = partition['close'] / ema_values[period]
            
            # Bollinger Bands
            upper, middle, lower = self.calculate_bollinger_bands(partition, 20, 2)
            feature_data['bb_position'] = (partition['close'] - lower) / (upper - lower)
            feature_data['bb_width'] = (upper - lower) / middle
            
            # Other indicators
            feature_data['true_range'] = self.calculate_true_range(partition)
            feature_data['open_close_ratio'] = self.calculate_open_close_ratio(partition)
            feature_data['rsi'] = self.calculate_rsi(partition, 14)
            feature_data['autocorr_lag1'] = self.calculate_autocorrelation(partition, 1)
            feature_data['close_zscore'] = self.calculate_zscore(partition['close'])
            feature_data['close_minmax'] = self.calculate_minmax_scale(partition['close'], 20)
            feature_data['hl_range_pct'] = (partition['high'] - partition['low']) / partition['close']
            feature_data['volume_ratio_20m'] = self.calculate_volume_ratio(partition, 20)
            feature_data['obv'] = self.calculate_obv(partition)
            feature_data['obv_zscore'] = self.calculate_zscore(feature_data['obv'], 20)
            
            # BTC features
            if add_btc_features and self.btc_df is not None and not ('BTC' in symbol.upper()):
                btc_features = self.calculate_btc_features(partition.index, return_periods)
                for col, values in btc_features.items():
                    feature_data[col] = values
            
            return pd.DataFrame(feature_data, index=partition.index)
        
        # Define the expected column order for the metadata
        meta_columns = ['symbol']
        meta_columns.extend([f'return_{p}m' for p in return_periods])
        meta_columns.extend([f'ema_{p}m' for p in ema_periods])
        meta_columns.extend([f'ema_rel_{p}m' for p in ema_periods])
        meta_columns.extend([
            'bb_position', 'bb_width', 'true_range', 'open_close_ratio', 'rsi',
            'autocorr_lag1', 'close_zscore', 'close_minmax', 'hl_range_pct',
            'volume_ratio_20m', 'obv', 'obv_zscore'
        ])
        if add_btc_features:
            meta_columns.extend([f'btc_return_{p}m' for p in return_periods])
        
        # Create metadata dictionary
        meta = {col: 'float64' if col != 'symbol' else 'object' for col in meta_columns}
        
        # Apply function to each partition
        logger.info(f"Processing features using Dask with {self.df.npartitions} partitions")
        result_dask = self.df.map_partitions(process_symbol_partition, meta=meta)
        
        # Compute the result
        logger.info("Computing Dask result using processes scheduler")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_df = result_dask.compute(scheduler="processes")
        
        return result_df
    
    def create_sequence_features(self, sequence_length: int = 60,
                               return_periods: List[int] = [1, 5, 15, 30, 60, 120], 
                               ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
                               add_btc_features: bool = True) -> pd.DataFrame:
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
        # Get regular features first
        features_df = self.add_all_features(
            return_periods=return_periods,
            ema_periods=ema_periods,
            add_btc_features=add_btc_features
        )
        
        logger.info(f"Creating sequence features with length {sequence_length}")
        all_sequence_features = []
        
        # Group by symbol
        for symbol, group in features_df.groupby('symbol'):
            # Get feature columns (exclude 'symbol')
            feature_cols = [col for col in group.columns if col != 'symbol']
            
            # Pre-allocate sequences dictionary
            sequences = {}
            
            # For each feature column, create sequence arrays
            for col in feature_cols:
                col_sequences = []
                values = group[col].values
                
                for i in range(len(group)):
                    if i < sequence_length - 1:
                        # Not enough history, pad with NaN
                        seq = values[max(0, i-sequence_length+1):i+1]
                        padding = np.full(sequence_length - len(seq), np.nan)
                        col_sequences.append(np.concatenate([padding, seq]))
                    else:
                        # Full history available
                        col_sequences.append(values[i-sequence_length+1:i+1])
                
                sequences[col] = col_sequences
            
            # Add symbol column
            sequences['symbol'] = symbol
            
            # Create DataFrame
            symbol_sequences = pd.DataFrame(sequences, index=group.index)
            all_sequence_features.append(symbol_sequences)
        
        # Concatenate all sequence features
        if all_sequence_features:
            result_df = pd.concat(all_sequence_features)
            return result_df
        else:
            return pd.DataFrame(index=features_df.index)
    
    @staticmethod
    def calculate_return(df: pd.DataFrame, period: int = 1) -> pd.Series:
        """Calculate return over specified period in minutes."""
        return df['close'].pct_change(period)
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate exponential moving average over specified period in minutes."""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands over specified period in minutes."""
        middle_band = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_true_range(df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        
        ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
        true_range = ranges.max(axis=1)
        return true_range
    
    @staticmethod
    def calculate_open_close_ratio(df: pd.DataFrame) -> pd.Series:
        """Calculate Open/Close ratio."""
        return df['open'] / df['close']
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index over specified period in minutes."""
        delta = df['close'].diff()
        
        up = delta.copy()
        up[up < 0] = 0
        
        down = -delta.copy()
        down[down < 0] = 0
        
        avg_gain = up.rolling(window=period).mean()
        avg_loss = down.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi, index=df.index)
    
    @staticmethod
    def calculate_autocorrelation(df: pd.DataFrame, lag: int = 1) -> pd.Series:
        """Calculate autocorrelation with specified lag in minutes."""
        # Fill initial NaN with 0
        result = df['close'].autocorr(lag=lag)
        # For a Series, autocorr returns a single value, so create a Series of that value
        if isinstance(result, (int, float)):
            result = pd.Series(np.nan, index=df.index)
            # Calculate rolling autocorrelation for a more meaningful result
            for i in range(lag + 14, len(df)):
                window = df['close'].iloc[i-14:i+1]
                result.iloc[i] = window.autocorr(lag=lag)
        return result

    @staticmethod
    def calculate_zscore(series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Z-score normalization over specified window in minutes."""
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return (series - mean) / std
    
    @staticmethod
    def calculate_minmax_scale(series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Min-Max scaling over specified window in minutes."""
        roll_min = series.rolling(window=window).min()
        roll_max = series.rolling(window=window).max()
        return (series - roll_min) / (roll_max - roll_min)
    
    @staticmethod
    def calculate_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate volume ratio versus N-minute average."""
        avg_volume = df['volume'].rolling(window=period).mean()
        return df['volume'] / avg_volume
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        close_diff = df['close'].diff()
        obv = pd.Series(0, index=df.index)
        
        # Using a vectorized approach where possible
        # This is a cumulative sum with conditional adjustments
        # which is hard to fully vectorize, so we'll use a more efficient loop approach
        
        # Pre-allocate the array
        values = np.zeros(len(df))
        values[0] = df['volume'].iloc[0]
        
        # Process in one pass
        for i in range(1, len(df)):
            if close_diff.iloc[i] > 0:
                values[i] = values[i-1] + df['volume'].iloc[i]
            elif close_diff.iloc[i] < 0:
                values[i] = values[i-1] - df['volume'].iloc[i]
            else:
                values[i] = values[i-1]
        
        # Assign values to Series all at once
        obv = pd.Series(values, index=df.index)
        return obv

    def calculate_btc_features(self, index: pd.DatetimeIndex, return_periods: List[int]) -> pd.DataFrame:
        """
        Calculate BTC-specific features.
        
        Args:
            index: DatetimeIndex to align with
            return_periods: Periods in minutes for calculating returns
            
        Returns:
            DataFrame with BTC features
        """
        if self.btc_df is None:
            logger.warning("Bitcoin data not found. Cannot calculate BTC features.")
            return pd.DataFrame(index=index)
        
        btc_data = {}
        
        for period in return_periods:
            btc_return = self.calculate_return(self.btc_df, period)
            btc_return = btc_return.reindex(index)
            btc_data[f'btc_return_{period}m'] = btc_return
        
        return pd.DataFrame(btc_data, index=index)


# Convenience functions
def create_features(df: Union[pd.DataFrame, dd.DataFrame], 
                    return_periods: List[int] = [1, 5, 15, 30, 60, 120],
                    ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
                    add_btc_features: bool = True,
                    n_workers: int = None) -> pd.DataFrame:
    """
    Convenience function to add all features to a market data DataFrame.
    Raw OHLCV data is excluded from the result.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        return_periods: Periods (in minutes) for calculating returns
        ema_periods: Periods (in minutes) for calculating EMAs
        add_btc_features: Whether to add BTC-related features
        n_workers: Number of worker threads/processes to use (default: auto)
        
    Returns:
        DataFrame with normalized/relative features only
    """
    engineer = FeatureEngineer(df)
    return engineer.add_all_features(
        return_periods=return_periods,
        ema_periods=ema_periods,
        add_btc_features=add_btc_features
    )


def create_sequence_features(df: Union[pd.DataFrame, dd.DataFrame], 
                             sequence_length: int = 60,
                             return_periods: List[int] = [1, 5, 15, 30, 60, 120],
                             ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
                             add_btc_features: bool = True,
                             n_workers: int = None) -> pd.DataFrame:
    """
    Convenience function to create sequence features from market data.
    Each column in the resulting DataFrame contains arrays of the past values.
    Raw OHLCV data is excluded.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        sequence_length: Length of the sequence arrays
        return_periods: Periods (in minutes) for calculating returns
        ema_periods: Periods (in minutes) for calculating EMAs
        add_btc_features: Whether to add BTC-related features
        n_workers: Number of worker threads/processes to use (default: auto)
        
    Returns:
        DataFrame with sequence features only
    """
    engineer = FeatureEngineer(df)
    return engineer.create_sequence_features(
        sequence_length=sequence_length,
        return_periods=return_periods,
        ema_periods=ema_periods,
        add_btc_features=add_btc_features
    )
