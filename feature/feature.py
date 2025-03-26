import pandas as pd
import numpy as np
from typing import List, Union, Dict, Tuple, Literal
import logging
import warnings

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Class for engineering features from OHLCV market data.
    
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
        self.btc_df = None
        
        # Identify Bitcoin data by looking for common BTC patterns in symbol names
        # This handles different exchange formats: BTC, BTCUSDT, BTC-USDT-SWAP, BTC/USDT, etc.
        btc_symbols = [s for s in self.df['symbol'].unique() if 'BTC' in s.upper()]
        if btc_symbols:
            # Use the first matching BTC symbol found
            btc_symbol = btc_symbols[0]
            logger.info(f"Using {btc_symbol} as Bitcoin reference for BTC features")
            self.btc_df = self.df[self.df['symbol'] == btc_symbol].copy()
            # If multiple BTC symbols exist, log a warning
            if len(btc_symbols) > 1:
                logger.warning(f"Multiple Bitcoin symbols found: {btc_symbols}. Using {btc_symbol}")
    
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
        # Create an empty list to hold all features DataFrames
        all_features = []
        
        # Group by symbol to calculate per-symbol features
        for symbol, group in self.df.groupby('symbol'):
            # Create a dictionary to hold all feature columns
            feature_data = {'symbol': symbol}
            
            # Price indicators
            for period in return_periods:
                feature_data[f'return_{period}m'] = self.calculate_return(group, period)
            
            for period in ema_periods:
                ema = self.calculate_ema(group, period)
                feature_data[f'ema_{period}m'] = ema
                # Add EMA relative to price (normalized)
                feature_data[f'ema_rel_{period}m'] = group['close'] / ema
            
            # Bollinger Bands
            upper, middle, lower = self.calculate_bollinger_bands(group, 20, 2)
            # Add relative positions within bands (normalized)
            feature_data['bb_position'] = (group['close'] - lower) / (upper - lower)
            feature_data['bb_width'] = (upper - lower) / middle
            
            # Other price indicators
            feature_data['true_range'] = self.calculate_true_range(group)
            feature_data['open_close_ratio'] = self.calculate_open_close_ratio(group)
            feature_data['rsi'] = self.calculate_rsi(group, 14)
            feature_data['autocorr_lag1'] = self.calculate_autocorrelation(group, 1)
            feature_data['close_zscore'] = self.calculate_zscore(group['close'])
            feature_data['close_minmax'] = self.calculate_minmax_scale(group['close'], 20)
            
            # High-Low range as percentage of price
            feature_data['hl_range_pct'] = (group['high'] - group['low']) / group['close']
            
            # Volume indicators
            feature_data['volume_ratio_20m'] = self.calculate_volume_ratio(group, 20)
            feature_data['obv'] = self.calculate_obv(group)
            # Normalize OBV with a rolling window
            feature_data['obv_zscore'] = self.calculate_zscore(feature_data['obv'], 20)
            
            # Add BTC features if requested
            if add_btc_features and self.btc_df is not None and not ('BTC' in symbol.upper()):
                btc_features = self.calculate_btc_features(group.index, return_periods)
                for col, values in btc_features.items():
                    feature_data[col] = values
            
            # Create a single DataFrame from all features at once
            symbol_features = pd.DataFrame(feature_data, index=group.index)
            
            # Append to the list of features
            all_features.append(symbol_features)
        
        # Concatenate all symbol features into a single DataFrame
        if all_features:
            result_df = pd.concat(all_features)
            return result_df
        else:
            # Return an empty DataFrame with only the symbol column
            return pd.DataFrame({'symbol': []}, index=self.df.index)
    
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
        # First get the regular features (which already exclude raw OHLCV)
        features_df = self.add_all_features(
            return_periods=return_periods,
            ema_periods=ema_periods,
            add_btc_features=add_btc_features
        )
        
        # Create an empty list to hold all sequence features DataFrames
        all_sequence_features = []
        
        # Group by symbol to calculate per-symbol sequences
        for symbol, group in features_df.groupby('symbol'):
            # Get feature columns to create sequences for (exclude 'symbol')
            feature_cols = [col for col in group.columns if col != 'symbol']
            
            # Pre-compute all sequences at once for better performance
            sequences = {}
            for col in feature_cols:
                # Create a list of arrays for this column
                col_sequences = []
                values = group[col].values
                
                for i in range(len(group)):
                    if i < sequence_length - 1:
                        # Not enough history yet, use available data and pad with NaNs
                        seq = values[max(0, i-sequence_length+1):i+1]
                        padding = np.full(sequence_length - len(seq), np.nan)
                        col_sequences.append(np.concatenate([padding, seq]))
                    else:
                        # Full history available
                        col_sequences.append(values[i-sequence_length+1:i+1])
                
                sequences[col] = col_sequences
            
            # Add symbol column
            sequences['symbol'] = symbol
            
            # Create DataFrame all at once
            symbol_sequences = pd.DataFrame(sequences, index=group.index)
            
            # Convert the list columns to object type containing numpy arrays
            for col in feature_cols:
                symbol_sequences[col] = symbol_sequences[col].astype('object')
            
            # Append to the list of sequence features
            all_sequence_features.append(symbol_sequences)
        
        # Concatenate all symbol sequence features into a single DataFrame
        if all_sequence_features:
            result_df = pd.concat(all_sequence_features)
            return result_df
        else:
            # Return empty DataFrame with same index
            return pd.DataFrame(index=self.df.index)
    
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
        # Let division by zero result in NaN
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
        
        # Let division by zero result in NaN (which will propagate through later calculations)
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
        
        # Let division by zero result in NaN
        return (series - mean) / std
    
    @staticmethod
    def calculate_minmax_scale(series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Min-Max scaling over specified window in minutes."""
        roll_min = series.rolling(window=window).min()
        roll_max = series.rolling(window=window).max()
        
        # Let division by zero result in NaN
        return (series - roll_min) / (roll_max - roll_min)
    
    @staticmethod
    def calculate_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate volume ratio versus N-minute average."""
        avg_volume = df['volume'].rolling(window=period).mean()
        
        # Let division by zero result in NaN
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
            logger.warning("Bitcoin data not found in DataFrame. Cannot calculate BTC features. "
                         "Ensure your DataFrame includes a symbol containing 'BTC' (e.g., 'BTC-USDT-SWAP', 'BTCUSDT').")
            return pd.DataFrame(index=index)
        
        # Create a dictionary to hold all BTC features
        btc_data = {}
        
        for period in return_periods:
            # Calculate BTC return
            btc_return = self.calculate_return(self.btc_df, period)
            
            # Reindex to match the target symbol's index
            btc_return = btc_return.reindex(index)
            
            # Add as a feature
            btc_data[f'btc_return_{period}m'] = btc_return
        
        # Create a single DataFrame from all features at once
        return pd.DataFrame(btc_data, index=index)


# Example usage
def create_features(df: pd.DataFrame, 
                    return_periods: List[int] = [1, 5, 15, 30, 60, 120],
                    ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
                    add_btc_features: bool = True) -> pd.DataFrame:
    """
    Convenience function to add all features to a market data DataFrame.
    Raw OHLCV data is excluded from the result.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        return_periods: Periods (in minutes) for calculating returns
        ema_periods: Periods (in minutes) for calculating EMAs
        add_btc_features: Whether to add BTC-related features
        
    Returns:
        DataFrame with normalized/relative features only
    """
    engineer = FeatureEngineer(df)
    # Create a copy to defragment
    return engineer.add_all_features(
        return_periods=return_periods,
        ema_periods=ema_periods,
        add_btc_features=add_btc_features
    ).copy()


def create_sequence_features(df: pd.DataFrame, 
                             sequence_length: int = 60,
                             return_periods: List[int] = [1, 5, 15, 30, 60, 120],
                             ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
                             add_btc_features: bool = True) -> pd.DataFrame:
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
        
    Returns:
        DataFrame with sequence features only
    """
    engineer = FeatureEngineer(df)
    # Create a copy to defragment
    return engineer.create_sequence_features(
        sequence_length=sequence_length,
        return_periods=return_periods,
        ema_periods=ema_periods,
        add_btc_features=add_btc_features
    ).copy()
