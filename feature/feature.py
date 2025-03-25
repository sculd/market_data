import pandas as pd
import numpy as np
from typing import List, Union, Dict
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
        if 'BTC' in self.df['symbol'].unique():
            self.btc_df = self.df[self.df['symbol'] == 'BTC'].copy()
    
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
        # Create an empty result DataFrame to hold all features
        all_features = []
        
        # Group by symbol to calculate per-symbol features
        for symbol, group in self.df.groupby('symbol'):
            # Initialize a new DataFrame for features, only keeping index and symbol
            symbol_features = pd.DataFrame(index=group.index)
            symbol_features['symbol'] = symbol
            
            # Price indicators
            for period in return_periods:
                symbol_features[f'return_{period}m'] = self.calculate_return(group, period)
            
            for period in ema_periods:
                symbol_features[f'ema_{period}m'] = self.calculate_ema(group, period)
                # Add EMA relative to price (normalized)
                symbol_features[f'ema_rel_{period}m'] = group['close'] / self.calculate_ema(group, period)
            
            # Bollinger Bands
            upper, middle, lower = self.calculate_bollinger_bands(group, 20, 2)
            # Add relative positions within bands (normalized)
            symbol_features['bb_position'] = (group['close'] - lower) / (upper - lower)
            symbol_features['bb_width'] = (upper - lower) / middle
            
            # Other price indicators
            symbol_features['true_range'] = self.calculate_true_range(group)
            symbol_features['open_close_ratio'] = self.calculate_open_close_ratio(group)
            symbol_features['rsi'] = self.calculate_rsi(group, 14)
            symbol_features['autocorr_lag1'] = self.calculate_autocorrelation(group, 1)
            symbol_features['close_zscore'] = self.calculate_zscore(group['close'])
            symbol_features['close_minmax'] = self.calculate_minmax_scale(group['close'], 20)
            
            # High-Low range as percentage of price
            symbol_features['hl_range_pct'] = (group['high'] - group['low']) / group['close']
            
            # Volume indicators
            symbol_features['volume_ratio_20m'] = self.calculate_volume_ratio(group, 20)
            symbol_features['obv'] = self.calculate_obv(group)
            # Normalize OBV with a rolling window
            symbol_features['obv_zscore'] = self.calculate_zscore(symbol_features['obv'], 20)
            
            # Add BTC features if requested
            if add_btc_features and self.btc_df is not None and symbol != 'BTC':
                btc_features = self.calculate_btc_features(symbol_features.index, return_periods)
                for col in btc_features.columns:
                    symbol_features[col] = btc_features[col].values
            
            # Append to the list of features
            all_features.append(symbol_features)
        
        # Concatenate all symbol features into a single DataFrame
        if all_features:
            result_df = pd.concat(all_features)
            return result_df
        else:
            # Return an empty DataFrame with only the symbol column
            return pd.DataFrame({'symbol': []}, index=self.df.index)
    
    def create_sequence_features(self, sequence_length: int = 60) -> pd.DataFrame:
        """
        Create a DataFrame with sequence features only.
        Each column contains arrays of past values for each feature.
        Raw OHLCV columns are excluded.
        
        Args:
            sequence_length: Length of the sequence arrays
            
        Returns:
            DataFrame with sequence features only
        """
        # First get the regular features (which already exclude raw OHLCV)
        features_df = self.add_all_features()
        
        # Create an empty result DataFrame to hold all sequence features
        all_sequence_features = []
        
        # Group by symbol to calculate per-symbol sequences
        for symbol, group in features_df.groupby('symbol'):
            # Create a DataFrame to hold sequences for this symbol
            symbol_sequences = pd.DataFrame(index=group.index)
            symbol_sequences['symbol'] = symbol
            
            # Get feature columns to create sequences for (exclude 'symbol')
            feature_cols = [col for col in group.columns if col != 'symbol']
            
            # Add sequence columns
            for col in feature_cols:
                # Initialize the column with object type to hold arrays
                symbol_sequences[col] = None
                symbol_sequences[col] = symbol_sequences[col].astype('object')
                
                # For each row, create array of past values
                for i in range(len(group)):
                    if i < sequence_length - 1:
                        # Not enough history yet, use available data and pad with NaNs
                        seq = np.array(group[col].iloc[max(0, i-sequence_length+1):i+1])
                        padding = np.full(sequence_length - len(seq), np.nan)
                        symbol_sequences.at[group.index[i], col] = np.concatenate([padding, seq])
                    else:
                        # Full history available
                        symbol_sequences.at[group.index[i], col] = np.array(group[col].iloc[i-sequence_length+1:i+1])
            
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
        obv = pd.Series(index=df.index)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if close_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif close_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
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
            logger.warning("BTC data not found in DataFrame. Cannot calculate BTC features.")
            return pd.DataFrame(index=index)
        
        # Calculate BTC returns for each period
        btc_features = pd.DataFrame(index=index)
        
        for period in return_periods:
            # Calculate BTC return
            btc_return = self.calculate_return(self.btc_df, period)
            
            # Reindex to match the target symbol's index
            btc_return = btc_return.reindex(index)
            
            # Add as a feature
            btc_features[f'btc_return_{period}m'] = btc_return
        
        return btc_features


# Example usage
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all features to a market data DataFrame.
    Raw OHLCV data is excluded from the result.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        
    Returns:
        DataFrame with normalized/relative features only
    """
    engineer = FeatureEngineer(df)
    return engineer.add_all_features()


def create_sequence_features(df: pd.DataFrame, sequence_length: int = 60) -> pd.DataFrame:
    """
    Convenience function to create sequence features from market data.
    Each column in the resulting DataFrame contains arrays of the past values.
    Raw OHLCV data is excluded.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        sequence_length: Length of the sequence arrays
        
    Returns:
        DataFrame with sequence features only
    """
    engineer = FeatureEngineer(df)
    return engineer.create_sequence_features(sequence_length=sequence_length)
