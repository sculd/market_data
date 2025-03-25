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
            if add_btc_features and self.btc_df is not None and symbol != 'BTC':
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
            logger.warning("BTC data not found in DataFrame. Cannot calculate BTC features.")
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
    # Create a copy to defragment
    return engineer.add_all_features().copy()


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
    engineer = FeatureEngineer(df)
    # Create a copy to defragment
    return engineer.add_target_features(forward_periods, tp_values, sl_values).copy()


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
    # Create a copy to defragment
    return engineer.create_sequence_features(sequence_length=sequence_length).copy()
