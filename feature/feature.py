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
            DataFrame with all features added
        """
        # Create an empty result DataFrame to hold all features
        all_features = []
        
        # Group by symbol to calculate per-symbol features
        for symbol, group in self.df.groupby('symbol'):
            # Calculate features
            symbol_features = group.copy()
            
            # Price indicators
            for period in return_periods:
                symbol_features[f'return_{period}m'] = self.calculate_return(symbol_features, period)
            
            for period in ema_periods:
                symbol_features[f'ema_{period}m'] = self.calculate_ema(symbol_features, period)
            
            # Bollinger Bands
            symbol_features['bb_upper'], symbol_features['bb_middle'], symbol_features['bb_lower'] = \
                self.calculate_bollinger_bands(symbol_features, 20, 2)
            
            # Other price indicators
            symbol_features['true_range'] = self.calculate_true_range(symbol_features)
            symbol_features['open_close_ratio'] = self.calculate_open_close_ratio(symbol_features)
            symbol_features['rsi'] = self.calculate_rsi(symbol_features, 14)
            symbol_features['autocorr_lag1'] = self.calculate_autocorrelation(symbol_features, 1)
            symbol_features['close_zscore'] = self.calculate_zscore(symbol_features['close'])
            symbol_features['close_minmax'] = self.calculate_minmax_scale(symbol_features['close'], 20)
            
            # Volume indicators
            symbol_features['volume_ratio_20m'] = self.calculate_volume_ratio(symbol_features, 20)
            symbol_features['obv'] = self.calculate_obv(symbol_features)
            
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
            return self.df.copy()
    
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
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        
    Returns:
        DataFrame with features added
    """
    engineer = FeatureEngineer(df)
    return engineer.add_all_features()
