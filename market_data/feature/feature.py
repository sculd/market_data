import pandas as pd
import numpy as np
from typing import List, Union, Dict, Tuple, Literal
import logging
import warnings
import numba as nb
import os
from dataclasses import dataclass, field
logger = logging.getLogger(__name__)

# Default values for feature parameters
DEFAULT_RETURN_PERIODS = [1, 5, 15, 30, 60, 120]
DEFAULT_EMA_PERIODS = [5, 15, 30, 60, 120, 240]

# Default values for feature parameters
DEFAULT_RETURN_PERIODS = [1, 5, 15, 30, 60, 120]
DEFAULT_EMA_PERIODS = [5, 15, 30, 60, 120, 240]

@dataclass
class FeatureParams:
    """
    Encapsulates parameters for feature engineering.
    
    This class holds all the parameters needed for feature calculation,
    providing a single source of truth for feature configuration.
    
    Attributes:
        return_periods (List[int]): Periods (in minutes) for calculating returns
        ema_periods (List[int]): Periods (in minutes) for calculating EMAs
        add_btc_features (bool): Whether to add BTC-related features
        volatility_windows (List[int]): Window sizes in minutes for volatility regime calculation
    """
    return_periods: List[int] = field(default_factory=lambda: DEFAULT_RETURN_PERIODS)
    ema_periods: List[int] = field(default_factory=lambda: DEFAULT_EMA_PERIODS)
    add_btc_features: bool = True
    volatility_windows: List[int] = field(default_factory=lambda: [240])  # 1d (1w, 1m)
    
    def __repr__(self) -> str:
        """String representation of the parameters."""
        return (
            f"FeatureParams(return_periods={self.return_periods}, "
            f"ema_periods={self.ema_periods}, "
            f"add_btc_features={self.add_btc_features}, "
            f"volatility_windows={self.volatility_windows})"
        )
    
    def get_params_dir(self) -> str:
        """
        Generate a directory name string from feature parameters.
        
        Returns:
            str: Directory name string suitable for caching
        """
        from market_data.util.cache.path import params_to_dir_name
        
        params_dict = {
            'rp': self.return_periods,
            'ep': self.ema_periods,
            'btc': self.add_btc_features
        }
        return params_to_dir_name(params_dict)
        
    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on feature parameters.
        
        Uses the maximum window size from return_periods, ema_periods, and volatility_windows,
        plus a buffer to ensure sufficient historical data for feature calculations.
        
        Returns:
            int: Recommended number of warm-up days
        """
        import math
        
        # Find the maximum window period from all feature types
        max_window = max(
            max(self.return_periods) if self.return_periods else 0,
            max(self.ema_periods) if self.ema_periods else 0,
            max(self.volatility_windows) if self.volatility_windows else 0
        )
        
        # Convert to days (assuming periods are in minutes for 24/7 markets)
        days_needed = math.ceil(max_window / (24 * 60))
        
        return days_needed

# Numba-accelerated functions for performance-critical calculations
@nb.njit(cache=True)
def _calculate_obv_numba(prices, volumes):
    """Numba-accelerated On-Balance Volume calculation."""
    n = len(prices)
    obv = np.zeros(n)
    obv[0] = volumes[0]
    
    for i in range(1, n):
        if prices[i] > prices[i-1]:
            obv[i] = obv[i-1] + volumes[i]
        elif prices[i] < prices[i-1]:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]
    
    return obv

@nb.njit(cache=True)
def _calculate_rolling_corr_numba(x, y, window_size):
    """Numba-accelerated rolling correlation calculation."""
    n = len(x)
    result = np.full(n, np.nan)
    
    for i in range(window_size, n):
        x_window = x[i-window_size:i]
        y_window = y[i-window_size:i]
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))
        if np.sum(valid_mask) < 5:  # Not enough valid points
            continue
            
        x_valid = x_window[valid_mask]
        y_valid = y_window[valid_mask]
        
        if len(x_valid) < 2:
            continue
            
        # Calculate means
        x_mean = np.mean(x_valid)
        y_mean = np.mean(y_valid)
        
        # Calculate standard deviations
        x_std = np.std(x_valid)
        y_std = np.std(y_valid)
        
        if x_std == 0 or y_std == 0:
            continue
            
        # Calculate correlation
        cov = np.mean((x_valid - x_mean) * (y_valid - y_mean))
        corr = cov / (x_std * y_std)
        
        result[i] = corr
        
    return result

@nb.njit(cache=True)
def _calculate_rsi_numba(closes, period):
    """Numba-accelerated RSI calculation."""
    n = len(closes)
    deltas = np.zeros(n)
    ups = np.zeros(n)
    downs = np.zeros(n)
    rsi = np.full(n, np.nan)
    
    # Calculate deltas
    for i in range(1, n):
        deltas[i] = closes[i] - closes[i-1]
    
    # Calculate up and down moves
    for i in range(1, n):
        if deltas[i] > 0:
            ups[i] = deltas[i]
            downs[i] = 0
        else:
            ups[i] = 0
            downs[i] = -deltas[i]
    
    # Calculate RSI
    for i in range(period, n):
        avg_gain = 0.0
        avg_loss = 0.0
        
        # Calculate average gain and loss
        for j in range(i-period+1, i+1):
            avg_gain += ups[j]
            avg_loss += downs[j]
        
        avg_gain /= period
        avg_loss /= period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

@nb.njit(cache=True)
def _calculate_bollinger_bands_numba(closes, period, std_dev):
    """Numba-accelerated Bollinger Bands calculation."""
    n = len(closes)
    middle_band = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(period-1, n):
        window = closes[i-period+1:i+1]
        valid_window = window[~np.isnan(window)]
        
        if len(valid_window) < period // 2:
            continue
            
        mean = np.mean(valid_window)
        std = np.std(valid_window)
        
        middle_band[i] = mean
        upper_band[i] = mean + (std * std_dev)
        lower_band[i] = mean - (std * std_dev)
    
    return upper_band, middle_band, lower_band

@nb.njit(cache=True, parallel=False, fastmath=True)
def _calculate_garch_volatility(returns, omega=0.1, alpha=0.1, beta=0.8):
    """Numba-accelerated GARCH(1,1) volatility calculation."""
    n = len(returns)
    sigma2 = np.zeros(n)
    
    # Initialize with variance of non-NaN returns
    valid_mask = ~np.isnan(returns)
    if np.any(valid_mask):
        sigma2[0] = np.nanvar(returns[valid_mask])
    else:
        sigma2[0] = omega  # Default value if all returns are NaN
    
    # Pre-calculate squared returns for performance
    squared_returns = np.where(np.isnan(returns), 0.0, returns * returns)
    
    # Main GARCH loop with optimized operations
    for i in range(1, n):
        if np.isnan(returns[i-1]):
            sigma2[i] = sigma2[i-1]
        else:
            sigma2[i] = omega + alpha * squared_returns[i-1] + beta * sigma2[i-1]
    
    return np.sqrt(sigma2)

@nb.njit(cache=True)
def _calculate_rolling_volatility_regime(volatility: np.ndarray, window: int = 1440) -> np.ndarray:
    """
    Calculate rolling volatility regime using a moving window.
    
    Args:
        volatility: Array of volatility values
        window: Rolling window size in minutes (default: 1440 = 1 day)
    
    Returns:
        Array of regime classifications:
        -1: Low volatility (stable/bull)
        0: Medium volatility (transition)
        1: High volatility (bear)
    """
    n = len(volatility)
    regime = np.zeros(n)
    
    for i in range(window, n):
        # Get the window of volatility values
        window_vol = volatility[i-window:i]
        
        # Calculate percentiles for this window
        low_threshold = np.percentile(window_vol, 33)
        high_threshold = np.percentile(window_vol, 66)
        
        # Classify current point
        if volatility[i] > high_threshold:
            regime[i] = 1  # High volatility
        elif volatility[i] < low_threshold:
            regime[i] = -1  # Low volatility
    
    # Fill initial window with zeros
    regime[:window] = 0
    
    return regime

@nb.njit(cache=True)
def _calculate_rolling_zscore(volatility: np.ndarray, window: int = 1440) -> np.ndarray:
    """
    Calculate rolling z-score of volatility using a moving window.
    
    Args:
        volatility: Array of volatility values
        window: Rolling window size in minutes (default: 1440 = 1 day)
    
    Returns:
        Array of z-scores capped to the range [-10, 10]
    """
    n = len(volatility)
    zscore = np.zeros(n)
    
    # Add a minimum threshold for standard deviation to avoid division by near-zero
    min_std = 1e-8
    
    for i in range(window, n):
        # Get the window of volatility values
        window_vol = volatility[i-window:i]
        
        # Calculate mean and std for this window
        mean = np.mean(window_vol)
        std = np.std(window_vol)
        
        # Calculate z-score with minimum std threshold
        if std > 0:
            # Use max(std, min_std) to avoid division by very small numbers
            effective_std = max(std, min_std)
            z = (volatility[i] - mean) / effective_std
            
            # Cap z-score to the range [-10, 10]
            if z > 10:
                z = 10
            elif z < -10:
                z = -10
                
            zscore[i] = z
        else:
            zscore[i] = 0
    
    # Fill initial window with zeros
    zscore[:window] = 0
    
    return zscore

class FeatureEngineer:
    """
    Class for engineering features from OHLCV market data.
    
    The input DataFrame should have:
    - 'timestamp' as index (with minute-level data)
    - Columns: 'symbol', 'open', 'high', 'low', 'close', 'volume'
    """
    
    def __init__(self, df: pd.DataFrame, params: FeatureParams = None):
        """
        Initialize with a DataFrame of market data.
        
        Args:
            df: DataFrame with timestamp index and OHLCV columns
            params: Feature parameters. If None, uses default parameters
        """
        self.validate_dataframe(df)
        self.df = df.copy()
        self.btc_df = None
        self.params = params or FeatureParams()
        
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
    
    def add_all_features(self) -> pd.DataFrame:
        """
        Add all features to the DataFrame.
        
        Returns:
            DataFrame with all features added (excluding raw OHLCV data)
        """
        # Create an empty list to hold all features DataFrames
        all_features = []
        
        # Group by symbol to calculate per-symbol features
        for symbol, group in self.df.groupby('symbol'):
            # Create a dictionary to hold all feature columns
            feature_data = {'symbol': symbol}
            
            # Calculate period 1 returns (needed for GARCH)
            returns_1 = self.calculate_return(group, period=1)
            i = 0
            while i < len(returns_1) and np.isnan(returns_1.iloc[i]):
                returns_1.iloc[i] = 0
                i += 1
            
            # Calculate GARCH volatility
            volatility = _calculate_garch_volatility(returns_1.values)
            feature_data['garch_volatility'] = volatility
            
            # Calculate rolling regime and zscore for different time horizons
            for window in self.params.volatility_windows:
                feature_data[f'volatility_regime_{window}'] = _calculate_rolling_volatility_regime(volatility, window=window)
                feature_data[f'volatility_zscore_{window}'] = _calculate_rolling_zscore(volatility, window=window)
            
            # Price indicators
            for period in self.params.return_periods:
                if period != 1:  # Skip if we already calculated it
                    feature_data[f'return_{period}m'] = self.calculate_return(group, period)
            
            for period in self.params.ema_periods:
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
            # Note: true_range feature will be reimplemented later
            # feature_data['true_range'] = self.calculate_true_range(group)
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
            if self.params.add_btc_features and self.btc_df is not None and not ('BTC' in symbol.upper()):
                btc_features = self.calculate_btc_features(group.index)
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
        # Use Numba-accelerated version
        closes = df['close'].values
        upper, middle, lower = _calculate_bollinger_bands_numba(closes, period, std_dev)
        return pd.Series(upper, index=df.index), pd.Series(middle, index=df.index), pd.Series(lower, index=df.index)
    
    # Note: true_range calculation will be reimplemented later with a focus on
    # calculating the range of values relative to the latest value for a fixed time window
    
    @staticmethod
    def calculate_open_close_ratio(df: pd.DataFrame) -> pd.Series:
        """Calculate Open/Close ratio."""
        # Let division by zero result in NaN
        return df['open'] / df['close']
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index over specified period in minutes."""
        # Use Numba-accelerated version
        closes = df['close'].values
        rsi_values = _calculate_rsi_numba(closes, period)
        return pd.Series(rsi_values, index=df.index)
    
    @staticmethod
    def calculate_autocorrelation(df: pd.DataFrame, lag: int = 1) -> pd.Series:
        """Calculate autocorrelation with specified lag in minutes."""
        # Use Numba-accelerated version for rolling correlation
        closes = df['close'].values
        shifted = np.roll(closes, lag)
        window_size = min(14, len(closes) // 2)
        
        # Avoid correlation with rolled-around values
        shifted[:lag] = np.nan
        
        corr_values = _calculate_rolling_corr_numba(closes, shifted, window_size)
        return pd.Series(corr_values, index=df.index)
    
    @staticmethod
    def calculate_zscore(series: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Z-score normalization over specified window in minutes."""
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        
        # Calculate z-scores with protection against small standard deviations
        # Use a minimum threshold for standard deviation
        min_std = 1e-8
        effective_std = std.clip(lower=min_std)
        
        # Calculate z-score
        zscore = (series - mean) / effective_std
        
        # Cap z-scores to range [-10, 10]
        return zscore.clip(lower=-10, upper=10)
    
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
        # Use Numba-accelerated version
        closes = df['close'].values
        volumes = df['volume'].values
        obv_values = _calculate_obv_numba(closes, volumes)
        return pd.Series(obv_values, index=df.index)
    
    def calculate_btc_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calculate BTC-specific features.
        
        Args:
            index: DatetimeIndex to align with
            
        Returns:
            DataFrame with BTC features
        """
        if self.btc_df is None:
            logger.warning("Bitcoin data not found in DataFrame. Cannot calculate BTC features. "
                         "Ensure your DataFrame includes a symbol containing 'BTC' (e.g., 'BTC-USDT-SWAP', 'BTCUSDT').")
            return pd.DataFrame(index=index)
        
        # Create a dictionary to hold all BTC features
        btc_data = {}
        
        for period in self.params.return_periods:
            # Calculate BTC return
            btc_return = self.calculate_return(self.btc_df, period)
            
            # Reindex to match the target symbol's index
            btc_return = btc_return.reindex(index)
            
            # Add as a feature
            btc_data[f'btc_return_{period}m'] = btc_return
        
        # Create a single DataFrame from all features at once
        return pd.DataFrame(btc_data, index=index)


# Example usage
def create_features(df: pd.DataFrame, params: FeatureParams = None) -> pd.DataFrame:
    """
    Convenience function to add all features to a market data DataFrame.
    Raw OHLCV data is excluded from the result.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        params: Feature calculation parameters. If None, uses default parameters.
        
    Returns:
        DataFrame with normalized/relative features only
    """
    # Configure Numba threads
    num_threads = os.cpu_count() or 4
    nb.set_num_threads(min(num_threads, 8))  # Limit to max 8 threads
    
    engineer = FeatureEngineer(df, params=params)
    # Create a copy to defragment
    return engineer.add_all_features().copy()


def create_sequence_features(df: pd.DataFrame, 
                             sequence_length: int = 60,
                             params: FeatureParams = None) -> pd.DataFrame:
    """
    Convenience function to create sequence features from market data.
    Each column in the resulting DataFrame contains arrays of the past values.
    Raw OHLCV data is excluded.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        sequence_length: Length of the sequence arrays
        params: Feature calculation parameters. If None, uses default parameters.
        
    Returns:
        DataFrame with sequence features only
    """
    # Configure Numba threads
    num_threads = os.cpu_count() or 4
    nb.set_num_threads(min(num_threads, 8))  # Limit to max 8 threads
    
    engineer = FeatureEngineer(df, params=params)
    # Create a copy to defragment
    return engineer.create_sequence_features(
        sequence_length=sequence_length
    ).copy()
