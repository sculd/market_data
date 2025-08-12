import logging
from dataclasses import dataclass, field
from typing import List, Literal, Tuple

import numba as nb
import numpy as np
import pandas as pd

from market_data.util.param import Param

logger = logging.getLogger(__name__)

# Default values for target parameters
DEFAULT_FORWARD_PERIODS = [5, 10, 30]
DEFAULT_TP_VALUES = [0.015, 0.03, 0.05]

@dataclass
class TargetParams(Param):
    forward_period: int = 30
    tp_value: float = 0.03
    sl_value: float = 0.03
    
    @classmethod
    def from_str(cls, param_str: str) -> 'TargetParams':
        """Parse parameters from string format: forward_period:30,tp_value:0.03,sl_value:0.03"""
        params = {}
        for pair in param_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'forward_period':
                    params['forward_period'] = int(value)
                elif key == 'tp_value':
                    params['tp_value'] = float(value)
                elif key == 'sl_value':
                    params['sl_value'] = float(value)
        return cls(**params)
    
    def to_str(self) -> str:
        """Convert parameters to string format: forward_period:30,tp_value:0.03,sl_value:0.03"""
        return f"forward_period:{self.forward_period},tp_value:{self.tp_value},sl_value:{self.sl_value}"
    
    def __repr__(self) -> str:
        """String representation of the parameters."""
        return (
            f"TargetParams(forward_period={self.forward_period}, "
            f"tp_value={self.tp_value}, "
            f"sl_value={self.sl_value})"
        )

def _get_default_target_params_list() -> List[TargetParams]:
    return [TargetParams(forward_period=period, tp_value=tp, sl_value=tp) 
            for period in DEFAULT_FORWARD_PERIODS 
            for tp in DEFAULT_TP_VALUES]

@dataclass
class TargetParamsBatch(Param):
    """
    Encapsulates parameters for target engineering.
    
    This class holds all the parameters needed for target calculation,
    providing a single source of truth for target configuration.
    """
    target_params_list: List[TargetParams] = field(default_factory=lambda: _get_default_target_params_list())

    @classmethod
    def from_str(cls, param_str: str) -> 'TargetParamsBatch':
        """Parse parameters from string format: fp:5,10,30|tp:0.015,0.03,0.05|sl:0.015,0.03,0.05"""
        if not param_str:
            return cls()
        
        params_dict = {}
        for part in param_str.split('|'):
            if ':' in part:
                key, values = part.split(':', 1)
                if key == 'fp':
                    params_dict['forward_periods'] = [int(v) for v in values.split(',')]
                elif key == 'tp':
                    params_dict['tp_values'] = [float(v) for v in values.split(',')]
                elif key == 'sl':
                    params_dict['sl_values'] = [float(v) for v in values.split(',')]
        
        # Create target params list from the parsed values
        forward_periods = params_dict.get('forward_periods', DEFAULT_FORWARD_PERIODS)
        tp_values = params_dict.get('tp_values', DEFAULT_TP_VALUES)
        sl_values = params_dict.get('sl_values', DEFAULT_TP_VALUES)
        
        target_params_list = [
            TargetParams(forward_period=fp, tp_value=tp, sl_value=sl)
            for fp in forward_periods
            for tp in tp_values
            for sl in sl_values
        ]
        
        return cls(target_params_list=target_params_list)
    
    def to_str(self) -> str:
        """Convert parameters to string format: fp:5,10,30|tp:0.015,0.03,0.05|sl:0.015,0.03,0.05"""
        # Extract unique values from the list
        forward_periods = sorted(set(p.forward_period for p in self.target_params_list))
        tp_values = sorted(set(p.tp_value for p in self.target_params_list))
        sl_values = sorted(set(p.sl_value for p in self.target_params_list))
        
        parts = []
        if forward_periods:
            parts.append(f"fp:{','.join(map(str, forward_periods))}")
        if tp_values:
            parts.append(f"tp:{','.join(map(str, tp_values))}")
        if sl_values:
            parts.append(f"sl:{','.join(map(str, sl_values))}")
        
        return '|'.join(parts)

    def __repr__(self) -> str:
        """String representation of the parameters."""
        return '\n'.join([f'{p.__repr__}' for p in self.target_params_list])

@nb.njit(cache=True)
def _calculate_tp_sl_labels_and_scores_numba(closes, opens, period, take_profit, stop_loss, position_type):
    """
    Numba-accelerated take-profit/stop-loss label, return, and score calculation.
    
    Args:
        closes: Array of close prices
        highs: Array of high prices
        lows: Array of low prices
        period: Forward-looking period
        take_profit: Take-profit threshold as decimal
        stop_loss: Stop-loss threshold as decimal
        position_type: 0 for long, 1 for short
        
    Returns:
        Tuple of (labels, returns, scores) where:
        - labels: Array with classification labels (1=TP hit, -1=SL hit, 0=neither)
        - returns: Array with actual returns when TP/SL is hit or forward return
        - scores: Array with score when TP/SL is hit or forward return scaled by threshold
    """
    n = len(closes)
    labels = np.zeros(n)
    returns = np.full(n, np.nan)
    scores = np.full(n, np.nan)
    
    # Pre-compute TP/SL price levels for all data points
    if position_type == 0:  # long position
        tp_prices = closes * (1 + take_profit)
        sl_prices = closes * (1 - stop_loss)
    else:  # short position
        tp_prices = closes * (1 - take_profit)
        sl_prices = closes * (1 + stop_loss)
    
    # Process each position
    for i in range(n - 1):
        if i + period >= n:
            # Not enough future data
            continue

        price_cur = closes[i]
        end_idx = min(i + period, n - 1)  # End of period index
        tp_sl_hit = False

        for j in range(i + 1, end_idx):
            # note that the close price, not high/low values are used
            # one problem of using high/low is the difficulty of knowing 
            # the market price at tp/sl hit.
            price_fut = closes[j]

            # tp/sl happens at the next bar's open after observing price_fut crossing tp/sl level.
            price_tp_sl = opens[j+1]
            actual_return = (1 if position_type == 0 else -1) * (price_tp_sl - price_cur) / price_cur

            # Check forward window for TP/SL hits. 0 for long, 1 for short
            if (price_fut >= tp_prices[i] and position_type == 0) or \
                (price_fut <= tp_prices[i] and position_type == 1):
                labels[i] = 1
                returns[i] = actual_return
                scores[i] = actual_return / take_profit
                tp_sl_hit = True
                break
            elif (price_fut <= sl_prices[i] and position_type == 0) or \
                (price_fut >= sl_prices[i] and position_type == 1):
                labels[i] = -1
                returns[i] = actual_return
                scores[i] = actual_return / stop_loss
                tp_sl_hit = True
                break
        
        # If no TP/SL hit, use forward return at end of period
        if not tp_sl_hit:
            price_end = closes[end_idx]
            forward_return = (1 if position_type == 0 else -1) * (price_end - price_cur) / price_cur
            returns[i] = forward_return
            # Score based on whether forward return is positive or negative
            if forward_return >= 0:
                scores[i] = forward_return / take_profit
            else:
                scores[i] = forward_return / stop_loss
                
    return labels, returns, scores

def _calculate_tp_sl_labels_and_scores(df: pd.DataFrame, 
                                       period: int, 
                                       take_profit: float, 
                                       stop_loss: float, 
                                       position_type: Literal['long', 'short'] = 'long') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate classification labels, returns, and scores based on take-profit and stop-loss levels.
    
    Args:
        df: DataFrame with OHLCV data
        period: Number of periods (minutes) to look forward
        take_profit: Take-profit threshold as decimal (e.g., 0.01 for 1%)
        stop_loss: Stop-loss threshold as decimal (e.g., 0.01 for 1%)
        position_type: 'long' or 'short'
        
    Returns:
        Tuple of (labels, returns, scores) where:
        - labels: Series with classification labels (1=TP hit, -1=SL hit, 0=neither)
        - returns: Series with actual returns when TP/SL is hit or forward return
        - scores: Series with scores when TP/SL is hit or forward return scaled by threshold
    """
    # Extract arrays for Numba processing
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Convert position_type to integer for Numba
    pos_type_int = 0 if position_type == 'long' else 1
    
    # Call Numba-accelerated function (using the new function, ignoring returns and scores)
    labels, returns, scores = _calculate_tp_sl_labels_and_scores_numba(
        closes, opens, period, take_profit, stop_loss, pos_type_int
    )
    
    # Convert results back to pandas Series
    return (pd.Series(labels, index=df.index), 
            pd.Series(returns, index=df.index),
            pd.Series(scores, index=df.index))

class TargetEngineer:
    """
    Class for engineering target features from OHLCV market data.
    
    The input DataFrame should have:
    - 'timestamp' as index (with minute-level data)
    - Columns: 'symbol', 'open', 'high', 'low', 'close', 'volume'
    """
    
    def __init__(self, df: pd.DataFrame, params: TargetParamsBatch = None):
        """
        Initialize with a DataFrame of market data.
        
        Args:
            df: DataFrame with timestamp index and OHLCV columns
            params: Target parameters. If None, uses default parameters
        """
        self.validate_dataframe(df)
        self.df = df.copy()
        self.params = params or TargetParamsBatch()
    
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
    
    def add_target_features(self) -> pd.DataFrame:
        """
        Add target features for machine learning, including forward returns and
        classification labels based on take-profit and stop-loss levels.
        
        Returns:
            DataFrame with target features added
        """
        # Create an empty list to hold all target DataFrames
        all_targets = []
        
        # Group by symbol to calculate per-symbol targets
        for symbol, group in self.df.groupby('symbol'):
            # Create a dictionary to hold all target columns
            target_data = {'symbol': symbol}
            
            for target_params in self.params.target_params_list:
                period = target_params.forward_period
                target_data[f'label_forward_return_{period}m'] = self.calculate_forward_return(group, period)

                tp, sl = target_params.tp_value, target_params.sl_value
                
                # Long position labels and returns
                long_labels, long_returns, long_scores = _calculate_tp_sl_labels_and_scores(group, period, tp, sl, 'long')
                target_data[f'label_long_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m'] = long_labels
                target_data[f'label_long_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m_return'] = long_returns
                target_data[f'label_long_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m_score'] = long_scores

                # Short position labels and returns
                short_labels, short_returns, short_scores = _calculate_tp_sl_labels_and_scores(group, period, tp, sl, 'short')
                target_data[f'label_short_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m'] = short_labels
                target_data[f'label_short_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m_return'] = short_returns
                target_data[f'label_short_tp{int(tp*1000)}_sl{int(sl*1000)}_{period}m_score'] = short_scores

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


# Convenience functions
def create_targets(df: pd.DataFrame, params: TargetParamsBatch = None) -> pd.DataFrame:
    """
    Convenience function to create ML target features from market data.
    Includes forward returns and classification labels.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        params: Target calculation parameters. If None, uses default parameters.
        
    Returns:
        DataFrame with target features
    """
    engineer = TargetEngineer(df, params=params)
    # Create a copy to defragment
    return engineer.add_target_features().copy()

