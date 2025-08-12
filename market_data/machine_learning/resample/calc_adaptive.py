from dataclasses import dataclass

import pandas as pd

from market_data.machine_learning.resample.param import ResampleParam


@dataclass
class AdaptiveResampleParams(ResampleParam):
    """Parameters for resampling data at significant price movements."""
    price_col: str = 'close'
    threshold: float = 0.05
    k: float = 2.0  # Sensitivity multiplier for the volatility
    vol_lookback: int = 60
    
    def get_target_frequency(self) -> str:
        """Get target frequency - adaptive for volatility-based resampling."""
        return "adaptive"
    
    @classmethod
    def from_str(cls, param_str: str) -> 'AdaptiveResampleParams':
        """Parse parameters from string format: price_col:close,threshold:0.05,k:2.0,vol_lookback:60"""
        params = {}
        for pair in param_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'price_col':
                    params['price_col'] = value
                elif key == 'threshold':
                    params['threshold'] = float(value)
                elif key == 'k':
                    params['k'] = float(value)
                elif key == 'vol_lookback':
                    params['vol_lookback'] = int(value)
        return cls(**params)
    
    def to_str(self) -> str:
        """Convert parameters to string format: price_col:close,threshold:0.05,k:2.0,vol_lookback:60"""
        return f"price_col:{self.price_col},threshold:{self.threshold},k:{self.k},vol_lookback:{self.vol_lookback}"

def _get_adaptive_events_t(
        df: pd.DataFrame,
        params: AdaptiveResampleParams = None,
) -> pd.DatetimeIndex:
    """
    Adaptive CUSUM filter using volatility-scaled threshold.

    Args:
        df: DataFrame with timestamp index and target column (e.g., mid price)
        col: Name of the target column
        k: Sensitivity multiplier for the volatility
        vol_lookback: Rolling window size to estimate volatility (in time steps)

    Returns:
        DatetimeIndex of CUSUM-triggered event timestamps
    """
    t_events = []
    s_pos, s_neg = 0.0, 0.0

    # Log return changes
    diff = df[params.price_col].pct_change().fillna(0)
    vol = diff.rolling(params.vol_lookback).std().ffill() * 100

    for i in range(1, len(diff)):
        threshold = params.k * vol.iloc[i]
        s_pos = max(0.0, s_pos + diff.iloc[i])
        s_neg = min(0.0, s_neg + diff.iloc[i])

        if s_pos > threshold:
            t_events.append(df.index[i])
            s_pos = 0.0
        elif s_neg < -threshold:
            t_events.append(df.index[i])
            s_neg = 0.0

    return pd.DatetimeIndex(t_events).as_unit(df.index.unit)

def _get_events_t_multi(
        df: pd.DataFrame,
        params: AdaptiveResampleParams,
    ) -> pd.DataFrame:
    if 'symbol' not in df.columns:
        raise ValueError("DataFrame must contain a 'symbol' column")
    
    all_events = []
    
    # Group by symbol to process each symbol separately
    for symbol, symbol_df in df.groupby('symbol'):
        # Extract date from timestamp index for date-based resets
        symbol_df = symbol_df.copy()
        symbol_df['date'] = symbol_df.index.date
        
        # Process each date separately for this symbol
        for date, date_df in symbol_df.groupby('date'):
            # Skip dates with insufficient data
            if len(date_df) <= 1:
                continue
                
            # Use get_events_t to identify events for this symbol/date
            events_idx = _get_adaptive_events_t(date_df, params)
            
            # Add to all_events with the corresponding symbol
            for timestamp in events_idx:
                all_events.append({'timestamp': timestamp, 'symbol': symbol})
    
    # Convert list of events to DataFrame
    if not all_events:
        return pd.DataFrame(columns=['symbol'], index=pd.DatetimeIndex([]))
    
    events_df = pd.DataFrame(all_events)
    events_df = events_df.set_index('timestamp')    
    events_df.index = events_df.index.as_unit(df.index.unit)

    return events_df

def resample_at_events(
    df: pd.DataFrame, 
    params: AdaptiveResampleParams = None,
) -> pd.DataFrame:
    """
    Generic function to resample a DataFrame at significant price movement events.
    
    This function:
    1. Takes any DataFrame with 'timestamp' index and 'symbol' column
    2. Identifies significant price movements using the specified price column
    3. Filters the DataFrame to include only rows at those significant price movement timestamps
    
    Args:
        df: DataFrame with timestamp index, 'symbol' column, and price columns
        params: ResampleParams object containing resampling parameters. If None, uses default parameters.
        
    Returns:
        Filtered DataFrame containing only rows at significant price movement timestamps
    """
    # Use default parameters if none provided
    params = params or AdaptiveResampleParams()
    
    # Ensure required columns exist
    if params.price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{params.price_col}' column for detecting price movements")
    if 'symbol' not in df.columns:
        raise ValueError("DataFrame must contain 'symbol' column")
    
    # Identify significant price movements
    events_df = _get_events_t_multi(
        df,
        params
    )
    
    if events_df.empty:
        return pd.DataFrame(columns=df.columns, index=pd.DatetimeIndex([]))
    
    # Create multi-indices for filtering
    events_multi = events_df.reset_index().set_index(['timestamp', 'symbol'])
    df_multi = df.reset_index().set_index(['timestamp', 'symbol'])
    
    # Filter to event timestamps only
    common_idx = events_multi.index.intersection(df_multi.index)
    resampled_df = df_multi.loc[common_idx].reset_index().set_index('timestamp')
    
    return resampled_df
