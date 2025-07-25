import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass

from .resample_registry import register_resample, register_resample_function

@register_resample("cumsum")
@dataclass
class ResampleParams:
    """Parameters for resampling data at significant price movements using López de Prado's CUMSUM filter."""
    price_col: str = 'close'
    threshold: float = 0.05

    @staticmethod
    def get_default_params_for_config(config_name: str) -> List[str]:
        """
        Get default parameter strings for different market configurations.
        
        Args:
            config_name: Configuration name ('stock', 'forex', 'crypto', 'default')
            
        Returns:
            List of parameter strings in format 'price_col,threshold'
        """
        if config_name == 'stock':
            return ["close,0.03", "close,0.05", "close,0.07", "close,0.1", "close,0.15"]
        elif config_name == 'forex':
            return ["close,0.0025", "close,0.005", "close,0.01"]
        elif config_name == 'crypto':
            return ["close,0.03", "close,0.05", "close,0.07", "close,0.1", "close,0.15"]
        elif config_name == 'default':
            return ["close,0.03", "close,0.05", "close,0.07", "close,0.1", "close,0.15"]
        else:
            raise ValueError(f"Unknown config name: {config_name}")

    @staticmethod
    def parse_resample_params(param_str):
        """
        Parse a string in the format 'price_col,threshold' into ResampleParams.
        Example: 'close,0.07' -> ResampleParams(price_col='close', threshold=0.07)
        
        Args:
            param_str: String in format 'price_col,threshold'
            
        Returns:
            ResampleParams instance
        """
        if not param_str:
            return ResampleParams()
            
        try:
            parts = param_str.split(',')
            if len(parts) != 2:
                raise ValueError("Format should be 'price_col,threshold'")
                
            price_col = parts[0].strip()
            threshold = float(parts[1].strip())
            
            return ResampleParams(price_col=price_col, threshold=threshold)
        except Exception as e:
            raise ValueError(f"Invalid resample_params format: {e}. Format should be 'price_col,threshold' (e.g. 'close,0.07')")

def _get_events_t_arr(values: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    t_events = []
    s_pos, s_neg = 0, 0
    pct_change = np.diff(values) / values[:-1]
    diff = np.diff(pct_change)
    for i in diff[1:]:
        s_pos = max(0, s_pos + diff[i])
        s_neg = min(0, s_neg + diff[i])
        if s_pos > threshold:
            t_events.append(i)
            s_pos = 0
        elif s_neg < -threshold:
            t_events.append(i)
            s_neg = 0
            
    return np.array(t_events)

def _get_events_t(
        df: pd.DataFrame,
        params: ResampleParams,
    ) -> pd.DataFrame:
    """
    Get time index from a DataFrame where the target column cumulatively changes by more than threshold.
    
    Args:
        df: DataFrame with timestamp index and target column
        params: ResampleParams object containing resampling parameters
        
    Returns:
        DataFrame with columns: timestamp (index), breakout_side
        where breakout_side is +1 for positive breakout, -1 for negative breakout
    """
    events = []
    s_pos, s_neg = 0, 0
    diff = df[params.price_col].pct_change()
    
    for i in diff.index[1:]:
        s_pos = max(0, s_pos + diff.loc[i])
        s_neg = min(0, s_neg + diff.loc[i])
        
        if s_pos > params.threshold:
            events.append({
                'timestamp': i,
                'breakout_side': 1  # Positive breakout
            })
            s_pos = 0
        elif s_neg < -params.threshold:
            events.append({
                'timestamp': i,
                'breakout_side': -1  # Negative breakout
            })
            s_neg = 0
    
    if not events:
        return pd.DataFrame(columns=['breakout_side'], 
                          index=pd.DatetimeIndex([], name='timestamp'))
    
    events_df = pd.DataFrame(events)
    events_df = events_df.set_index('timestamp')
    events_df.index = events_df.index.as_unit(df.index.unit)
    
    return events_df

def _get_events_t_multi(
        df: pd.DataFrame,
        params: ResampleParams,
    ) -> pd.DataFrame:
    """
    Get time index from a DataFrame with multiple symbols where the target column cumulatively 
    changes by more than threshold, resetting the cumulative sums for each new symbol and date.
    
    Args:
        df: DataFrame with timestamp index and target column, must contain 'symbol' column
        params: ResampleParams object containing resampling parameters. If None, uses default parameters.
        
    Returns:
        DataFrame with columns: timestamp (index), symbol, breakout_side
        where breakout_side is +1 for positive breakout, -1 for negative breakout
    """
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
            events_df = _get_events_t(date_df, params)
            
            # Add to all_events with the corresponding symbol and event details
            for timestamp, row in events_df.iterrows():
                all_events.append({
                    'timestamp': timestamp, 
                    'symbol': symbol,
                    'breakout_side': row['breakout_side']
                })
    
    # Convert list of events to DataFrame
    if not all_events:
        return pd.DataFrame(columns=['symbol', 'breakout_side'], 
                          index=pd.DatetimeIndex([], name='timestamp'))
    
    events_df = pd.DataFrame(all_events)
    events_df = events_df.set_index('timestamp')    
    events_df.index = events_df.index.as_unit(df.index.unit)

    return events_df

@register_resample_function("cumsum")
def resample_at_events(
    df: pd.DataFrame, 
    params: ResampleParams = None,
) -> pd.DataFrame:
    """
    Generic function to resample a DataFrame at significant price movement events.
    
    This function:
    1. Takes any DataFrame with 'timestamp' index and 'symbol' column
    2. Identifies significant price movements using the specified price column
    3. Filters the DataFrame to include only rows at those significant price movement timestamps
    4. Adds event information: breakout_side column indicating breakout direction
    
    Args:
        df: DataFrame with timestamp index, 'symbol' column, and price columns
        params: ResampleParams object containing resampling parameters. If None, uses default parameters.
        
    Returns:
        Filtered DataFrame containing only rows at significant price movement timestamps,
        with additional column: breakout_side (+1 for positive breakout, -1 for negative breakout)
    """
    # Use default parameters if none provided
    params = params or ResampleParams()
    
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
        # Return empty DataFrame with all original columns plus event columns
        empty_df = pd.DataFrame(columns=list(df.columns) + ['breakout_side'], 
                               index=pd.DatetimeIndex([]))
        return empty_df
    
    # Create multi-indices for merging
    events_multi = events_df.reset_index().set_index(['timestamp', 'symbol'])
    df_multi = df.reset_index().set_index(['timestamp', 'symbol'])
    
    # Merge the DataFrames to include event information
    merged_df = df_multi.join(events_multi, how='inner')
    
    # Reset index to get timestamp back as index
    resampled_df = merged_df.reset_index().set_index('timestamp')
    
    return resampled_df
