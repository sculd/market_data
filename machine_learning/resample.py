import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple

def get_events_t(df: pd.DataFrame, col: str, threshold: float = 0.05) -> pd.DataFrame:
    """
    Get time index from a DataFrame where the target column cumulatively changes by more than threshold.
    
    Args:
        df: DataFrame with timestamp index and target column
        col: Name of the target column
        threshold: Threshold for the change in target column
    """
    t_events = []
    s_pos, s_neg = 0, 0
    diff =  df[col].pct_change().diff()
    for i in diff.index[1:]:
        s_pos = max(0, s_pos + diff.loc[i])
        s_neg = min(0, s_neg + diff.loc[i])
        if s_pos > threshold:
            t_events.append(i)
            s_pos = 0
        elif s_neg < -threshold:
            t_events.append(i)
            s_neg = 0
            
    return pd.DatetimeIndex(t_events).as_unit(df.index.unit)

def get_events_t_multi(df: pd.DataFrame, col: str, threshold: float = 0.05) -> pd.DataFrame:
    """
    Get time index from a DataFrame with multiple symbols where the target column cumulatively 
    changes by more than threshold, resetting the cumulative sums for each new symbol and date.
    
    Args:
        df: DataFrame with timestamp index and target column, must contain 'symbol' column
        col: Name of the target column
        threshold: Threshold for the change in target column
        
    Returns:
        DataFrame with timestamp index and symbol column, containing event timestamps
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
            events_idx = get_events_t(date_df, col, threshold)
            
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
    price_col: str = 'close',
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Generic function to resample a DataFrame at significant price movement events.
    
    This function:
    1. Takes any DataFrame with 'timestamp' index and 'symbol' column
    2. Identifies significant price movements using the specified price column
    3. Filters the DataFrame to include only rows at those significant price movement timestamps
    
    Args:
        df: DataFrame with timestamp index, 'symbol' column, and price columns
        price_col: Column to use for detecting significant price movements
        threshold: Threshold for significant price movement (as decimal)
        
    Returns:
        Filtered DataFrame containing only rows at significant price movement timestamps
    """
    # Ensure required columns exist
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column for detecting price movements")
    if 'symbol' not in df.columns:
        raise ValueError("DataFrame must contain 'symbol' column")
    
    # Identify significant price movements
    events_df = get_events_t_multi(
        df,
        price_col,
        threshold
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
