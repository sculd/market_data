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
                
            # Initialize cumulative sums for this symbol and date
            s_pos, s_neg = 0, 0
            diff = date_df[col].pct_change().diff()
            
            # Process each timestamp
            for i in diff.index[1:]:  # Skip first row as diff is NaN
                s_pos = max(0, s_pos + diff.loc[i])
                s_neg = min(0, s_neg + diff.loc[i])
                
                if s_pos > threshold:
                    all_events.append({'timestamp': i, 'symbol': symbol})
                    s_pos = 0
                elif s_neg < -threshold:
                    all_events.append({'timestamp': i, 'symbol': symbol})
                    s_neg = 0
    
    # Convert list of events to DataFrame
    if not all_events:
        return pd.DataFrame(columns=['symbol'], index=pd.DatetimeIndex([]))
    
    events_df = pd.DataFrame(all_events)
    events_df = events_df.set_index('timestamp')    
    events_df.index = events_df.index.as_unit(df.index.unit)

    return events_df

def create_resampled_dataset(
    df: pd.DataFrame, 
    price_col: str = 'close',
    threshold: float = 0.01,
    forward_periods: List[int] = [2, 10],
    tp_values: List[float] = [0.03],
    sl_values: List[float] = [0.03],
    return_periods: List[int] = [1, 5, 15, 30, 60, 120],
    ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
    add_btc_features: bool = True
) -> pd.DataFrame:
    """
    Create a resampled dataset with regular features, sampled at significant price movements.
    
    This function:
    1. Generates features and targets using create_features_with_targets
    2. Identifies significant price movements using get_events_t_multi
    3. Filters the dataset to include only those timestamps
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        price_col: Column to use for detecting significant price movements
        threshold: Threshold for significant price movement (as decimal)
        forward_periods: Periods (in minutes) for calculating forward returns
        tp_values: Take-profit threshold values (as decimal)
        sl_values: Stop-loss threshold values (as decimal)
        return_periods: Periods (in minutes) for calculating returns
        ema_periods: Periods (in minutes) for calculating EMAs
        add_btc_features: Whether to add BTC-related features
        
    Returns:
        Resampled DataFrame with features and targets at event timestamps
    """
    from feature import create_features_with_targets
    
    # Generate features and targets
    combined_df = create_features_with_targets(
        df,
        forward_periods=forward_periods,
        tp_values=tp_values,
        sl_values=sl_values,
        return_periods=return_periods,
        ema_periods=ema_periods,
        add_btc_features=add_btc_features
    )
    
    # Identify significant price movements
    events_df = get_events_t_multi(
        df,
        price_col,
        threshold
    )
    
    if events_df.empty:
        return pd.DataFrame()
    
    # Create multi-indices for filtering
    events_multi = events_df.reset_index().set_index(['timestamp', 'symbol'])
    combined_multi = combined_df.reset_index().set_index(['timestamp', 'symbol'])
    
    # Filter to event timestamps only
    common_idx = events_multi.index.intersection(combined_multi.index)
    resampled_df = combined_multi.loc[common_idx].reset_index().set_index('timestamp')
    
    return resampled_df

def create_resampled_seq_dataset(
    df: pd.DataFrame, 
    price_col: str = 'close',
    threshold: float = 0.01,
    sequence_length: int = 60,
    forward_periods: List[int] = [2, 10],
    tp_values: List[float] = [0.03],
    sl_values: List[float] = [0.03],
    return_periods: List[int] = [1, 5, 15, 30, 60, 120],
    ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
    add_btc_features: bool = True
) -> pd.DataFrame:
    """
    Create a resampled dataset with sequence features, sampled at significant price movements.
    
    This function:
    1. Generates sequence features and targets using create_sequence_features_with_targets
    2. Identifies significant price movements using get_events_t_multi
    3. Filters the dataset to include only those timestamps
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        price_col: Column to use for detecting significant price movements
        threshold: Threshold for significant price movement (as decimal)
        sequence_length: Length of the sequence arrays
        forward_periods: Periods (in minutes) for calculating forward returns
        tp_values: Take-profit threshold values (as decimal)
        sl_values: Stop-loss threshold values (as decimal)
        return_periods: Periods (in minutes) for calculating returns
        ema_periods: Periods (in minutes) for calculating EMAs
        add_btc_features: Whether to add BTC-related features
        
    Returns:
        Resampled DataFrame with sequence features and targets at event timestamps
    """
    from feature import create_sequence_features_with_targets
    
    # Generate sequence features and targets
    seq_df = create_sequence_features_with_targets(
        df,
        sequence_length=sequence_length,
        forward_periods=forward_periods,
        tp_values=tp_values,
        sl_values=sl_values,
        return_periods=return_periods,
        ema_periods=ema_periods,
        add_btc_features=add_btc_features
    )
    
    # Identify significant price movements
    events_df = get_events_t_multi(
        df,
        price_col,
        threshold
    )
    
    if events_df.empty:
        return pd.DataFrame()
    
    # Create multi-indices for filtering
    events_multi = events_df.reset_index().set_index(['timestamp', 'symbol'])
    seq_multi = seq_df.reset_index().set_index(['timestamp', 'symbol'])
    
    # Filter to event timestamps only
    common_idx = events_multi.index.intersection(seq_multi.index)
    resampled_seq_df = seq_multi.loc[common_idx].reset_index().set_index('timestamp')
    
    return resampled_seq_df
