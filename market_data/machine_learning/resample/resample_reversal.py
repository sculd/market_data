import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass

from .resample_registry import register_resample, register_resample_function

@register_resample("reversal")
@dataclass
class ResampleReversalParams:
    """Parameters for reversal-based resampling at significant price movements.
    
    This implements a two-threshold approach:
    1. Breakout detection: Wait for movement beyond `threshold`
    2. Reversal detection: Emit signal when movement reverses by `threshold_reversal`
    
    This approach captures complete movement cycles rather than just breakout moments.
    """
    price_col: str = 'close'
    threshold: float = 0.1
    threshold_reversal: float = 0.03

    @staticmethod
    def parse_resample_params(param_str):
        """
        Parse a string in the format 'price_col,threshold,threshold_reversal' into ResampleReversalParams.
        Example: 'close,0.1,0.03' -> ResampleReversalParams(price_col='close', threshold=0.1, threshold_reversal=0.03)
        
        Args:
            param_str: String in format 'price_col,threshold,threshold_reversal'
            
        Returns:
            ResampleReversalParams instance
        """
        if not param_str:
            return ResampleReversalParams()

        try:
            parts = param_str.split(',')
            if len(parts) != 3:
                raise ValueError("Format should be 'price_col,threshold,threshold_reversal'")
                
            price_col = parts[0].strip()
            threshold = float(parts[1].strip())
            threshold_reversal = float(parts[2].strip())
            
            return ResampleReversalParams(
                price_col=price_col, 
                threshold=threshold,
                threshold_reversal=threshold_reversal,
            )
        except Exception as e:
            raise ValueError(f"Invalid resample_params format: {e}. Format should be 'price_col,threshold,threshold_reversal' (e.g. 'close,0.1,0.03')")

    @staticmethod
    def get_default_params_for_config(config_name: str) -> List[str]:
        """
        Get default parameter strings for different market configurations.
        
        Args:
            config_name: Configuration name ('stock', 'forex', 'crypto', 'default')
            
        Returns:
            List of parameter strings in format 'price_col,threshold,threshold_reversal'
        """
        if config_name == 'stock':
            # For stock: threshold_reversal = 40% of threshold
            return [
                "close,0.07,0.02", 
                "close,0.1,0.02", 
                "close,0.15,0.03"
            ]
        elif config_name == 'forex':
            # For forex: threshold_reversal = 40% of threshold
            return [
                "close,0.0025,0.005", 
            ]
        elif config_name == 'crypto':
            # For crypto: threshold_reversal = 40% of threshold
            return [
                "close,0.07,0.02", 
                "close,0.1,0.02", 
                "close,0.15,0.03"
            ]
        elif config_name == 'default':
            # For default: threshold_reversal = 40% of threshold
            return [
                "close,0.1,0.02", 
            ]
        else:
            raise ValueError(f"Unknown config name: {config_name}")

            
def _get_events_t(
        df: pd.DataFrame,
        params: ResampleReversalParams,
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
    s_pos = s_neg = 0
    s_max = s_min = 0
    state = 0
    diff = df[params.price_col].pct_change()
    
    for i in diff.index[1:]:
        ret = diff.loc[i]
        s_pos = max(0, s_pos + ret)
        s_neg = min(0, s_neg + ret)  # Fixed: use min(0, ...) for negative accumulation

        if state == 0:  # Neutral state
            if s_pos > params.threshold:
                state = 1
                s_max = s_pos
            elif s_neg < -params.threshold:
                state = -1
                s_min = s_neg

        elif state == 1:  # Positive breakout state
            if s_pos > s_max:
                s_max = s_pos
            elif s_max - s_pos > params.threshold_reversal:
                # Positive breakout reversal detected
                state = 0
                events.append({
                    'timestamp': i,
                    'breakout_side': 1,
                    's_final': s_pos,
                    's_extreme': s_max,
                })
                s_pos = s_neg = 0
                s_max = s_min = 0

        elif state == -1:  # Negative breakout state
            if s_neg < s_min:
                s_min = s_neg
            elif s_neg - s_min > params.threshold_reversal:
                # Negative breakout reversal detected
                state = 0
                events.append({
                    'timestamp': i,
                    'breakout_side': -1,
                    's_final': s_neg,
                    's_extreme': s_min,
                })
                s_pos = s_neg = 0
                s_max = s_min = 0

    if not events:
        return pd.DataFrame(columns=['breakout_side', 's_final', 's_extreme'], 
                          index=pd.DatetimeIndex([], name='timestamp'))
    
    events_df = pd.DataFrame(events)
    events_df = events_df.set_index('timestamp')
    events_df.index = events_df.index.as_unit(df.index.unit)
    
    return events_df

def _get_events_t_multi(
        df: pd.DataFrame,
        params: ResampleReversalParams,
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
                    **row,
                })
    
    # Convert list of events to DataFrame
    if not all_events:
        return pd.DataFrame(columns=['symbol', 'breakout_side', 's_final', 's_extreme'], 
                          index=pd.DatetimeIndex([], name='timestamp'))
    
    events_df = pd.DataFrame(all_events)
    events_df = events_df.set_index('timestamp')    
    events_df.index = events_df.index.as_unit(df.index.unit)

    return events_df

@register_resample_function("reversal")
def resample_at_events(
    df: pd.DataFrame, 
    params: ResampleReversalParams = None,
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
    params = params or ResampleReversalParams()
    
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
        empty_df = pd.DataFrame(columns=list(df.columns) + ['breakout_side', 's_final', 's_extreme'], 
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
