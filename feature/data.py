import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from feature.feature import create_features, create_sequence_features
from feature.target import create_targets, get_target_columns, TARGET_COLUMNS_DEFAULT

def create_features_with_targets(
        df: pd.DataFrame, 
        forward_periods: List[int] = [2, 10],
        tp_values: List[float] = [0.03],
        sl_values: List[float] = [0.03],
        return_periods: List[int] = [1, 5, 15, 30, 60, 120],
        ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
        add_btc_features: bool = True
    ) -> pd.DataFrame:
    """
    Create both feature and target columns in a single DataFrame.
    
    This function creates both feature and target DataFrames, then joins them
    using a multi-index of ["timestamp", "symbol"] to ensure proper alignment.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        forward_periods: Periods (in minutes) for calculating forward returns
        tp_values: Take-profit threshold values (as decimal)
        sl_values: Stop-loss threshold values (as decimal)
        return_periods: Periods (in minutes) for calculating returns
        ema_periods: Periods (in minutes) for calculating EMAs
        add_btc_features: Whether to add BTC-related features
        
    Returns:
        Combined DataFrame with both feature and target columns
    """
    # Generate feature and target DataFrames
    df_features = create_features(
        df, 
        return_periods=return_periods,
        ema_periods=ema_periods,
        add_btc_features=add_btc_features
    )
    
    df_targets = create_targets(
        df,
        forward_periods=forward_periods,
        tp_values=tp_values,
        sl_values=sl_values
    )
    
    # Join the DataFrames using a multi-index
    combined_df = df_features.reset_index().set_index(["timestamp", "symbol"]).join(
        df_targets.reset_index().set_index(["timestamp", "symbol"])
    )
    
    # Reset the index to restore timestamp as the index
    combined_df = combined_df.reset_index().set_index("timestamp")
    
    return combined_df


def create_sequence_features_with_targets(
        df: pd.DataFrame,
        sequence_length: int = 60,
        forward_periods: List[int] = [2, 10],
        tp_values: List[float] = [0.03],
        sl_values: List[float] = [0.03],
        return_periods: List[int] = [1, 5, 15, 30, 60, 120],
        ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
        add_btc_features: bool = True
    ) -> pd.DataFrame:
    """
    Create sequence features and targets in a single DataFrame.
    
    This function creates both sequence feature and target DataFrames, then joins them
    using a multi-index of ["timestamp", "symbol"] to ensure proper alignment.
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        sequence_length: Length of the sequence arrays
        forward_periods: Periods (in minutes) for calculating forward returns
        tp_values: Take-profit threshold values (as decimal)
        sl_values: Stop-loss threshold values (as decimal)
        return_periods: Periods (in minutes) for calculating returns
        ema_periods: Periods (in minutes) for calculating EMAs
        add_btc_features: Whether to add BTC-related features
        
    Returns:
        Combined DataFrame with both sequence features and target columns
    """
    # Create sequence features
    df_sequences = create_sequence_features(
        df,
        sequence_length=sequence_length
    )
    
    # Create targets
    df_targets = create_targets(
        df,
        forward_periods=forward_periods,
        tp_values=tp_values,
        sl_values=sl_values
    )
    
    # Join the DataFrames using a multi-index
    combined_df = df_sequences.reset_index().set_index(["timestamp", "symbol"]).join(
        df_targets.reset_index().set_index(["timestamp", "symbol"])
    )
    
    # Reset the index to restore timestamp as the index
    combined_df = combined_df.reset_index().set_index("timestamp")
    
    return combined_df