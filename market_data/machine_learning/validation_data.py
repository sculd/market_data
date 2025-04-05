import pandas as pd
import datetime
from typing import Tuple, Optional, List
from market_data.machine_learning.cache_ml_data import load_cached_ml_data

def create_train_validation_test_splits(
    dataset_mode,
    export_mode,
    aggregation_mode,
    time_range,
    feature_params=None,
    target_params=None,
    resample_params=None,
    purge_period: datetime.timedelta = datetime.timedelta(days=0),
    embargo_period: datetime.timedelta = datetime.timedelta(days=1),
    window_type: str = 'fixed',  # 'fixed' or 'expanding'
    fixed_window_size: datetime.timedelta = datetime.timedelta(days=10),
    step_size: datetime.timedelta = datetime.timedelta(days=5),
    split_ratio: List[float] = [0.7, 0.2, 0.1]
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Perform a time series split with purge and embargo.

    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_params: Feature calculation parameters
        target_params: Target calculation parameters
        resample_params: Resampling parameters
        purge_period: Timedelta for the purge period
        embargo_period: Timedelta for the embargo period
        window_type: 'fixed' for fixed-size window, 'expanding' for expanding window
        fixed_window_size: Timedelta for the fixed window size (default: 10 days)
        step_size: Timedelta for how often to generate new datasets (default: 5 days)
        split_ratio: List of three floats representing the ratio of train, validation, and test sets.
                    Must sum to 1.0 (default: [0.7, 0.2, 0.1])

    Returns:
        A list of tuples, where each tuple contains (train_set, validation_set, test_set)
    """
    # Validate split ratio
    if len(split_ratio) != 3:
        raise ValueError("split_ratio must be a list of three floats")
    if abs(sum(split_ratio) - 1.0) > 1e-10:
        raise ValueError("split_ratio must sum to 1.0")
    
    split_train, split_validation, split_test = split_ratio

    # Load the cached ML data
    ml_data = load_cached_ml_data(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        time_range=time_range,
        feature_params=feature_params,
        target_params=target_params,
        resample_params=resample_params
    )

    # Ensure the data is sorted by timestamp
    ml_data = ml_data.sort_index()

    # Preprocess: Purge data points within each symbol
    if purge_period > datetime.timedelta(0):
        purged_data = []
        for symbol, symbol_data in ml_data.groupby('symbol'):
            symbol_data = symbol_data.sort_index()
            if len(symbol_data) == 0:
                continue
                
            # Initialize with the first data point
            selected_indices = [symbol_data.index[0]]
            last_selected_time = symbol_data.index[0]
            
            # Iterate through remaining data points
            for idx in symbol_data.index[1:]:
                if idx - last_selected_time >= purge_period:
                    selected_indices.append(idx)
                    last_selected_time = idx
            
            # Keep only the selected data points
            purged_data.append(symbol_data.loc[selected_indices])
        
        # Combine all purged data
        ml_data = pd.concat(purged_data)
        ml_data = ml_data.sort_index()

    data_sets = []

    # Define the initial window
    window_start = ml_data.index[0]
    if window_type == 'fixed' and fixed_window_size:
        window_end = window_start + fixed_window_size
    else:
        window_end = window_start

    while window_end < ml_data.index[-1]:
        # Get the data up to window_end
        window_data = ml_data[window_start:window_end]
        
        # Calculate split points based on ratios
        total_points = len(window_data)
        train_end_idx = int(total_points * split_train) - 1 
        val_end_idx = int(total_points * (split_train + split_validation)) - 1

        # Apply purge and embargo periods
        train_end = window_data.index[train_end_idx]
        val_start = window_data.index[train_end_idx] + embargo_period
        val_end = window_data.index[val_end_idx]
        test_start = window_data.index[val_end_idx] + embargo_period

        data_sets.append((window_data[:train_end], window_data[val_start:val_end], window_data[test_start:],))

        # Move the window by step_size
        if window_type == 'fixed':
            window_start += step_size
        window_end += step_size

    return data_sets
