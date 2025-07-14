import datetime
import pandas as pd
import os
from pathlib import Path


from market_data.ingest.bq.cache import to_filename, _cache_base_path
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from market_data.util.time import TimeRange
from market_data.util.cache.time import split_t_range
from market_data.feature.util import parse_feature_label_param
from market_data.target.target import TargetParamsBatch
from market_data.machine_learning.resample.resample import ResampleParams
from market_data.feature.util import parse_feature_label_params
from market_data.machine_learning.cache_ml_data import (
    CACHE_BASE_PATH
)


import market_data.feature.impl  # Import to ensure all features are registered

def group_consecutive_dates(date_ranges):
    """
    Group consecutive date ranges into larger ranges.
    
    Args:
        date_ranges: List of (start_date, end_date) tuples
        
    Returns:
        List of (start_date, end_date) tuples with consecutive dates grouped
    """
    if not date_ranges:
        return []
    
    # Sort by start date
    sorted_ranges = sorted(date_ranges, key=lambda x: x[0])
    
    grouped_ranges = []
    current_start, current_end = sorted_ranges[0]
    
    for i in range(1, len(sorted_ranges)):
        next_start, next_end = sorted_ranges[i]
        
        # If next range starts on the same day as current range ends,
        # they are consecutive
        if next_start.date() == current_end.date():
            current_end = next_end
        else:
            grouped_ranges.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    
    # Don't forget to add the last range
    grouped_ranges.append((current_start, current_end))
    
    return grouped_ranges


def check_missing_raw_data(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        time_range: TimeRange
) -> list:
    """
    Check which date ranges are missing from the cache.
    
    Returns a list of (start_date, end_date) tuples for missing days.
    """
    t_from, t_to = time_range.to_datetime()
    t_id = get_full_table_id(dataset_mode, export_mode)
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        # Check if file exists in cache
        filename = to_filename(_cache_base_path, "market_data", t_id, aggregation_mode, d_from, d_to)
        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges



def check_missing_feature_data(
        feature_label: str,
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        time_range: TimeRange
) -> list:
    """
    Check which date ranges are missing from the feature cache.
    
    Returns a list of (start_date, end_date) tuples for missing days.
    """
    from market_data.feature.cache_feature import FEATURE_CACHE_BASE_PATH
    from market_data.util.cache.path import to_filename
    
    # Parse feature_label to get params
    feature_label, params = parse_feature_label_param(feature_label)
    params_dir = params.get_params_dir()
    
    t_from, t_to = time_range.to_datetime()
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        # Check if file exists in cache
        cache_path = f"{FEATURE_CACHE_BASE_PATH}/features"
        filename = to_filename(
            cache_path, 
            feature_label, 
            d_from, 
            d_to, 
            params_dir=params_dir,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode
        )
        
        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges


def check_missing_target_data(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        time_range: TimeRange,
        target_params: TargetParamsBatch = None
) -> list:
    """
    Check which date ranges are missing from the target data cache.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range to check
        target_params: Target parameters. If None, uses default parameters
        
    Returns:
        A list of (start_date, end_date) tuples for missing days
    """
    from market_data.target.cache_target import TARGET_CACHE_BASE_PATH
    from market_data.util.cache.path import to_filename, params_to_dir_name
    from dataclasses import asdict
    
    # Create default params if None
    target_params = target_params or TargetParamsBatch()
    
    # Convert params to directory name
    params_dict = {
        'fp': sorted(set([p.forward_period for p in target_params.target_params_list])),
        'tp': sorted(set([p.tp_value for p in target_params.target_params_list])),
        'sl': sorted(set([p.sl_value for p in target_params.target_params_list])),
    }
    params_dir = params_to_dir_name(params_dict)
    
    t_from, t_to = time_range.to_datetime()
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        # Check if file exists in cache
        filename = to_filename(
            TARGET_CACHE_BASE_PATH, 
            "targets", 
            d_from, 
            d_to, 
            params_dir=params_dir,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode
        )
        
        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges


def check_missing_resampled_data(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        time_range: TimeRange,
        resample_params: ResampleParams = None
) -> list:
    """
    Check which date ranges are missing from the resampled data cache.
    
    Returns a list of (start_date, end_date) tuples for missing days.
    """
    from market_data.machine_learning.resample.cache_resample import RESAMPLE_CACHE_BASE_PATH
    from market_data.util.cache.path import to_filename, params_to_dir_name
    from market_data.ingest.bq.common import get_full_table_id
    from dataclasses import asdict
    
    resample_params = resample_params or ResampleParams()
    params_dir = params_to_dir_name(asdict(resample_params))
    
    t_from, t_to = time_range.to_datetime()
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        # Check if file exists in cache
        cache_path = f"{RESAMPLE_CACHE_BASE_PATH}"
        dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"
        
        # The resampled data filename pattern
        filename = to_filename(
            cache_path, 
            "resampled", 
            d_from, 
            d_to, 
            params_dir=params_dir,
            dataset_id=dataset_id
        )
        
        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges


def check_missing_feature_resampled_data(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        time_range: TimeRange,
        feature_label: str,
        feature_params=None,
        resample_params: ResampleParams = None,
        seq_params=None
) -> list:
    """
    Check which date ranges are missing from the feature_resampled data cache.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range to check
        feature_label: Name of the feature (e.g., 'bollinger_bands', 'rsi')
        feature_params: Parameters for the feature calculation
        resample_params: Resampling parameters. If None, uses default parameters
        seq_params: Sequential feature parameters. If provided, checks sequential feature_resampled data
        
    Returns:
        A list of (start_date, end_date) tuples for missing days
    """
    from market_data.machine_learning.cache_feature_resample import CACHE_BASE_PATH
    from market_data.machine_learning.cache_feature_resample import _get_feature_resampled_params_dir
    from market_data.util.cache.path import to_filename
    from market_data.ingest.bq.common import get_full_table_id
    from market_data.feature.util import parse_feature_label_param
    
    # Parse feature parameters
    feature_label_param = parse_feature_label_param((feature_label, feature_params))
    resample_params = resample_params or ResampleParams()
    
    # Get parameter directory using the same logic as cache_feature_resample
    params_dir = _get_feature_resampled_params_dir(resample_params, feature_label_param, seq_params)
    
    t_from, t_to = time_range.to_datetime()
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        # Check if file exists in cache - use "feature_resampled" as the label
        filename = to_filename(
            CACHE_BASE_PATH,
            feature_label,
            d_from, 
            d_to, 
            params_dir=params_dir,
            dataset_id=dataset_id
        )
        
        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges


def check_missing_ml_data(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        time_range: TimeRange,
        feature_label_params=None,
        target_params_batch=None,
        resample_params=None,
        seq_params=None
) -> list:
    """
    Check which date ranges are missing from the ML data cache.
    
    Returns a list of (start_date, end_date) tuples for missing days.
    """
    from market_data.util.cache.path import to_filename
    from market_data.machine_learning.cache_ml_data import _get_mldata_params_dir
    from market_data.ingest.bq.common import get_full_table_id
    
    # Normalize parameters
    feature_label_params = parse_feature_label_params(feature_label_params)
    target_params_batch = target_params_batch or TargetParamsBatch()
    resample_params = resample_params or ResampleParams()
    
    # Get parameter directory
    params_dir = _get_mldata_params_dir(resample_params, feature_label_params, target_params_batch)
    if seq_params is not None:
        params_dir = os.path.join(seq_params.get_params_dir(), params_dir)
    
    t_from, t_to = time_range.to_datetime()
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        # Check if file exists in cache
        filename = to_filename(
            CACHE_BASE_PATH,
            "ml_data",
            d_from, 
            d_to, 
            params_dir=params_dir,
            dataset_id=dataset_id
        )
        
        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges
