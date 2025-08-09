import datetime
import pandas as pd
import os
from typing import Any

from market_data.util.cache.cache_common import to_local_filename
from market_data.ingest.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.ingest.bq.common import get_full_table_id
from market_data.util.time import TimeRange
from market_data.util.cache.time import split_t_range
from market_data.target.target import TargetParamsBatch
from market_data.machine_learning.resample.resample import ResampleParams
from market_data.feature.util import parse_feature_label_param, parse_feature_label_params


import market_data.feature.impl  # Import to ensure all features are registered


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
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
            
        base_label = market_data.util.cache.cache_common.get_label(dataset_mode, export_mode)
        folder_path = os.path.join(market_data.util.cache.cache_common.cache_base_path, "market_data", base_label)
        filename = to_local_filename(folder_path, d_from, d_to)
        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges



def check_missing_feature_data(
        feature_label: str,
        feature_params: Any,
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        time_range: TimeRange
) -> list:
    """
    Check which date ranges are missing from the feature cache.
    
    Returns a list of (start_date, end_date) tuples for missing days.
    """
    feature_label, params = parse_feature_label_param((feature_label, feature_params,))
    params_dir = params.get_params_dir()
    
    t_from, t_to = time_range.to_datetime()
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        base_label = market_data.util.cache.cache_common.get_label(dataset_mode, export_mode)
        folder_path = os.path.join(market_data.util.cache.cache_common.cache_base_path, "feature_data", "features", feature_label, params_dir, base_label)
        filename = to_local_filename(folder_path, d_from, d_to)

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
    from market_data.util.cache.path import params_to_dir_name
    
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
        
        base_label = market_data.util.cache.cache_common.get_label(dataset_mode, export_mode)
        folder_path = os.path.join(market_data.util.cache.cache_common.cache_base_path, "feature_data", "targets", params_dir, base_label)
        filename = to_local_filename(folder_path, d_from, d_to)

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
    from market_data.util.cache.path import params_to_dir_name
    from dataclasses import asdict
    
    resample_params = resample_params or ResampleParams()
    params_dir = params_to_dir_name(asdict(resample_params))
    
    t_from, t_to = time_range.to_datetime()
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        base_label = market_data.util.cache.cache_common.get_label(dataset_mode, export_mode)
        folder_path = os.path.join(market_data.util.cache.cache_common.cache_base_path, "feature_data", "resampled", params_dir, base_label)
        filename = to_local_filename(folder_path, d_from, d_to)

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
    from market_data.machine_learning.feature_resample.cache_feature_resample import _get_feature_resampled_params_dir
    from market_data.feature.util import parse_feature_label_param
    
    # Parse feature parameters
    feature_label_param = parse_feature_label_param((feature_label, feature_params))
    resample_params = resample_params or ResampleParams()
    
    # Get parameter directory using the same logic as cache_feature_resample
    params_dir = _get_feature_resampled_params_dir(resample_params, feature_label_param, seq_params)
    
    t_from, t_to = time_range.to_datetime()
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        base_label = market_data.util.cache.cache_common.get_label(dataset_mode, export_mode)
        folder_path = os.path.join(market_data.util.cache.cache_common.cache_base_path, "feature_data", "feature_resampled", params_dir, base_label)
        filename = to_local_filename(folder_path, d_from, d_to)

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
    from market_data.machine_learning.ml_data.cache_ml_data import _get_mldata_params_dir
    
    # Normalize parameters
    feature_label_params = parse_feature_label_params(feature_label_params)
    target_params_batch = target_params_batch or TargetParamsBatch()
    resample_params = resample_params or ResampleParams()
    
    # Get parameter directory
    params_dir = _get_mldata_params_dir(resample_params, feature_label_params, target_params_batch)
    if seq_params is not None:
        params_dir = os.path.join(seq_params.get_params_dir(), params_dir)
    
    t_from, t_to = time_range.to_datetime()
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        base_label = market_data.util.cache.cache_common.get_label(dataset_mode, export_mode)
        folder_path = os.path.join(market_data.util.cache.cache_common.cache_base_path, "feature_data", "ml_data", params_dir, base_label)
        filename = to_local_filename(folder_path, d_from, d_to)

        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges
