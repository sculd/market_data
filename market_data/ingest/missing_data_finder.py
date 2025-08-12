import os
from typing import Any

from market_data.util.cache.common import to_local_filename
from market_data.ingest.common import CacheContext
from market_data.util.time import TimeRange
from market_data.util.cache.time import split_t_range
from market_data.target.calc import TargetParamsBatch
from market_data.machine_learning.resample.calc import CumSumResampleParams
from market_data.feature.label import FeatureLabelCollection, FeatureLabel


import market_data.feature.impl  # Import to ensure all features are registered


def check_missing_raw_data(
        cache_context: CacheContext,
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
            
        folder_path = cache_context.get_market_data_path()
        filename = to_local_filename(folder_path, d_from, d_to)
        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges



def check_missing_feature_data(
        cache_context: CacheContext,
        feature_label: FeatureLabel,
        time_range: TimeRange
) -> list:
    """
    Check which date ranges are missing from the feature cache.
    
    Returns a list of (start_date, end_date) tuples for missing days.
    """
    feature_name, params = feature_label.feature_label, feature_label.params
    params_dir = params.get_params_dir()
    
    t_from, t_to = time_range.to_datetime()
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        # Note: feature cache path structure is different - params_dir then base_label
        folder_path = cache_context.get_folder_path(["feature_data", "features", feature_name, params_dir])
        filename = to_local_filename(folder_path, d_from, d_to)

        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges


def check_missing_target_data(
        cache_context: CacheContext,
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
    target_params = target_params or TargetParamsBatch()
    params_dir = target_params.get_params_dir()
    
    t_from, t_to = time_range.to_datetime()
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        # Note: target cache path structure has params_dir before base_label  
        folder_path = cache_context.get_folder_path(["feature_data", "targets", params_dir])
        filename = to_local_filename(folder_path, d_from, d_to)

        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges


def check_missing_resampled_data(
        cache_context: CacheContext,
        time_range: TimeRange,
        resample_params: CumSumResampleParams = None
) -> list:
    """
    Check which date ranges are missing from the resampled data cache.
    
    Returns a list of (start_date, end_date) tuples for missing days.
    """
    resample_params = resample_params or CumSumResampleParams()
    params_dir = resample_params.get_params_dir()
    
    t_from, t_to = time_range.to_datetime()
    
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        # Note: resampled cache path structure has params_dir before base_label
        folder_path = cache_context.get_folder_path(["feature_data", "resampled", params_dir])
        filename = to_local_filename(folder_path, d_from, d_to)

        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges


def check_missing_feature_resampled_data(
        cache_context: CacheContext,
        time_range: TimeRange,
        feature_label: FeatureLabel,
        resample_params: CumSumResampleParams = None,
        seq_params=None
) -> list:
    """
    Check which date ranges are missing from the feature_resampled data cache.
    
    Args:
        cache_context: Cache context containing dataset, export and aggregation modes
        time_range: TimeRange object specifying the time range to check
        feature_label: FeatureLabel object containing feature name and parameters
        resample_params: Resampling parameters. If None, uses default parameters
        seq_params: Sequential feature parameters. If provided, checks sequential feature_resampled data
        
    Returns:
        A list of (start_date, end_date) tuples for missing days
    """
    from market_data.machine_learning.feature_resample.cache import _get_feature_resampled_params_dir
    
    # Parse feature parameters
    feature_label_param = (feature_label.feature_label, feature_label.params)
    resample_params = resample_params or CumSumResampleParams()
    
    # Get parameter directory using the same logic as cache_feature_resample
    params_dir = _get_feature_resampled_params_dir(resample_params, feature_label_param, seq_params)
    
    t_from, t_to = time_range.to_datetime()
    # Split the range into daily intervals
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        # Note: feature_resampled cache path structure has params_dir before base_label
        folder_path = cache_context.get_folder_path(["feature_data", "feature_resampled", params_dir])
        filename = to_local_filename(folder_path, d_from, d_to)

        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges


def check_missing_ml_data(
        cache_context: CacheContext,
        time_range: TimeRange,
        feature_collection: FeatureLabelCollection,
        target_params_batch=None,
        resample_params=None,
        seq_params=None
) -> list:
    """
    Check which date ranges are missing from the ML data cache.
    
    Returns a list of (start_date, end_date) tuples for missing days.
    """
    from market_data.machine_learning.ml_data.cache import _get_mldata_params_dir
    
    # Normalize parameters
    target_params_batch = target_params_batch or TargetParamsBatch()
    resample_params = resample_params or CumSumResampleParams()
    
    # Get parameter directory
    params_dir = _get_mldata_params_dir(resample_params, feature_collection, target_params_batch)
    if seq_params is not None:
        params_dir = os.path.join(seq_params.get_params_dir(), params_dir)
    
    t_from, t_to = time_range.to_datetime()
    daily_ranges = split_t_range(t_from, t_to)
    
    missing_ranges = []
    for d_range in daily_ranges:
        d_from, d_to = d_range
        
        # Note: ml_data cache path structure has params_dir before base_label
        folder_path = cache_context.get_folder_path(["feature_data", "ml_data", params_dir])
        filename = to_local_filename(folder_path, d_from, d_to)

        if not os.path.exists(filename):
            missing_ranges.append(d_range)
    
    return missing_ranges
