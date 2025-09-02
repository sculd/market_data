import datetime
import hashlib
import json
import logging
import os
from dataclasses import asdict
from typing import List, Optional, Tuple

import pandas as pd

import market_data.util.cache.read
import market_data.util.cache.write
from market_data.feature.label import FeatureLabel, FeatureLabelCollection
from market_data.feature.param import SequentialFeatureParam
from market_data.ingest.common import CacheContext
from market_data.machine_learning.ml_data.calc import calculate
from market_data.machine_learning.resample.calc import CumSumResampleParams
from market_data.machine_learning.resample.param import ResampleParam
from market_data.target.calc import TargetParamsBatch
from market_data.util.cache.parallel_processing import read_multithreaded
from market_data.util.cache.time import anchor_to_begin_of_day
from market_data.util.time import TimeRange

from market_data.machine_learning.ml_data.cache_description import (
    find_cached_ml_data_with_features,
    write_description_file,
)
logger = logging.getLogger(__name__)



def _generate_params_uuid(
    resample_params: ResampleParam,
    feature_collection: FeatureLabelCollection,
    target_params_batch: TargetParamsBatch,
    seq_param: Optional[SequentialFeatureParam] = None,
) -> str:
    """
    Generate a deterministic UUID based on all ML parameters.
    """
    # Create a consistent dictionary for all parameters
    params_dict = {
        'resample': asdict(resample_params),
        'target': target_params_batch.get_params_dir(),
        'features': [],
        'sequential': None
    }
    
    # Add features in sorted order for consistency
    for feature_label_obj in sorted(feature_collection.feature_labels, key=lambda x: x.feature_label):
        params_dict['features'].append({
            'label': feature_label_obj.feature_label,
            'params': feature_label_obj.params.get_params_dir()
        })
    
    # Add sequential params if present
    if seq_param is not None:
        params_dict['sequential'] = seq_param.get_params_dir()
    
    # Convert to JSON string with sorted keys for consistency
    params_str = json.dumps(params_dict, sort_keys=True)
    # Use SHA256 and take first 12 characters for a short but unique identifier
    return hashlib.sha256(params_str.encode()).hexdigest()[:12]

def _get_ml_data_params_dir(
    resample_params: ResampleParam,
    feature_collection: FeatureLabelCollection,
    target_params_batch: TargetParamsBatch,
    seq_param: Optional[SequentialFeatureParam] = None,
) -> str:
    """
    Generate a clean directory path using deterministic UUID for ML data parameters.
    
    Instead of concatenating all parameter descriptions (which creates very long paths),
    this function generates a deterministic UUID based on all parameters and stores
    the parameter descriptions in a description.txt file.
    """
    # Generate deterministic UUID from all parameters
    params_uuid = _generate_params_uuid(resample_params, feature_collection, target_params_batch, seq_param)
    
    # Return just the UUID - dataset_id is handled separately
    return params_uuid

def _calculate_and_cache__daily_ml_data(
    date: datetime.datetime,
    cache_context: CacheContext,
    feature_collection: FeatureLabelCollection,
    target_params_batch: TargetParamsBatch,
    resample_params: ResampleParam,
    seq_param: Optional[SequentialFeatureParam] = None,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache ML data for a single day.
    
    Can handle both regular and sequential features based on seq_param.
    Creates a description.txt file with human-readable parameter information.
    
    Args:
        date: The date to calculate ML data for
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        feature_collection: Collection of feature labels and their parameters
        target_params_batch: Target calculation parameters
        resample_params: Resampling parameters
        seq_param: Sequential feature parameters. If provided, creates sequential ML data.
        overwrite_cache: Whether to overwrite existing cache files
    """
    # Create time range for the specific day
    t_from = date
    t_to = date + datetime.timedelta(days=1)
    time_range = TimeRange(t_from, t_to)
    
    ml_data_df = calculate(
        cache_context=cache_context,
        time_range=time_range,
        feature_collection=feature_collection,
        target_params_batch=target_params_batch,
        resample_params=resample_params,
        seq_param=seq_param,
        )
    
    if ml_data_df is None or len(ml_data_df) == 0:
        logger.warning(f"No ML data available for {date}")
        return
    
    # Cache the data
    data_type = "sequential" if seq_param is not None else "regular"
    logger.info(f"Caching {data_type} ML data for {date}")
    
    params_dir = _get_ml_data_params_dir(resample_params, feature_collection, target_params_batch, seq_param)
    folder_path = cache_context.get_ml_data_path(params_dir)
    market_data.util.cache.write.split_and_cache_daily_df(
        df=ml_data_df,
        folder_path=folder_path,
        overwrite=overwrite_cache,
        warm_up_period_days=0,
    )
    
    
    logger.info(f"Successfully cached {data_type} ML data for {date} with {len(ml_data_df)} rows")


def calculate_and_cache_ml_data(
    cache_context: CacheContext,
    time_range: TimeRange,
    feature_collection: FeatureLabelCollection,
    target_params_batch: TargetParamsBatch = None,
    resample_params: ResampleParam = None,
    seq_param: Optional[SequentialFeatureParam] = None,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache ML data by preparing the data and caching it daily.
    
    Uses deterministic UUID-based caching for clean directory structure.
    Creates description.txt files with human-readable parameter information.
    """
    target_params_batch = target_params_batch or TargetParamsBatch()
    resample_params = resample_params or CumSumResampleParams()
    t_from, t_to = time_range.to_datetime()
    current_date = t_from
    
    params_dir = _get_ml_data_params_dir(resample_params, feature_collection, target_params_batch, seq_param)
    folder_path = cache_context.get_ml_data_path(params_dir)

    data_type = "sequential" if seq_param is not None else "regular"
    logger.info(f"Starting {data_type} ML data processing for {len(feature_collection.feature_labels)} features")
    logger.info(f"Cache UUID: {params_dir}")
    
    # Write description file (only if it doesn't exist to avoid overwriting)
    description_path = os.path.join(folder_path, "description.txt")
    print(f"{description_path=}")
    if not os.path.exists(description_path):
        write_description_file(folder_path, resample_params, feature_collection, target_params_batch, seq_param)
        logger.info(f"Created parameter description file: {description_path}")
    
    # Process each day
    while current_date < t_to:
        logger.info(f"Processing day {current_date}")
        
        _calculate_and_cache__daily_ml_data(
            date=current_date,
            cache_context=cache_context,
            feature_collection=feature_collection,
            target_params_batch=target_params_batch,
            resample_params=resample_params,
            seq_param=seq_param,
            overwrite_cache=overwrite_cache
        )

        current_date = anchor_to_begin_of_day(current_date + datetime.timedelta(days=1))


def _load_ml_data_at_params_dir(
    cache_context: CacheContext,
    time_range: TimeRange,
    params_dir: str,
    columns: Optional[List[str]] = None,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Load cached ML data at a specific params_dir.
    """
    def load(d_from, d_to):
        folder_path = cache_context.get_ml_data_path(params_dir)
        return d_from, market_data.util.cache.read.read_daily_from_local_cache(
                folder_path,
                d_from,
                d_to,
                columns=columns,
        )

    ml_data_df = read_multithreaded(
        read_func=load,
        time_range=time_range,
        max_workers=max_workers
    )
    # Log cache information for debugging
    logger.debug(f"Loaded ML data from UUID: {params_dir}")
    
    return ml_data_df.sort_values(["timestamp", "symbol"])
    

def load_cached_ml_data(
    cache_context: CacheContext,
    time_range: TimeRange,
    feature_collection: FeatureLabelCollection,
    target_params_batch: TargetParamsBatch = None,
    resample_params: ResampleParam = None,
    seq_param: Optional[SequentialFeatureParam] = None,
    columns: Optional[List[str]] = None,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Load cached ML data for a specific time range.
    
    Can load both regular and sequential ML data based on seq_param.
    Uses deterministic UUID-based caching for clean directory structure.
    Parameter descriptions are stored in description.txt files.
    
    Returns:
        DataFrame with ML data (regular or sequential based on seq_param),
        or empty DataFrame if no data is available
    """
    target_params_batch = target_params_batch or TargetParamsBatch()
    resample_params = resample_params or CumSumResampleParams()
    params_dir = _get_ml_data_params_dir(resample_params, feature_collection, target_params_batch, seq_param)

    return _load_ml_data_at_params_dir(cache_context, time_range, params_dir, columns, max_workers)


def load_cached_and_select_columns_ml_data(
    cache_context: CacheContext,
    time_range: TimeRange,
    feature_collection: FeatureLabelCollection,
    target_params_batch: TargetParamsBatch = None,
    resample_params: ResampleParam = None,
    seq_param: Optional[SequentialFeatureParam] = None,
    columns: Optional[List[str]] = None,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Load cached ML data that contains all requested features and select only the needed columns.
    
    This function finds a cached ML dataset that includes all the requested feature labels
    (possibly with additional features), loads it, and returns only the columns corresponding
    to the requested features.
    """
    target_params_batch = target_params_batch or TargetParamsBatch()
    resample_params = resample_params or CumSumResampleParams()

    # Find a cache folder containing all requested features
    params_dir = find_cached_ml_data_with_features(
        cache_context,
        feature_collection,
        target_params_batch,
        resample_params,
        seq_param
    )

    if params_dir is None:
        logger.error("No cached ML data found with all requested features. Falling back to exact match.")
        return pd.DataFrame()

    ml_data_df = _load_ml_data_at_params_dir(cache_context, time_range, params_dir, columns, max_workers)
    logger.info(f"Loaded ML data from matched cache UUID: {params_dir}")

    if ml_data_df is None or ml_data_df.empty:
        logger.warning("No ML data loaded")
        return pd.DataFrame()

    # Get columns for requested features
    requested_columns = []
    for feature_label_obj in feature_collection.feature_labels:
        feature_columns = feature_label_obj.params.get_columns()
        requested_columns.extend(feature_columns)

    # Ensure all requested columns exist
    missing_columns = [col for col in requested_columns if col not in ml_data_df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in ML data: {missing_columns}")

    # Remove duplicates
    result_columns = list(set(requested_columns))
    result_df = ml_data_df[result_columns]
    logger.info(f"Returning {len(result_columns)} columns from {len(ml_data_df.columns)} available columns")

    return result_df


def calculate_and_cache_and_load_ml_data(
    cache_context: CacheContext,
    time_range: TimeRange,
    feature_collection: FeatureLabelCollection,
    target_params_batch: TargetParamsBatch = None,
    resample_params: ResampleParam = None,
    seq_param: Optional[SequentialFeatureParam] = None,
    overwrite_cache: bool = True,
    columns: Optional[List[str]] = None,
    max_workers: int = 10,
) -> None:
    """
    Cache if needed and then load.
    """
    calculate_and_cache_ml_data(
        cache_context,
        time_range,
        feature_collection,
        target_params_batch = target_params_batch,
        resample_params = resample_params,
        seq_param = seq_param,
        overwrite_cache = overwrite_cache)

    return load_cached_ml_data(
        cache_context,
        time_range,
        feature_collection,
        target_params_batch = target_params_batch,
        resample_params = resample_params,
        seq_param = seq_param,
        columns = columns,
        max_workers = max_workers)
