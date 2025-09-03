import datetime
import hashlib
import json
import logging
import os
from dataclasses import asdict
from typing import List, Optional

import pandas as pd

import market_data.util.cache.read
import market_data.util.cache.write
from market_data.feature.registry import get_feature_by_label
from market_data.ingest.common import CacheContext
from market_data.machine_learning.ml_data.calc import calculate
from market_data.machine_learning.ml_data.param import MlDataParam
from market_data.util.cache.parallel_processing import read_multithreaded
from market_data.util.cache.time import anchor_to_begin_of_day
from market_data.util.time import TimeRange

logger = logging.getLogger(__name__)


def _generate_params_uuid(ml_data_param: MlDataParam) -> str:
    """
    Generate a deterministic UUID based on all ML parameters.
    """
    # Create a consistent dictionary for all parameters
    params_dict = {
        'resample': asdict(ml_data_param.resample_params),
        'target': ml_data_param.target_params_batch.get_params_dir(),
        'features': [],
        'sequential': None
    }
    
    # Add features in sorted order for consistency
    for feature_label_obj in sorted(ml_data_param.feature_collection.feature_labels, key=lambda x: x.feature_label):
        params_dict['features'].append({
            'label': feature_label_obj.feature_label,
            'params': feature_label_obj.params.get_params_dir()
        })
    
    # Add sequential params if present
    if ml_data_param.seq_param is not None:
        params_dict['sequential'] = ml_data_param.seq_param.get_params_dir()
    
    # Convert to JSON string with sorted keys for consistency
    params_str = json.dumps(params_dict, sort_keys=True)
    # Use SHA256 and take first 12 characters for a short but unique identifier
    return hashlib.sha256(params_str.encode()).hexdigest()[:12]


def _get_ml_data_params_dir(ml_data_param: MlDataParam) -> str:
    """
    Use uuid for params_dir.
    """
    return _generate_params_uuid(ml_data_param)


def _calculate_and_cache__daily_ml_data(
    date: datetime.datetime,
    cache_context: CacheContext,
    ml_data_param: MlDataParam,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache ML data for a single day.
    
    Can handle both regular and sequential features based on seq_param.
    Creates a description.txt file with human-readable parameter information.
    
    Args:
        date: The date to calculate ML data for
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        ml_data_param: MlDataParam object containing all ML parameters
        overwrite_cache: Whether to overwrite existing cache files
    """
    # Create time range for the specific day
    t_from = date
    t_to = date + datetime.timedelta(days=1)
    time_range = TimeRange(t_from, t_to)
    
    ml_data_df = calculate(
        cache_context=cache_context,
        time_range=time_range,
        ml_data_param=ml_data_param,
    )
    
    if ml_data_df is None or len(ml_data_df) == 0:
        logger.warning(f"No ML data available for {date}")
        return
    
    # Cache the data
    data_type = "sequential" if ml_data_param.seq_param is not None else "regular"
    logger.info(f"Caching {data_type} ML data for {date}")
    
    params_dir = _get_ml_data_params_dir(ml_data_param)
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
    ml_data_param: MlDataParam,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache ML data by preparing the data and caching it daily.
    
    Uses deterministic UUID-based caching for clean directory structure.
    Creates description.txt files with human-readable parameter information.
    """
    t_from, t_to = time_range.to_datetime()
    current_date = t_from
    
    params_dir = _get_ml_data_params_dir(ml_data_param)
    folder_path = cache_context.get_ml_data_path(params_dir)

    data_type = "sequential" if ml_data_param.seq_param is not None else "regular"
    logger.info(f"Starting {data_type} ML data processing for {len(ml_data_param.feature_collection.feature_labels)} features")
    logger.info(f"Cache UUID: {params_dir}")
    
    # Write description file (only if it doesn't exist to avoid overwriting)
    description_path = os.path.join(folder_path, "description.txt")
    print(f"{description_path=}")
    if not os.path.exists(description_path):
        ml_data_param.write_description_file(folder_path)
        logger.info(f"Created parameter description file: {description_path}")
    
    # Process each day
    while current_date < t_to:
        logger.info(f"Processing day {current_date}")
        
        _calculate_and_cache__daily_ml_data(
            date=current_date,
            cache_context=cache_context,
            ml_data_param=ml_data_param,
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
    

def load_cached_exact_spec_ml_data(
    cache_context: CacheContext,
    time_range: TimeRange,
    ml_data_param: MlDataParam,
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
    params_dir = _get_ml_data_params_dir(ml_data_param)
    return _load_ml_data_at_params_dir(cache_context, time_range, params_dir, columns, max_workers)


def load_cached_and_select_columns_ml_data(
    cache_context: CacheContext,
    time_range: TimeRange,
    ml_data_param: MlDataParam,
    columns: Optional[List[str]] = None,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Load cached ML data that contains all requested features and select only the needed columns.
    
    This function finds a cached ML dataset that includes all the requested feature labels
    (possibly with additional features), loads it, and returns only the columns corresponding
    to the requested features.
    """
    # Find a cache folder containing all requested features
    params_dir = ml_data_param.find_cached_ml_data_with_features(cache_context)

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
    for feature_label_obj in ml_data_param.feature_collection.feature_labels:
        # Get feature class
        feature_cls = get_feature_by_label(feature_label_obj.feature_label)
        if not feature_cls:
            raise ValueError(f"Feature class not found for label: {feature_label_obj.feature_label}")
            
        feature_columns = feature_cls.get_columns()
        requested_columns.extend(feature_columns)

    # Ensure all requested columns exist
    missing_columns = [col for col in requested_columns if col not in ml_data_df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in ML data: {missing_columns}")

    label_columns = [col for col in ml_data_df.columns if col.startswith("label_")]

    # Remove duplicates
    result_columns = list(set(requested_columns))
    result_df = ml_data_df[result_columns + label_columns + ml_data_param.resample_columns]
    logger.info(f"Returning {len(result_columns)} df from {len(ml_data_df.columns)} available columns")

    return result_df


def load_cached_ml_data(
    cache_context: CacheContext,
    time_range: TimeRange,
    ml_data_param: MlDataParam,
    columns: Optional[List[str]] = None,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    load_cached_and_select_columns_ml_data.
    """
    return load_cached_and_select_columns_ml_data(
        cache_context=cache_context,
        time_range=time_range,
        ml_data_param=ml_data_param,
        columns=columns,
        max_workers=max_workers
    )
