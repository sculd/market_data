import pandas as pd
import logging
from typing import Optional, List, Any, Tuple, Union
import os
import datetime
from pathlib import Path
from dataclasses import asdict
import hashlib
import json

import market_data.target.cache
from market_data.target.calc import TargetParamsBatch
from market_data.feature.util import parse_feature_label_params
import market_data.ingest.common
from market_data.ingest.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, CacheContext
from market_data.util.time import TimeRange
from market_data.machine_learning.resample.calc import ResampleParams
from market_data.machine_learning.ml_data.calc import prepare_ml_data
from market_data.feature.impl.common import SequentialFeatureParam
from market_data.util.cache.time import (
    anchor_to_begin_of_day
)
from market_data.util.cache.parallel_processing import (
    read_multithreaded,
)
import market_data.util.cache.cache_common
import market_data.util.cache.cache_read
import market_data.util.cache.cache_write
from market_data.util.cache.path import get_cache_base_path

logger = logging.getLogger(__name__)


def get_local_cache_base_path():
    '''
    For ML data, use separate cache base.
    As the access to external disk via network is slow.
    '''
    base_path = os.environ.get('ML_DATA_LOCAL_CACHE_BASE', '~/algo_cache')
    return os.path.expanduser(base_path) 

# Global paths configuration - use configurable base path
CACHE_BASE_PATH = os.path.join(get_cache_base_path(), 'ml_data')
Path(CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

# local cache for ml data
LOCAL_CACHE_BASE_PATH = os.path.join(get_local_cache_base_path(), 'ml_data')
Path(LOCAL_CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

def _write_description_file(
    params_dir: str,
    resample_params: ResampleParams,
    feature_label_params: List[Tuple[str, Any]],
    target_params_batch: TargetParamsBatch,
    seq_params: Optional[SequentialFeatureParam] = None,
) -> None:
    """
    Write parameter description to description.txt file in the cache directory.
    """
    lines = []
    lines.append("ML Data Cache Parameters")
    lines.append("=" * 30)
    lines.append("")
    
    # Resample parameters
    lines.append("Resample Parameters:")
    for key, value in asdict(resample_params).items():
        lines.append(f"{key}: {value}")
    lines.append("")
    
    # Target parameters
    lines.append("Target Parameters:")
    lines.append(f"{market_data.target.cache._get_target_params_dir(target_params_batch)}")
    lines.append("")
    
    # Sequential parameters
    if seq_params is not None:
        lines.append("Sequential Parameters:")
        lines.append(f"{seq_params.get_params_dir()}")
        lines.append("")
    
    # Feature parameters
    lines.append("Feature Parameters:")
    for feature_label, param in sorted(feature_label_params, key=lambda x: x[0]):
        lines.append(f"{feature_label}: {param.get_params_dir()}")
    lines.append("")
    
    # Add timestamp
    lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
    
    description = "\n".join(lines)    
    description_path = os.path.join(params_dir, "description.txt")
    
    # Create directory if it doesn't exist
    os.makedirs(params_dir, exist_ok=True)
    
    # Write description file
    with open(description_path, 'w') as f:
        f.write(description)

def _generate_params_uuid(
    resample_params: ResampleParams,
    feature_label_params: List[Tuple[str, Any]],
    target_params_batch: TargetParamsBatch,
    seq_params: Optional[SequentialFeatureParam] = None,
) -> str:
    """
    Generate a deterministic UUID based on all ML parameters.
    """
    # Create a consistent dictionary for all parameters
    params_dict = {
        'resample': asdict(resample_params),
        'target': market_data.target.cache._get_target_params_dir(target_params_batch),
        'features': [],
        'sequential': None
    }
    
    # Add features in sorted order for consistency
    for feature_label, param in sorted(feature_label_params, key=lambda x: x[0]):
        params_dict['features'].append({
            'label': feature_label,
            'params': param.get_params_dir()
        })
    
    # Add sequential params if present
    if seq_params is not None:
        params_dict['sequential'] = seq_params.get_params_dir()
    
    # Convert to JSON string with sorted keys for consistency
    params_str = json.dumps(params_dict, sort_keys=True)
    # Use SHA256 and take first 12 characters for a short but unique identifier
    return hashlib.sha256(params_str.encode()).hexdigest()[:12]

def _get_mldata_params_dir(
    resample_params: ResampleParams,
    feature_label_params: List[Tuple[str, Any]],
    target_params_batch: TargetParamsBatch,
    seq_params: Optional[SequentialFeatureParam] = None,
) -> str:
    """
    Generate a clean directory path using deterministic UUID for ML data parameters.
    
    Instead of concatenating all parameter descriptions (which creates very long paths),
    this function generates a deterministic UUID based on all parameters and stores
    the parameter descriptions in a description.txt file.
    """
    # Generate deterministic UUID from all parameters
    params_uuid = _generate_params_uuid(resample_params, feature_label_params, target_params_batch, seq_params)
    
    # Return just the UUID - dataset_id is handled separately
    return params_uuid


def _calculate_daily_ml_data(
    date: datetime.datetime,
    cache_context: CacheContext,
    feature_label_params: Optional[List[Union[str, Tuple[str, Any]]]],
    target_params_batch: TargetParamsBatch,
    resample_params: ResampleParams,
    seq_params: Optional[SequentialFeatureParam] = None,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache ML data for a single day.
    
    Can handle both regular and sequential features based on seq_params.
    Creates a description.txt file with human-readable parameter information.
    
    Args:
        date: The date to calculate ML data for
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        feature_label_params: List of feature labels and their parameters
        target_params_batch: Target calculation parameters
        resample_params: Resampling parameters
        seq_params: Sequential feature parameters. If provided, creates sequential ML data.
        overwrite_cache: Whether to overwrite existing cache files
    """
    # Create time range for the specific day
    t_from = date
    t_to = date + datetime.timedelta(days=1)
    time_range = TimeRange(t_from, t_to)
    
    ml_data_df = prepare_ml_data(
        dataset_mode=cache_context.dataset_mode,
        export_mode=cache_context.export_mode,
        aggregation_mode=cache_context.aggregation_mode,
        time_range=time_range,
        feature_label_params=feature_label_params,
        target_params_batch=target_params_batch,
        resample_params=resample_params,
        seq_params=seq_params,
        )
    
    if ml_data_df is None or len(ml_data_df) == 0:
        logger.warning(f"No ML data available for {date}")
        return
    
    # Cache the data
    data_type = "sequential" if seq_params is not None else "regular"
    logger.info(f"Caching {data_type} ML data for {date}")
    
    params_dir = _get_mldata_params_dir(resample_params, feature_label_params, target_params_batch, seq_params)
    folder_path = cache_context.get_ml_data_path(params_dir)
    market_data.util.cache.cache_write.cache_locally_df(
        df=ml_data_df,
        folder_path=folder_path,
        overwrite=overwrite_cache,
        warm_up_period_days=0,
    )
    
    
    logger.info(f"Successfully cached {data_type} ML data for {date} with {len(ml_data_df)} rows")


def calculate_and_cache_ml_data(
    cache_context: CacheContext,
    time_range: TimeRange,
    feature_label_params: Optional[List[Union[str, Tuple[str, Any]]]] = None,
    target_params_batch: TargetParamsBatch = None,
    resample_params: ResampleParams = None,
    seq_params: Optional[SequentialFeatureParam] = None,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache ML data by preparing the data and caching it daily.
    
    Uses deterministic UUID-based caching for clean directory structure.
    Creates description.txt files with human-readable parameter information.
    
    This function:
    1. Splits the time range into individual days
    2. For each day, prepares ML data using prepare_ml_data (sequential or regular based on seq_params)
    3. Caches each daily piece in a UUID-based directory structure
    4. Creates a description.txt file with parameter details for easy debugging
    
    Args:
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        time_range: TimeRange object specifying the time range
        feature_label_params: List of feature labels and their parameters. If None, uses default parameters.
        target_params_batch: Target calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        seq_params: Sequential feature parameters. If provided, creates sequential ML data.
        overwrite_cache: Whether to overwrite existing cache files
    """
    feature_label_params = parse_feature_label_params(feature_label_params)
    target_params_batch = target_params_batch or TargetParamsBatch()
    resample_params = resample_params or ResampleParams()
    t_from, t_to = time_range.to_datetime()
    current_date = t_from
    
    params_dir = _get_mldata_params_dir(resample_params, feature_label_params, target_params_batch, seq_params)
    folder_path = cache_context.get_ml_data_path(params_dir)

    data_type = "sequential" if seq_params is not None else "regular"
    logger.info(f"Starting {data_type} ML data processing for {len(feature_label_params)} features")
    logger.info(f"Cache UUID: {params_dir}")
    
    # Write description file (only if it doesn't exist to avoid overwriting)
    description_path = os.path.join(folder_path, "description.txt")
    print(f"{description_path=}")
    if not os.path.exists(description_path):
        _write_description_file(folder_path, resample_params, feature_label_params, target_params_batch, seq_params)
        logger.info(f"Created parameter description file: {description_path}")
    
    # Process each day
    while current_date < t_to:
        logger.info(f"Processing day {current_date}")
        
        _calculate_daily_ml_data(
            date=current_date,
            cache_context=cache_context,
            feature_label_params=feature_label_params,
            target_params_batch=target_params_batch,
            resample_params=resample_params,
            seq_params=seq_params,
            overwrite_cache=overwrite_cache
        )

        current_date = anchor_to_begin_of_day(current_date + datetime.timedelta(days=1))


def load_cached_ml_data(
    cache_context: CacheContext,
    time_range: TimeRange,
    feature_label_params: Optional[List[Union[str, Tuple[str, Any]]]] = None,
    target_params_batch: TargetParamsBatch = None,
    resample_params: ResampleParams = None,
    seq_params: Optional[SequentialFeatureParam] = None,
    columns: Optional[List[str]] = None,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Load cached ML data for a specific time range.
    
    Can load both regular and sequential ML data based on seq_params.
    Uses deterministic UUID-based caching for clean directory structure.
    Parameter descriptions are stored in description.txt files.
    
    Args:
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        time_range: TimeRange object specifying the time range
        feature_label_params: List of feature labels and their parameters
        target_params_batch: Target calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        seq_params: Sequential feature parameters. If provided, loads sequential ML data.
        columns: Optional list of columns to load. If None, loads all columns.
        
    Returns:
        DataFrame with ML data (regular or sequential based on seq_params),
        or empty DataFrame if no data is available
    """
    feature_label_params = parse_feature_label_params(feature_label_params)
    target_params_batch = target_params_batch or TargetParamsBatch()
    resample_params = resample_params or ResampleParams()
    params_dir = _get_mldata_params_dir(resample_params, feature_label_params, target_params_batch, seq_params)

    def load(d_from, d_to):
        folder_path = cache_context.get_ml_data_path(params_dir)
        return d_from, market_data.util.cache.cache_read.read_daily_from_local_cache(
                folder_path,
                d_from = d_from,
                d_to = d_to,
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
