import pandas as pd
import logging
from typing import Optional, List, Any, Tuple, Union
import os
import datetime
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from market_data.util.cache.time import split_t_range
import market_data.target.cache_target
from market_data.target.target import TargetParamsBatch
from market_data.feature.util import parse_feature_label_params
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from market_data.util.time import TimeRange
from market_data.machine_learning.resample.resample import ResampleParams
from market_data.machine_learning.ml_data import prepare_ml_data
from market_data.feature.impl.common import SequentialFeatureParam
from market_data.util.cache.time import (
    anchor_to_begin_of_day
)
from market_data.util.cache.dataframe import (
    cache_data_by_day,
    read_from_local_cache,
    read_multithreaded,
)
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
    lines.append(f"{market_data.target.cache_target._get_target_params_dir(target_params_batch)}")
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
        'target': market_data.target.cache_target._get_target_params_dir(target_params_batch),
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
    
    # Return just the UUID - dataset_id is handled separately by cache_data_by_day
    return params_uuid


def _calculate_daily_ml_data(
    date: datetime.datetime,
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
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
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
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
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
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
    
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"
    params_dir = _get_mldata_params_dir(resample_params, feature_label_params, target_params_batch, seq_params)
    
    cache_data_by_day(
        df=ml_data_df,
        label="ml_data",
        t_from=t_from,
        t_to=t_to,
        params_dir=params_dir,
        overwrite=overwrite_cache,
        dataset_id=dataset_id,
        cache_base_path=CACHE_BASE_PATH,
        warm_up_period_days=0,
    )
    
    logger.info(f"Successfully cached {data_type} ML data for {date} with {len(ml_data_df)} rows")


def calculate_and_cache_ml_data(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
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
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
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
    
    data_type = "sequential" if seq_params is not None else "regular"
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"
    params_dir = _get_mldata_params_dir(resample_params, feature_label_params, target_params_batch, seq_params)
    
    logger.info(f"Starting {data_type} ML data processing for {len(feature_label_params)} features")
    logger.info(f"Cache UUID: {params_dir}")
    
    # Construct full cache path with dataset_id and UUID
    label="ml_data"
    full_cache_path = os.path.join(CACHE_BASE_PATH, label, dataset_id, params_dir)
    
    # Write description file (only if it doesn't exist to avoid overwriting)
    description_path = os.path.join(full_cache_path, "description.txt")
    print(f"{description_path=}")
    if not os.path.exists(description_path):
        _write_description_file(full_cache_path, resample_params, feature_label_params, target_params_batch, seq_params)
        logger.info(f"Created parameter description file: {description_path}")
    
    # Process each day
    while current_date < t_to:
        logger.info(f"Processing day {current_date}")
        
        _calculate_daily_ml_data(
            date=current_date,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            feature_label_params=feature_label_params,
            target_params_batch=target_params_batch,
            resample_params=resample_params,
            seq_params=seq_params,
            overwrite_cache=overwrite_cache
        )

        current_date = anchor_to_begin_of_day(current_date + datetime.timedelta(days=1))


def load_cached_ml_data(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
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
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
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
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"
    params_dir = _get_mldata_params_dir(resample_params, feature_label_params, target_params_batch, seq_params)

    def load(d_from, d_to):
        daily_time_range = TimeRange(d_from, d_to)
        return d_from, read_from_local_cache(
            label="ml_data",
            params_dir=params_dir,
            time_range=daily_time_range,
            columns=columns,
            dataset_id=dataset_id,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            local_cache_base_path=LOCAL_CACHE_BASE_PATH,
            global_cache_base_path=CACHE_BASE_PATH,
        )

    ml_data_df = read_multithreaded(
        read_func=load,
        time_range=time_range,
        max_workers=max_workers
    )
    
    # Log cache information for debugging
    logger.debug(f"Loaded ML data from UUID: {params_dir}")
    
    return ml_data_df.sort_values(["timestamp", "symbol"])
