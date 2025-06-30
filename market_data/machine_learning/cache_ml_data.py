import pandas as pd
import logging
from typing import Optional, List, Any, Tuple, Union
import os
import datetime
import math
from pathlib import Path
from dataclasses import asdict
import numpy as np
from dataclasses import dataclass, field

import market_data.target.cache_target
from market_data.target.target import TargetParamsBatch
from market_data.feature.util import parse_feature_label_params
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from market_data.util.time import TimeRange
from market_data.machine_learning.resample import ResampleParams
from market_data.machine_learning.ml_data import prepare_ml_data, prepare_sequential_ml_data
from market_data.feature.impl.common import SequentialFeatureParam
from market_data.feature.sequential_feature import sequentialize_feature
from market_data.util.cache.time import (
    anchor_to_begin_of_day
)
from market_data.util.cache.dataframe import (
    cache_data_by_day,
    read_from_cache_generic,
)
from market_data.util.cache.path import (
    params_to_dir_name,
    get_cache_base_path,
)
from market_data.util.cache.core import calculate_and_cache_data

logger = logging.getLogger(__name__)

# Global paths configuration - use configurable base path
CACHE_BASE_PATH = os.path.join(get_cache_base_path(), 'ml_data')
Path(CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

def _get_mldata_params_dir(
    resample_params: ResampleParams,
    feature_label_params: List[Tuple[str, Any]],
    target_params_batch: TargetParamsBatch
) -> str:
    """
    Convert all ML data parameters to a directory path structure.
    
    Creates a nested directory structure with one directory per feature parameter
    to avoid hitting the 256-character filename limit in macOS.
    
    Args:
        resample_params: Parameters for resampling
        feature_label_params: List of feature label parameters
        target_params: Parameters for target calculation
        
    Returns:
        Path string with nested directories for parameters
    """
    # Start with base path for resample params
    resample_dir = params_to_dir_name({
        f'r_{key}': value for key, value in asdict(resample_params).items()
    })
    
    # Create a directory for target params
    target_dir = f't_{market_data.target.cache_target._get_target_params_dir(target_params_batch)}'
    
    # Combine base directories
    base_path = os.path.join(resample_dir, target_dir)
    
    # If no feature_label_params, return just the base path
    if not feature_label_params:
        return base_path
    
    # Process each feature and its parameters to create nested directories
    feature_paths = []
    for feature_label, param in feature_label_params:
        # Get the params directory name from the instance's method
        params_dir = param.get_params_dir()
        feature_dir = f"{feature_label},{params_dir}"
        feature_paths.append(feature_dir)
    
    # Sort feature paths for consistency
    feature_paths.sort()
    
    # Create a nested path by joining all feature paths
    nested_path = base_path
    for feature_path in feature_paths:
        nested_path = os.path.join(nested_path, feature_path)
    
    return nested_path


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
    
    Args:
        date: The date to calculate ML data for
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        feature_params: Feature calculation parameters
        target_params: Target calculation parameters
        resample_params: Resampling parameters
        overwrite_cache: Whether to overwrite existing cache files
    """
    # Create time range for the specific day
    t_from = date
    t_to = date + datetime.timedelta(days=1)
    time_range = TimeRange(t_from, t_to)
    
    # Prepare ML data for the day
    if seq_params is not None:
        ml_data_df = prepare_sequential_ml_data(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            feature_label_params=feature_label_params,
            target_params=target_params_batch,
            resample_params=resample_params,
            seq_params=seq_params,
        )
    else:
        ml_data_df = prepare_ml_data(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            feature_label_params=feature_label_params,
            target_params_batch=target_params_batch,
            resample_params=resample_params
        )
    
    if ml_data_df is None or len(ml_data_df) == 0:
        logger.warning(f"No ML data available for {date}")
        return
    
    # Cache the data
    logger.info(f"Caching ML data for {date}")
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"
    params_dir = _get_mldata_params_dir(resample_params, feature_label_params, target_params_batch)
    
    if seq_params is not None:
        params_dir = os.path.join(seq_params.get_params_dir(), params_dir)
    
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
    
    logger.info(f"Successfully cached ML data for {date} with {len(ml_data_df)} rows")


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
    
    This function:
    1. Splits the time range into individual days
    2. For each day, prepares ML data using prepare_ml_data
    3. Caches each daily piece
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_params: Feature calculation parameters. If None, uses default parameters.
        target_params: Target calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        overwrite_cache: Whether to overwrite existing cache files
    """
    feature_label_params = parse_feature_label_params(feature_label_params)
    target_params_batch = target_params_batch or TargetParamsBatch()
    resample_params = resample_params or ResampleParams()
    t_from, t_to = time_range.to_datetime()
    current_date = t_from
    
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
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load cached ML data for a specific time range.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_label_params: List of feature labels and their parameters
        target_params: Target calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        seq_params: Sequential feature parameters. If None, loads regular non-sequential data.
                    If provided, loads sequential data cached with these parameters.
        columns: Optional list of columns to load. If None, loads all columns.
        
    Returns:
        DataFrame with ML data, or empty DataFrame if no data is available
    """
    feature_label_params = parse_feature_label_params(feature_label_params)
    target_params_batch = target_params_batch or TargetParamsBatch()
    resample_params = resample_params or ResampleParams()
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"
    params_dir = _get_mldata_params_dir(resample_params, feature_label_params, target_params_batch)
    
    if seq_params is not None:
        params_dir = os.path.join(seq_params.get_params_dir(), params_dir)
    
    ml_data_df = read_from_cache_generic(
        label="ml_data",
        params_dir=params_dir,
        time_range=time_range,
        columns=columns,
        dataset_id=dataset_id,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        cache_base_path=CACHE_BASE_PATH
    )

    if ml_data_df.empty:
        return ml_data_df
    
    return ml_data_df.sort_values(["timestamp", "symbol"])
