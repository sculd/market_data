import pandas as pd
import logging
from typing import Optional, List, Any, Tuple, Union
import os
import datetime
import math
from pathlib import Path
from dataclasses import asdict

from market_data.feature.util import parse_feature_label_param
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from market_data.util.time import TimeRange
from market_data.machine_learning.resample.resample import ResampleParams
from market_data.machine_learning.feature_resample import prepare_feature_resampled, prepare_sequential_feature_resampled
from market_data.feature.impl.common import SequentialFeatureParam
from market_data.feature.sequential_feature import sequentialize_feature
from market_data.util.cache.time import (
    anchor_to_begin_of_day
)
from market_data.util.cache.dataframe import (
    cache_data_by_day,
    read_from_cache_generic,
    read_multithreaded,
)
from market_data.util.cache.path import (
    params_to_dir_name,
    get_cache_base_path,
)

logger = logging.getLogger(__name__)

# Global paths configuration - use configurable base path
CACHE_BASE_PATH = os.path.join(get_cache_base_path(), 'feature_data', 'feature_resampled')
Path(CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

def _get_feature_resampled_params_dir(
    resample_params: ResampleParams,
    feature_label_params: Tuple[str, Any],
    seq_params: Optional[SequentialFeatureParam] = None,
) -> str:
    """
    Convert feature resampled parameters to a directory path structure.
    
    Creates a nested directory structure for caching feature resampled data.
    Sequential and non-sequential data are cached separately.
    
    Args:
        resample_params: Parameters for resampling
        feature_label_params: Single feature label and parameters tuple
        seq_params: Sequential feature parameters. If provided, creates separate cache path.
        
    Returns:
        Path string with nested directories for parameters
    """
    # Start with base path for resample params
    base_path = params_to_dir_name({
        f'r_{key}': value for key, value in asdict(resample_params).items()
    })
    
    # Add sequential params if provided
    if seq_params is not None:
        seq_dir = seq_params.get_params_dir()
        base_path = os.path.join(base_path, f"seq_{seq_dir}")
    
    # Process the feature and its parameters
    _, param = feature_label_params
    params_dir = param.get_params_dir()
    feature_dir = f"{params_dir}"
    
    # Create nested path
    nested_path = os.path.join(base_path, feature_dir)
    
    return nested_path


def _calculate_and_cache_daily_feature_resampled(
    date: datetime.datetime,
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    feature_label: str,
    feature_params: Any,
    resample_params: ResampleParams,
    seq_params: Optional[SequentialFeatureParam] = None,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache feature resampled data for a single day.
    
    Can handle both regular and sequential features based on seq_params.
    
    Args:
        date: The date to calculate data for
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        feature_label: Name of the feature to process
        feature_params: Feature calculation parameters
        resample_params: Resampling parameters
        seq_params: Sequential feature parameters. If provided, creates sequential features.
        overwrite_cache: Whether to overwrite existing cache files
    """
    # Create time range for the specific day
    t_from = date
    t_to = date + datetime.timedelta(days=1)
    time_range = TimeRange(t_from, t_to)

    # Prepare feature data for the day (sequential or regular)
    if seq_params is not None:
        feature_resampled_df = prepare_sequential_feature_resampled(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            feature_label=feature_label,
            feature_params=feature_params,
            resample_params=resample_params,
            seq_params=seq_params,
        )
    else:
        feature_resampled_df = prepare_feature_resampled(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            feature_label=feature_label,
            feature_params=feature_params,
            resample_params=resample_params
        )
        
    if feature_resampled_df is None or len(feature_resampled_df) == 0:
        logger.warning(f"No feature resampled data available for {date}")
        return
    
    # Cache the data
    data_type = "sequential" if seq_params is not None else "regular"
    logger.info(f"Caching {data_type} feature resampled data for {date}")
    
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"
    feature_label_params = parse_feature_label_param((feature_label, feature_params))
    params_dir = _get_feature_resampled_params_dir(resample_params, feature_label_params, seq_params)
    
    cache_data_by_day(
        df=feature_resampled_df,
        label=feature_label,
        t_from=t_from,
        t_to=t_to,
        params_dir=params_dir,
        overwrite=overwrite_cache,
        dataset_id=dataset_id,
        cache_base_path=CACHE_BASE_PATH,
        warm_up_period_days=0,
    )
    
    logger.info(f"Successfully cached {data_type} feature resampled data for {date} with {len(feature_resampled_df)} rows")


def calculate_and_cache_feature_resampled(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_label: str,
    feature_params: Any,
    resample_params: ResampleParams = None,
    seq_params: Optional[SequentialFeatureParam] = None,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache feature resampled data by processing the data daily.
    
    This function:
    1. Splits the time range into individual days
    2. For each day, prepares feature resampled data (sequential or regular based on seq_params)
    3. Caches each daily piece separately for sequential vs regular features
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_label: Name of the feature to process
        feature_params: Feature calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        seq_params: Sequential feature parameters. If provided, creates sequential features.
        overwrite_cache: Whether to overwrite existing cache files
    """
    resample_params = resample_params or ResampleParams()
    t_from, t_to = time_range.to_datetime()
    current_date = t_from
    
    data_type = "sequential" if seq_params is not None else "regular"
    logger.info(f"Starting {data_type} feature resampled data processing for {feature_label}")
    
    try:
        # Process each day
        while current_date < t_to:
            logger.info(f"Processing day {current_date}")
            
            _calculate_and_cache_daily_feature_resampled(
                date=current_date,
                dataset_mode=dataset_mode,
                export_mode=export_mode,
                aggregation_mode=aggregation_mode,
                feature_label=feature_label,
                feature_params=feature_params,
                resample_params=resample_params,
                seq_params=seq_params,
                overwrite_cache=overwrite_cache
            )

            current_date = anchor_to_begin_of_day(current_date + datetime.timedelta(days=1))

        logger.info(f"Successfully cached {feature_label} resampled for {time_range}")
        return True
    except Exception as e:
        logger.error(f"[cache_writer] Error calculating/caching {feature_label} resampled: {e}")
        return False


def load_cached_feature_resampled(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_label: str,
    feature_params: Any,
    resample_params: ResampleParams = None,
    seq_params: Optional[SequentialFeatureParam] = None,
    columns: Optional[List[str]] = None,
        max_workers: int = 10,
) -> pd.DataFrame:
    """
    Load cached feature resampled data for a specific time range.
    
    Can load both regular and sequential feature data based on seq_params.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_label: Name of the feature to load
        feature_params: Feature calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        seq_params: Sequential feature parameters. If provided, loads sequential features.
        columns: Optional list of columns to load. If None, loads all columns.
        
    Returns:
        DataFrame with feature resampled data (regular or sequential based on seq_params),
        or empty DataFrame if no data is available
    """
    resample_params = resample_params or ResampleParams()
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"
    feature_label_params = parse_feature_label_param((feature_label, feature_params))
    params_dir = _get_feature_resampled_params_dir(resample_params, feature_label_params, seq_params)
    
    # Create worker function that properly handles daily ranges
    def load(d_from, d_to):
        daily_time_range = TimeRange(d_from, d_to)
        df = read_from_cache_generic(
            label=feature_label,
            params_dir=params_dir,
            time_range=daily_time_range,
            columns=columns,
            dataset_id=dataset_id,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            cache_base_path=CACHE_BASE_PATH
        )
        return d_from, df

    feature_resampled_df = read_multithreaded(
        read_func=load,
        time_range=time_range,
        max_workers=max_workers
    )
    
    return feature_resampled_df.sort_values(["timestamp", "symbol"])
