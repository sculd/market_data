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

from market_data.feature.util import parse_feature_label_param
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from market_data.util.time import TimeRange
from market_data.machine_learning.resample import ResampleParams
from market_data.machine_learning.feature_resample import prepare_feature_resampled
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
CACHE_BASE_PATH = os.path.join(get_cache_base_path(), 'feature_resampled_data')
Path(CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

def _get_feature_resampled_params_dir(
    resample_params: ResampleParams,
    feature_label_params: Tuple[str, Any],
) -> str:
    """
    Convert feature resampled parameters to a directory path structure.
    
    Creates a nested directory structure for caching feature resampled data.
    
    Args:
        resample_params: Parameters for resampling
        feature_label_params: Single feature label and parameters tuple
        
    Returns:
        Path string with nested directories for parameters
    """
    # Start with base path for resample params
    base_path = params_to_dir_name({
        f'r_{key}': value for key, value in asdict(resample_params).items()
    })
    
    # Process the feature and its parameters
    feature_label, param = feature_label_params
    params_dir = param.get_params_dir()
    feature_dir = f"{feature_label},{params_dir}"
    
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
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache feature resampled data for a single day.
    
    Args:
        date: The date to calculate data for
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        feature_label: Name of the feature to process
        feature_params: Feature calculation parameters
        resample_params: Resampling parameters
        overwrite_cache: Whether to overwrite existing cache files
    """
    # Create time range for the specific day
    t_from = date
    t_to = date + datetime.timedelta(days=1)
    time_range = TimeRange(t_from, t_to)

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
    logger.info(f"Caching feature resampled data for {date}")
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"
    feature_label_params = parse_feature_label_param((feature_label, feature_params))
    params_dir = _get_feature_resampled_params_dir(resample_params, feature_label_params)
    
    cache_data_by_day(
        df=feature_resampled_df,
        label="feature_resampled",
        t_from=t_from,
        t_to=t_to,
        params_dir=params_dir,
        overwrite=overwrite_cache,
        dataset_id=dataset_id,
        cache_base_path=CACHE_BASE_PATH,
        warm_up_period_days=0,
    )
    
    logger.info(f"Successfully cached feature resampled data for {date} with {len(feature_resampled_df)} rows")


def calculate_and_cache_feature_resampled(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_label: str,
    feature_params: Any,
    resample_params: ResampleParams = None,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache feature resampled data by processing the data daily.
    
    This function:
    1. Splits the time range into individual days
    2. For each day, prepares feature resampled data using prepare_feature_resampled
    3. Caches each daily piece
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_label: Name of the feature to process
        feature_params: Feature calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        overwrite_cache: Whether to overwrite existing cache files
    """
    resample_params = resample_params or ResampleParams()
    t_from, t_to = time_range.to_datetime()
    current_date = t_from
    
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
            overwrite_cache=overwrite_cache
        )

        current_date = anchor_to_begin_of_day(current_date + datetime.timedelta(days=1))


def load_cached_feature_resampled(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_label: str,
    feature_params: Any,
    resample_params: ResampleParams = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load cached feature resampled data for a specific time range.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_label: Name of the feature to load
        feature_params: Feature calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        columns: Optional list of columns to load. If None, loads all columns.
        
    Returns:
        DataFrame with feature resampled data, or empty DataFrame if no data is available
    """
    resample_params = resample_params or ResampleParams()
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"
    feature_label_params = parse_feature_label_param((feature_label, feature_params))
    params_dir = _get_feature_resampled_params_dir(resample_params, feature_label_params)
    
    feature_resampled_df = read_from_cache_generic(
        label="feature_resampled",
        params_dir=params_dir,
        time_range=time_range,
        columns=columns,
        dataset_id=dataset_id,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        cache_base_path=CACHE_BASE_PATH
    )

    if feature_resampled_df.empty:
        return feature_resampled_df
    
    return feature_resampled_df.sort_values(["timestamp", "symbol"])
