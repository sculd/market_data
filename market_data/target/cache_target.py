"""
Target caching module.

This module provides functions for calculating and caching targets.
"""

import pandas as pd
import logging
import os
import datetime
from pathlib import Path
from typing import Optional, List, Union, Tuple

from market_data.ingest.bq.cache import read_from_cache_or_query_and_cache
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from market_data.util.time import TimeRange
from market_data.target.target import create_targets, TargetParamsBatch
from market_data.util.cache.time import (
    split_t_range,
)
from market_data.util.cache.dataframe import (
    cache_data_by_day,
    read_from_cache_generic,
    cache_daily_df
)
from market_data.util.cache.path import (
    params_to_dir_name
)
from market_data.util.cache.core import calculate_and_cache_data

# Define the cache base path for targets
TARGET_CACHE_BASE_PATH = os.path.expanduser('~/algo_cache/feature_data')
# Ensure the directory exists
Path(TARGET_CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

def _get_target_params_dir(params: TargetParamsBatch = None) -> str:
    """Convert target parameters to a directory name string"""
    params = params or TargetParamsBatch()
    params_dict = {
        'fp': sorted(set([p.forward_period for p in params.target_params_list])),
        'tp': sorted(set([p.tp_value for p in params.target_params_list])),
        'sl': sorted(set([p.sl_value for p in params.target_params_list])),
    }
    return params_to_dir_name(params_dict)

def _get_recommended_warm_up_days(params: TargetParamsBatch) -> int:
    """
    Calculate the recommended warm-up period based on target parameters.
    
    Uses the maximum forward period plus a buffer to ensure sufficient
    historical data for all target calculations.
    
    Returns:
        int: Recommended number of warm-up days
    """
    # Find the maximum forward period
    max_forward = max(p.forward_period for p in params.target_params_list)
    
    # Convert to days (assuming periods are in minutes for 24/7 markets)
    # Add a small buffer of 2 days to be safe
    import math
    days_needed = math.ceil(max_forward / (24 * 60)) + 2
    
    # Ensure at least 3 days minimum
    return max(3, days_needed)

def calculate_and_cache_targets(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        params: TargetParamsBatch = None,
        time_range: TimeRange = None,
        calculation_batch_days: int = 1,
        warm_up_days: Optional[int] = None,
        overwrite_cache: bool = True,
        ) -> None:
    """
    Calculate and cache targets for a specified time range.
    
    Parameters:
    -----------
    dataset_mode : DATASET_MODE
        Dataset mode (LIVE, REPLAY, etc.)
    export_mode : EXPORT_MODE
        Export mode (OHLC, TICKS, etc.)
    aggregation_mode : AGGREGATION_MODE
        Aggregation mode (MIN_1, MIN_5, etc.)
    params : TargetParamsBatch, optional
        Target calculation parameters. If None, uses default parameters.
    time_range : TimeRange, optional
        Time range for calculation. If None, must provide individual time parameters.
    calculation_batch_days : int, optional
        Number of days to calculate for in each batch, default 1
    warm_up_days : int, optional
        Number of warm-up days for calculation, default None (auto-calculated)
    overwrite_cache : bool, optional
        If True, overwrite existing cache files, default True
    """
    # Create default params if None
    params = params or TargetParamsBatch()
    
    # Calculate warm-up days if not provided
    if warm_up_days is None:
        warm_up_days = _get_recommended_warm_up_days(params)
        logger.info(f"Using {warm_up_days} warm-up days for targets")
    
    # Get the params directory name
    params_dir = _get_target_params_dir(params)
    
    # Use the generic calculate_and_cache_data function
    calculate_and_cache_data(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        params=params,
        time_range=time_range,
        calculation_batch_days=calculation_batch_days,
        warm_up_days=warm_up_days,
        overwrite_cache=overwrite_cache,
        label="targets",
        calculate_batch_fn=create_targets,
        cache_base_path=TARGET_CACHE_BASE_PATH,
        params_dir=params_dir
    )

def load_cached_targets(
        params: TargetParamsBatch = None,
        time_range: TimeRange = None,
        columns: List[str] = None,
        dataset_mode: DATASET_MODE = None,
        export_mode: EXPORT_MODE = None,
        aggregation_mode: AGGREGATION_MODE = None
    ) -> pd.DataFrame:
    """
    Load cached targets for a specific time range

    Parameters:
    -----------
    params : TargetParamsBatch, optional
        Target calculation parameters. If None, uses default parameters.
    time_range : TimeRange, optional
        Time range for target calculation. If None, must provide individual time parameters.
    columns : List[str], optional
        Columns to load from cache. If None, all columns are loaded.
    dataset_mode : DATASET_MODE, optional
        Dataset mode for cache path. If None, uses default dataset mode.
    export_mode : EXPORT_MODE, optional
        Export mode for cache path. If None, uses default export mode.
    aggregation_mode : AGGREGATION_MODE, optional
        Aggregation mode for cache path. If None, uses default aggregation mode.
    """
    return read_from_cache_generic(
        'targets', params_dir=_get_target_params_dir(params), time_range=time_range, columns=columns,
        dataset_mode=dataset_mode, export_mode=export_mode, aggregation_mode=aggregation_mode,
        cache_base_path=TARGET_CACHE_BASE_PATH
    ) 