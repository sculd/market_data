"""
Target caching module.

This module provides functions for calculating and caching targets.
"""

import pandas as pd
import logging
import os
from pathlib import Path
from dataclasses import asdict
from typing import Optional, List

from market_data.ingest.common import CacheContext
from market_data.util.time import TimeRange
from market_data.target.calc import create_targets, TargetParamsBatch
from market_data.util.cache.parallel_processing import (
    read_multithreaded,
)
import market_data.util.cache.cache_common
import market_data.util.cache.cache_read
import market_data.util.cache.cache_write
from market_data.util.cache.path import (
    params_to_dir_name,
    get_cache_base_path
)

# Global paths configuration - use configurable base path
TARGET_CACHE_BASE_PATH = os.path.join(get_cache_base_path(), 'feature_data')
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
        cache_context: CacheContext,
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
    cache_context : CacheContext
        Cache context containing dataset_mode, export_mode, aggregation_mode
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
    raw_data_folder_path = cache_context.get_market_data_path()
    folder_path = cache_context.get_target_path(params_dir)

    market_data.util.cache.cache_write.calculate_and_cache_data(
        raw_data_folder_path=raw_data_folder_path,
        folder_path=folder_path,
        params=params,
        time_range=time_range,
        calculation_batch_days=calculation_batch_days,
        warm_up_days=warm_up_days,
        overwrite_cache=overwrite_cache,
        calculate_batch_fn=create_targets,
    )

def load_cached_targets(
        cache_context: CacheContext,
        params: TargetParamsBatch = None,
        time_range: TimeRange = None,
        columns: List[str] = None,
        max_workers: int = 10,
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
    cache_context : CacheContext
        Cache context containing dataset_mode, export_mode, aggregation_mode
    """
    def load(d_from, d_to):
        params_dir = params_to_dir_name(asdict(params or TargetParamsBatch()))
        folder_path = cache_context.get_target_path(params_dir)
        df = market_data.util.cache.cache_read.read_daily_from_local_cache(
                folder_path,
                d_from = d_from,
                d_to = d_to,
        )        
        return d_from, df

    return read_multithreaded(
        read_func=load,
        time_range=time_range,
        max_workers=max_workers
    )