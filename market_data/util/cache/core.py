"""
Core caching functions for reading and writing data.

This module provides the fundamental functions for caching data,
including reading from cache, querying data, and writing to cache.
"""

import pandas as pd
import logging
import os
import typing
from typing import Optional, List, Union, Tuple, Callable, TypeVar, Any
import datetime
import warnings

from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from market_data.util.time import TimeRange
from market_data.util.cache.time import split_t_range, anchor_to_begin_of_day
from market_data.util.cache.dataframe import TIMESTAMP_INDEX_NAME, cache_daily_df, fetch_from_daily_cache, cache_data_by_day
from market_data.ingest.bq.cache import read_from_cache_or_query_and_cache

logger = logging.getLogger(__name__)

# Type variable for the params
T = TypeVar('T')

def calculate_and_cache_data(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        params: T,
        time_range: TimeRange,
        calculation_batch_days: int,
        warm_up_days: int,
        overwrite_cache: bool = True,
        label: str = None,
        calculate_batch_fn: Callable[[pd.DataFrame, T], pd.DataFrame] = None,
        cache_base_path: str = None,
        ) -> None:
    """
    Generic function to calculate and cache data for a specified time range.
    
    1. Seeks raw data files for the range
    2. Caches raw data if not present
    3. Calculates data in batches
    4. Caches results daily
    
    Parameters:
    -----------
    dataset_mode : DATASET_MODE
        Dataset mode (LIVE, REPLAY, etc.)
    export_mode : EXPORT_MODE
        Export mode (OHLC, TICKS, etc.)
    aggregation_mode : AGGREGATION_MODE
        Aggregation mode (MIN_1, MIN_5, etc.)
    params : T
        Calculation parameters
    time_range : TimeRange
        Time range for calculation
    calculation_batch_days : int
        Number of days to calculate for in each batch
    warm_up_days : int
        Number of warm-up days for calculation
    overwrite_cache : bool, optional
        If True, overwrite existing cache files, default True
    label : str, optional
        Label for the cache (e.g., "features", "targets")
    calculate_batch_fn : Callable[[pd.DataFrame, T], pd.DataFrame], optional
        Function to calculate data for a batch
    cache_base_path : str, optional
        Base path for caching data
    """
    # Resolve time range
    t_from, t_to = time_range.to_datetime()
    
    # Get dataset ID for cache path - use get_full_table_id from common.py
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{aggregation_mode}"
    
    # Set up calculation parameters
    if calculation_batch_days <= 0:
        calculation_batch_days = 1
    calculation_interval = datetime.timedelta(days=calculation_batch_days)
    warm_up_period = datetime.timedelta(days=warm_up_days)
    
    # Split the range into calculation batches, with warm-up included in each batch
    calculation_ranges = split_t_range(t_from, t_to, interval=calculation_interval, warm_up=warm_up_period)
    
    for calc_range in calculation_ranges:
        calc_t_from, calc_t_to = calc_range
        logger.info(f"Processing {label} calculation batch {calc_t_from} to {calc_t_to}")
        
        # Get raw data (fetch and cache if not present)
        raw_df = read_from_cache_or_query_and_cache(
            dataset_mode, export_mode, aggregation_mode,
            t_from=calc_t_from, t_to=calc_t_to,
            overwirte_cache=overwrite_cache
        )
        
        if raw_df is None or len(raw_df) == 0:
            logger.warning(f"No raw data available for {calc_t_from} to {calc_t_to}")
            continue
            
        # Calculate data for this batch
        try:
            result_df = calculate_batch_fn(raw_df, params)
            
            if result_df is None or len(result_df) == 0:
                logger.warning(f"{label} calculation returned empty result for {calc_t_from} to {calc_t_to}")
                continue
                
            # Determine if this is a warm-up batch or contains warm-up data
            is_warmup_batch = calc_range == calculation_ranges[0]  # First range doesn't have warm-up
            
            # For the first batch (without warm-up), use the regular warm-up period
            # For subsequent batches, we need to filter out the warm-up data from caching
            if is_warmup_batch:
                # First range doesn't include built-in warm-up, but we do need to skip some initial days
                warm_up_period_days = warm_up_days
            else:
                # For batches with built-in warm-up, calculate how many days from the start are warm-up data
                # warm_up_days is enough because warm_up is aligned to day boundaries
                warm_up_period_days = warm_up_days
                
            # Use cache_data_by_day instead of cache_daily_df to handle multiple days properly
            
            # Get param directory based on the params object
            if hasattr(params, 'forward_periods'):
                # It's a TargetParams
                from market_data.feature.cache_target import _get_target_params_dir
                params_dir = _get_target_params_dir(params)
            elif hasattr(params, 'return_periods'):
                # It's a FeatureParams
                from market_data.feature.cache_feature import _get_feature_params_dir
                params_dir = _get_feature_params_dir(params)
            else:
                # Generic params
                from market_data.util.cache.path import params_to_dir_name
                params_dict = params.__dict__ if hasattr(params, '__dict__') else {}
                params_dir = params_to_dir_name(params_dict)
            
            cache_data_by_day(
                result_df, label,
                calc_t_from, calc_t_to,
                params_dir=params_dir,
                overwrite=overwrite_cache,
                warm_up_period_days=warm_up_period_days,
                dataset_id=dataset_id,
                cache_base_path=cache_base_path
            )
            
        except Exception as e:
            logger.error(f"Error calculating {label} for {calc_t_from} to {calc_t_to}: {e}")
            continue
