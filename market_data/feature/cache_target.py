import pandas as pd
import datetime
import logging
import typing
import math
import os
from pathlib import Path

from market_data.ingest.bq.cache import read_from_cache_or_query_and_cache
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.ingest.bq.common import get_full_table_id
from market_data.util.time import TimeRange
from market_data.feature.target import create_targets, TargetParams
from market_data.util.cache.time import (
    split_t_range,
)
from market_data.util.cache.dataframe import (
    cache_data_by_day,
    read_from_cache_generic,
)
from market_data.util.cache.path import (
    params_to_dir_name
)

# Define the cache base path for targets
TARGET_CACHE_BASE_PATH = os.path.expanduser('~/algo_cache/feature_data')
# Ensure the directory exists
Path(TARGET_CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

def _get_target_params_dir(params: TargetParams = None) -> str:
    params = params or TargetParams()
    params_dict = {
        'fp': params.forward_periods,
        'tp': params.tp_values,
        'sl': params.sl_values
    }
    return params_to_dir_name(params_dict)

def _get_recommended_warm_up_days(params: TargetParams = None) -> int:
    """
    Calculate the recommended warm-up period based on target parameters.
    
    Uses the maximum forward period plus a buffer to ensure sufficient
    historical data for all target calculations.
    
    Returns:
        int: Recommended number of warm-up days
    """
    params = params or TargetParams()
    
    # Find the maximum forward period
    max_forward = max(params.forward_periods)
    
    # Convert to days (assuming periods are in minutes for 24/7 markets)
    # Add a small buffer of 2 days to be safe
    days_needed = math.ceil(max_forward / (24 * 60)) + 2
    
    # Ensure at least 3 days minimum
    return max(3, days_needed)

def _cache_targets(df: pd.DataFrame, label: str, t_from: datetime.datetime, t_to: datetime.datetime,
                 params: TargetParams = None, overwrite=False, warm_up_period_days=1, dataset_id=None) -> None:
    """Cache target features for a specific time range"""
    params_dir = _get_target_params_dir(params)
    
    cache_data_by_day(
        df, label, t_from, t_to, params_dir, overwrite, warm_up_period_days, dataset_id,
        cache_base_path=TARGET_CACHE_BASE_PATH
    )

def _calculate_target_batch(raw_df: pd.DataFrame, params: TargetParams = None) -> pd.DataFrame:
    """Calculate targets for a batch of data using pandas implementation"""
    params = params or TargetParams()
    
    # Ensure timestamp is proper datetime
    if 'timestamp' in raw_df.columns and not pd.api.types.is_datetime64_any_dtype(raw_df['timestamp']):
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    
    # Use the pandas implementation to calculate targets
    return create_targets(raw_df, params=params)

def calculate_and_cache_targets(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        params: TargetParams = None,
        time_range: TimeRange = None,
        calculation_batch_days: int = 1,
        warm_up_days: int = None,  # Now optional with None as default
        overwrite_cache: bool = True,
    ) -> None:
    """
    Calculate and cache targets for a specified time range.
    
    1. Seeks raw data files for the range
    2. Caches raw data if not present
    3. Calculates targets in batches
    4. Caches target results daily
    
    Parameters:
    -----------
    dataset_mode : DATASET_MODE, optional
        Dataset mode for cache path. If None, uses default dataset mode.
    export_mode : EXPORT_MODE, optional
        Export mode for cache path. If None, uses default export mode.
    aggregation_mode : AGGREGATION_MODE, optional
        Aggregation mode for cache path. If None, uses default aggregation mode.
    params : TargetParams, optional
        Target calculation parameters. If None, uses default parameters.
    time_range : TimeRange, optional
        Time range for target calculation. If None, must provide individual time parameters.
    calculation_batch_days : int, optional
        Number of days to calculate targets for in each batch.
    warm_up_days : int, optional
        Number of warm-up days for target calculation. If None, 
        automatically determined based on the maximum forward period.
    overwrite_cache : bool, optional
        If True, overwrite existing cache files. If False, skip cache files that already exist.
    """
    params = params or TargetParams()
    
    # If warm_up_days not provided, calculate based on target parameters
    if warm_up_days is None:
        warm_up_days = _get_recommended_warm_up_days(params)
        logging.info(f"Using auto-calculated warm-up period of {warm_up_days} days based on target parameters")
    
    # Resolve time range
    t_from, t_to = time_range.to_datetime() if time_range else (None, None)
    
    # Create label for target cache
    target_label = "targets"
    
    # Get dataset ID for cache path
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
        logging.info(f"Processing target calculation batch {calc_t_from} to {calc_t_to}")
        
        # Get raw data (fetch and cache if not present)
        raw_df = read_from_cache_or_query_and_cache(
            dataset_mode, export_mode, aggregation_mode,
            t_from=calc_t_from, t_to=calc_t_to,
            overwirte_cache=overwrite_cache
        )
        
        if raw_df is None or len(raw_df) == 0:
            logging.warning(f"No raw data available for {calc_t_from} to {calc_t_to}")
            continue
            
        # Calculate targets for this batch
        try:
            targets_df = _calculate_target_batch(raw_df, params)
            
            if targets_df is None or len(targets_df) == 0:
                logging.warning(f"Target calculation returned empty result for {calc_t_from} to {calc_t_to}")
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
            
            _cache_targets(
                targets_df, target_label, 
                calc_t_from, calc_t_to,
                params=params,
                overwrite=overwrite_cache,
                warm_up_period_days=warm_up_period_days,
                dataset_id=dataset_id
            )
            
        except Exception as e:
            logging.error(f"Error calculating targets for {calc_t_from} to {calc_t_to}: {e}")
            continue

def load_cached_targets(
        params: TargetParams = None,
        time_range: TimeRange = None,
        columns: typing.List[str] = None,
        dataset_mode: DATASET_MODE = None,
        export_mode: EXPORT_MODE = None,
        aggregation_mode: AGGREGATION_MODE = None
    ) -> pd.DataFrame:
    """
    Load cached targets for a specific time range

    Parameters:
    -----------
    params : TargetParams, optional
        Target calculation parameters. If None, uses default parameters.
    time_range : TimeRange, optional
        Time range for target calculation. If None, must provide individual time parameters.
    columns : typing.List[str], optional
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
