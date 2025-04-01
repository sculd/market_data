import pandas as pd
import datetime
import logging
import typing
import math

from ingest.bq import cache as raw_cache
from ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from ingest.bq.common import get_full_table_id
from ingest.util.time import TimeRange
from feature import target, TargetParams
from feature.cache_util import (
    split_t_range,
    cache_data_by_day,
    read_from_cache_generic,
    params_to_dir_name
)

def get_target_params_dir(params: TargetParams = None) -> str:
    """
    Convert target calculation parameters to a directory name string.
    
    Uses the default values when None is passed to ensure consistent directory paths.
    """
    params = params or TargetParams()
    params_dict = {
        'fp': params.forward_periods,
        'tp': params.tp_values,
        'sl': params.sl_values
    }
    return params_to_dir_name(params_dict)

def get_recommended_warm_up_days(params: TargetParams = None) -> int:
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

def cache_targets(df: pd.DataFrame, label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                 params: TargetParams = None, overwrite=True, warm_up_period_days=1, dataset_id=None) -> None:
    """Cache a target DataFrame, splitting it into daily pieces"""
    params_dir = get_target_params_dir(params)
    return cache_data_by_day(df, label, t_from, t_to, params_dir, overwrite, warm_up_period_days, dataset_id)

def read_targets_from_cache(label: str, 
                          params: TargetParams = None,
                          time_range: TimeRange = None,
                          columns: typing.List[str] = None,
                          dataset_id=None) -> pd.DataFrame:
    """Read cached target data for a specified time range"""
    params_dir = get_target_params_dir(params)
    t_from, t_to = time_range.to_datetime() if time_range else (None, None)
    return read_from_cache_generic(
        label,
        params_dir=params_dir,
        t_from=t_from, t_to=t_to,
        columns=columns,
        dataset_id=dataset_id
    )

def calculate_target_batch(raw_df: pd.DataFrame, params: TargetParams = None) -> pd.DataFrame:
    """Calculate targets for a batch of data using pandas implementation"""
    params = params or TargetParams()
    
    # Ensure timestamp is proper datetime
    if 'timestamp' in raw_df.columns and not pd.api.types.is_datetime64_any_dtype(raw_df['timestamp']):
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    
    # Use the pandas implementation to calculate targets
    return target.create_targets(raw_df, params=params)

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
    params : TargetParams, optional
        Target calculation parameters. If None, uses default parameters.
    time_range : TimeRange, optional
        Time range for target calculation. If None, must provide individual time parameters.
    warm_up_days : int, optional
        Number of warm-up days for target calculation. If None, 
        automatically determined based on the maximum forward period.
    """
    params = params or TargetParams()
    
    # If warm_up_days not provided, calculate based on target parameters
    if warm_up_days is None:
        warm_up_days = get_recommended_warm_up_days(params)
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
        raw_df = raw_cache.read_from_cache_or_query_and_cache(
            dataset_mode, export_mode, aggregation_mode,
            t_from=calc_t_from, t_to=calc_t_to,
            overwirte_cache=overwrite_cache
        )
        
        if raw_df is None or len(raw_df) == 0:
            logging.warning(f"No raw data available for {calc_t_from} to {calc_t_to}")
            continue
            
        # Calculate targets for this batch
        try:
            targets_df = calculate_target_batch(raw_df, params)
            
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
            
            cache_targets(
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
    """Load cached targets for a specific time range"""
    target_label = "targets"
    
    # Get dataset ID for cache path if dataset_mode, export_mode, and aggregation_mode are provided
    dataset_id = None
    if dataset_mode is not None and export_mode is not None and aggregation_mode is not None:
        dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{aggregation_mode}"
    
    return read_targets_from_cache(
        target_label,
        params=params,
        time_range=time_range,
        columns=columns,
        dataset_id=dataset_id
    )
