import pandas as pd
import datetime
import logging
import typing
import math

from ingest.bq import cache as raw_cache
from ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from ingest.bq.common import get_full_table_id
from ingest.util import time as util_time
from feature import feature
from feature.feature import DEFAULT_RETURN_PERIODS, DEFAULT_EMA_PERIODS
from feature.cache_util import (
    split_t_range,
    cache_data_by_day,
    read_from_cache_generic,
    params_to_dir_name
)

def get_feature_params_dir(return_periods=None, ema_periods=None, add_btc_features=True):
    """
    Convert feature calculation parameters to a directory name string.
    
    Uses the default values when None is passed to ensure consistent directory paths.
    """
    # Use default values if None is provided
    if return_periods is None:
        return_periods = DEFAULT_RETURN_PERIODS
    if ema_periods is None:
        ema_periods = DEFAULT_EMA_PERIODS
        
    params = {
        'rp': return_periods,
        'ep': ema_periods,
        'btc': add_btc_features
    }
    return params_to_dir_name(params)

def get_recommended_warm_up_days(return_periods=None, ema_periods=None):
    """
    Calculate the recommended warm-up period based on feature parameters.
    
    Uses the maximum window size from return_periods and ema_periods, plus a buffer
    to ensure sufficient historical data for feature calculations.
    
    Returns:
        int: Recommended number of warm-up days
    """
    # Use default values if None is provided
    if return_periods is None:
        return_periods = DEFAULT_RETURN_PERIODS
    if ema_periods is None:
        ema_periods = DEFAULT_EMA_PERIODS
    
    # Find the maximum window period
    max_window = max(max(return_periods), max(ema_periods))
    
    # Convert to days (assuming periods are in minutes for 24/7 markets)
    days_needed = math.ceil(max_window / (24 * 60))
    
    return days_needed

def cache_features(df: pd.DataFrame, label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                  return_periods=None, ema_periods=None, add_btc_features=True,
                  overwrite=True, warm_up_period_days=1, dataset_id=None) -> None:
    """Cache a feature DataFrame, splitting it into daily pieces"""
    params_dir = get_feature_params_dir(return_periods, ema_periods, add_btc_features)
    return cache_data_by_day(df, label, t_from, t_to, params_dir, overwrite, warm_up_period_days, dataset_id)

def read_features_from_cache(label: str, 
                           return_periods=None, ema_periods=None, add_btc_features=True,
                           t_from: datetime.datetime = None, t_to: datetime.datetime = None,
                           epoch_seconds_from: int = None, epoch_seconds_to: int = None,
                           date_str_from: str = None, date_str_to: str = None,
                           columns: typing.List[str] = None,
                           dataset_id=None) -> pd.DataFrame:
    """Read cached feature data for a specified time range"""
    params_dir = get_feature_params_dir(return_periods, ema_periods, add_btc_features)
    return read_from_cache_generic(
        label,
        params_dir=params_dir,
        t_from=t_from, t_to=t_to,
        epoch_seconds_from=epoch_seconds_from, epoch_seconds_to=epoch_seconds_to,
        date_str_from=date_str_from, date_str_to=date_str_to,
        columns=columns,
        dataset_id=dataset_id
    )

def calculate_feature_batch(raw_df: pd.DataFrame, 
                           return_periods: typing.List[int] = None, 
                           ema_periods: typing.List[int] = None,
                           add_btc_features: bool = True) -> pd.DataFrame:
    """Calculate features for a batch of data using pandas implementation"""
    if return_periods is None:
        return_periods = DEFAULT_RETURN_PERIODS
    if ema_periods is None:
        ema_periods = DEFAULT_EMA_PERIODS
        
    # Ensure timestamp is proper datetime
    if 'timestamp' in raw_df.columns and not pd.api.types.is_datetime64_any_dtype(raw_df['timestamp']):
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    
    # Use the pandas implementation to calculate features
    return feature.create_features(
        raw_df, 
        return_periods=return_periods, 
        ema_periods=ema_periods, 
        add_btc_features=add_btc_features
    )

def calculate_and_cache_features(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        return_periods: typing.List[int] = None,
        ema_periods: typing.List[int] = None,
        add_btc_features: bool = True,
        t_from: datetime.datetime = None,
        t_to: datetime.datetime = None,
        epoch_seconds_from: int = None,
        epoch_seconds_to: int = None,
        date_str_from: str = None,
        date_str_to: str = None,
        calculation_batch_days: int = 1,
        warm_up_days: int = None,  # Now optional with None as default
        overwrite_cache: bool = True,
    ) -> None:
    """
    Calculate and cache features for a specified time range.
    
    1. Seeks raw data files for the range
    2. Caches raw data if not present
    3. Calculates features in batches
    4. Caches feature results daily
    
    Parameters:
    -----------
    warm_up_days : int, optional
        Number of warm-up days for feature calculation. If None, 
        automatically determined based on the maximum window size
        in return_periods and ema_periods.
    """
    # If warm_up_days not provided, calculate based on feature parameters
    if warm_up_days is None:
        warm_up_days = get_recommended_warm_up_days(return_periods, ema_periods)
        logging.info(f"Using auto-calculated warm-up period of {warm_up_days} days based on feature parameters")
    
    # Resolve time range
    t_from, t_to = util_time.to_t(
        t_from=t_from, t_to=t_to,
        epoch_seconds_from=epoch_seconds_from, epoch_seconds_to=epoch_seconds_to,
        date_str_from=date_str_from, date_str_to=date_str_to,
    )
    
    # Create label for feature cache
    feature_label = "features"
    
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
        logging.info(f"Processing calculation batch {calc_t_from} to {calc_t_to}")
        
        # 1 & 2. Get raw data (fetch and cache if not present)
        raw_df = raw_cache.read_from_cache_or_query_and_cache(
            dataset_mode, export_mode, aggregation_mode,
            t_from=calc_t_from, t_to=calc_t_to,
            overwirte_cache=overwrite_cache
        )
        
        if raw_df is None or len(raw_df) == 0:
            logging.warning(f"No raw data available for {calc_t_from} to {calc_t_to}")
            continue
            
        # 3. Calculate features for this batch
        try:
            features_df = calculate_feature_batch(
                raw_df,
                return_periods=return_periods, 
                ema_periods=ema_periods,
                add_btc_features=add_btc_features
            )
            
            if features_df is None or len(features_df) == 0:
                logging.warning(f"Feature calculation returned empty result for {calc_t_from} to {calc_t_to}")
                continue
                
            # 4. Cache features daily
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
                
            cache_features(
                features_df, feature_label, 
                calc_t_from, calc_t_to,
                return_periods=return_periods,
                ema_periods=ema_periods,
                add_btc_features=add_btc_features,
                overwrite=overwrite_cache,
                warm_up_period_days=warm_up_period_days,
                dataset_id=dataset_id
            )
            
        except Exception as e:
            logging.error(f"Error calculating features for {calc_t_from} to {calc_t_to}: {e}")
            continue

def load_cached_features(
        return_periods: typing.List[int] = None,
        ema_periods: typing.List[int] = None,
        add_btc_features: bool = True,
        t_from: datetime.datetime = None,
        t_to: datetime.datetime = None,
        epoch_seconds_from: int = None,
        epoch_seconds_to: int = None,
        date_str_from: str = None,
        date_str_to: str = None,
        columns: typing.List[str] = None,
        dataset_mode: DATASET_MODE = None,
        export_mode: EXPORT_MODE = None,
        aggregation_mode: AGGREGATION_MODE = None
    ) -> pd.DataFrame:
    """Load cached features for a specific time range"""
    feature_label = "features"
    
    # Get dataset ID for cache path if dataset_mode, export_mode, and aggregation_mode are provided
    dataset_id = None
    if dataset_mode is not None and export_mode is not None and aggregation_mode is not None:
        dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{aggregation_mode}"
    
    return read_features_from_cache(
        feature_label,
        return_periods=return_periods,
        ema_periods=ema_periods,
        add_btc_features=add_btc_features,
        t_from=t_from, t_to=t_to,
        epoch_seconds_from=epoch_seconds_from, epoch_seconds_to=epoch_seconds_to,
        date_str_from=date_str_from, date_str_to=date_str_to,
        columns=columns,
        dataset_id=dataset_id
    )
