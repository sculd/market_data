import pandas as pd
import datetime
import logging
import typing
import math

from ..ingest.bq.cache import read_from_cache_or_query_and_cache
from ..ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from ..ingest.bq.common import get_full_table_id
from ..util.time import TimeRange
from .feature import create_features, FeatureParams
from ..util.cache.time import (
    split_t_range,
)
from ..util.cache.dataframe import (
    cache_data_by_day,
    read_from_cache_generic,
)
from ..util.cache.path import (
    params_to_dir_name
)

def get_feature_params_dir(params: FeatureParams = None) -> str:
    """
    Convert feature calculation parameters to a directory name string.
    
    Uses the default values when None is passed to ensure consistent directory paths.
    """
    params = params or FeatureParams()
    params_dict = {
        'rp': params.return_periods,
        'ep': params.ema_periods,
        'btc': params.add_btc_features
    }
    return params_to_dir_name(params_dict)

def get_recommended_warm_up_days(params: FeatureParams = None) -> int:
    """
    Calculate the recommended warm-up period based on feature parameters.
    
    Uses the maximum window size from return_periods and ema_periods, plus a buffer
    to ensure sufficient historical data for feature calculations.
    
    Returns:
        int: Recommended number of warm-up days
    """
    params = params or FeatureParams()
    
    # Find the maximum window period
    max_window = max(max(params.return_periods), max(params.ema_periods))
    
    # Convert to days (assuming periods are in minutes for 24/7 markets)
    days_needed = math.ceil(max_window / (24 * 60))
    
    return days_needed

def cache_features(df: pd.DataFrame, label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                  params: FeatureParams = None, overwrite=True, warm_up_period_days=1, dataset_id=None) -> None:
    """Cache a feature DataFrame, splitting it into daily pieces"""
    params_dir = get_feature_params_dir(params)
    return cache_data_by_day(df, label, t_from, t_to, params_dir, overwrite, warm_up_period_days, dataset_id)

def read_features_from_cache(label: str, 
                           params: FeatureParams = None,
                           time_range: TimeRange = None,
                           columns: typing.List[str] = None,
                           dataset_id=None) -> pd.DataFrame:
    """Read cached feature data for a specified time range"""
    params_dir = get_feature_params_dir(params)
    t_from, t_to = time_range.to_datetime() if time_range else (None, None)
    return read_from_cache_generic(
        label,
        params_dir=params_dir,
        t_from=t_from, t_to=t_to,
        columns=columns,
        dataset_id=dataset_id
    )

def calculate_feature_batch(raw_df: pd.DataFrame, params: FeatureParams = None) -> pd.DataFrame:
    """Calculate features for a batch of data using pandas implementation"""
    params = params or FeatureParams()
    
    # Ensure timestamp is proper datetime
    if 'timestamp' in raw_df.columns and not pd.api.types.is_datetime64_any_dtype(raw_df['timestamp']):
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    
    # Use the pandas implementation to calculate features
    return create_features(raw_df, params=params)

def calculate_and_cache_features(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        params: FeatureParams = None,
        time_range: TimeRange = None,
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
    params : FeatureParams, optional
        Feature calculation parameters. If None, uses default parameters.
    time_range : TimeRange, optional
        Time range for feature calculation. If None, must provide individual time parameters.
    warm_up_days : int, optional
        Number of warm-up days for feature calculation. If None, 
        automatically determined based on the maximum window size
        in return_periods and ema_periods.
    """
    params = params or FeatureParams()
    
    # If warm_up_days not provided, calculate based on feature parameters
    if warm_up_days is None:
        warm_up_days = get_recommended_warm_up_days(params)
        logging.info(f"Using auto-calculated warm-up period of {warm_up_days} days based on feature parameters")
    
    # Resolve time range
    t_from, t_to = time_range.to_datetime() if time_range else (None, None)
    
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
        raw_df = read_from_cache_or_query_and_cache(
            dataset_mode, export_mode, aggregation_mode,
            t_from=calc_t_from, t_to=calc_t_to,
            overwirte_cache=overwrite_cache
        )
        
        if raw_df is None or len(raw_df) == 0:
            logging.warning(f"No raw data available for {calc_t_from} to {calc_t_to}")
            continue
            
        # 3. Calculate features for this batch
        try:
            features_df = calculate_feature_batch(raw_df, params)
            
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
                params=params,
                overwrite=overwrite_cache,
                warm_up_period_days=warm_up_period_days,
                dataset_id=dataset_id
            )
            
        except Exception as e:
            logging.error(f"Error calculating features for {calc_t_from} to {calc_t_to}: {e}")
            continue

def load_cached_features(
        params: FeatureParams = None,
        time_range: TimeRange = None,
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
        params=params,
        time_range=time_range,
        columns=columns,
        dataset_id=dataset_id
    )
