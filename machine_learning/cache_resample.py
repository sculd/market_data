import pandas as pd
import datetime
import logging
import typing
import os
from pathlib import Path

from ingest.bq import cache as raw_cache
from ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from ingest.bq.common import get_full_table_id
from ingest.util import time as util_time
from machine_learning.resample import resample_at_events
from feature.feature import DEFAULT_RETURN_PERIODS, DEFAULT_EMA_PERIODS
from feature.target import DEFAULT_FORWARD_PERIODS, DEFAULT_TP_VALUES, DEFAULT_SL_VALUES
from feature.cache_util import params_to_dir_name, split_t_range

# The base directory for cache
CACHE_BASE_PATH = os.path.expanduser('~/algo_cache/resample')
Path(CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

def get_resample_params_dir(
    threshold: float = 0.05,
    forward_periods=DEFAULT_FORWARD_PERIODS, 
    tp_values=DEFAULT_TP_VALUES, 
    sl_values=DEFAULT_SL_VALUES,
    return_periods=DEFAULT_RETURN_PERIODS, 
    ema_periods=DEFAULT_EMA_PERIODS, 
    add_btc_features=True,
    sequence_length: int = None  # Only needed for sequence datasets
):
    """
    Convert resampling parameters to a directory name string.
    
    Uses the default values from feature.py and target.py for parameters.
    """
    # Build parameters dictionary
    params = {
        'th': threshold,
        'fp': forward_periods,
        'tp': tp_values,
        'sl': sl_values,
        'rp': return_periods,
        'ep': ema_periods,
        'btc': add_btc_features
    }
    
    # Add sequence_length only if provided (for sequence datasets)
    if sequence_length is not None:
        params['seq'] = sequence_length
        
    return params_to_dir_name(params)

def to_filename(label: str, t_from: datetime.datetime, t_to: datetime.datetime, params_dir: str, dataset_id: str = None) -> str:
    """
    Generate a filename for the cached resampled data.
    
    Parameters:
    -----------
    label : str
        Type of data (e.g., "resampled")
    t_from : datetime.datetime
        Start time of the data
    t_to : datetime.datetime
        End time of the data
    params_dir : str
        Parameters directory name
    dataset_id : str, optional
        Dataset identifier
    """
    t_str_from = t_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_str_to = t_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    
    # Base directory structure with dataset_id if provided
    if dataset_id:
        base_dir = os.path.join(CACHE_BASE_PATH, label, dataset_id)
    else:
        base_dir = os.path.join(CACHE_BASE_PATH, label)
    
    # Include params in the directory structure
    dir_path = os.path.join(base_dir, params_dir)
        
    fn = os.path.join(dir_path, f"{t_str_from}_{t_str_to}.parquet")
    
    # Ensure directory exists
    Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)
    return fn

def cache_resampled_df(df: pd.DataFrame, label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                      params_dir: str, overwrite=True, dataset_id: str = None):
    """Cache a resampled DataFrame for the given time range"""
    if len(df) == 0:
        logging.info(f"DataFrame for {t_from}-{t_to} is empty thus will not be cached.")
        return None

    filename = to_filename(label, t_from, t_to, params_dir, dataset_id)
    if os.path.exists(filename):
        logging.info(f"{filename} already exists.")
        if overwrite:
            logging.info(f"and would overwrite it.")
            df.to_parquet(filename)
        else:
            logging.info(f"and would not write it.")
    else:
        df.to_parquet(filename)
    
    return filename

def cache_resampled_data_by_day(df: pd.DataFrame, label: str, t_from: datetime.datetime, t_to: datetime.datetime,
                               params_dir: str, overwrite=True, warm_up_period_days=0, dataset_id=None) -> typing.List[str]:
    """
    Cache a resampled DataFrame, splitting it into daily pieces.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with timestamp index to cache
    label : str
        Type of data (e.g., "resampled")
    t_from, t_to : datetime.datetime
        Time range for the data
    params_dir : str
        Parameters directory name
    overwrite : bool
        Whether to overwrite existing cache files
    warm_up_period_days : int
        Number of days to skip at the beginning (warm-up period)
    dataset_id : str, optional
        Dataset identifier
        
    Returns:
    --------
    List[str]
        List of filenames that were cached
    """
    if len(df) == 0:
        logging.info(f"DataFrame for {t_from}-{t_to} is empty thus will not be cached.")
        return []
    
    # Skip warm-up period days
    if warm_up_period_days > 0:
        skip_until = t_from + datetime.timedelta(days=warm_up_period_days)
        df = df[df.index >= skip_until]
        
        if len(df) == 0:
            logging.info(f"DataFrame is empty after skipping warm-up period, will not be cached.")
            return []
    
    # Force dates to be rounded to days for consistent chunking
    start_date = t_from.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = t_to.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # Split by day
    filenames = []
    current_date = start_date
    while current_date <= end_date:
        next_date = current_date + datetime.timedelta(days=1)
        # Extract data for the current day
        day_df = df[(df.index >= current_date) & (df.index < next_date)]
        
        if len(day_df) > 0:
            # Cache the daily chunk
            filename = cache_resampled_df(
                day_df, 
                label, 
                current_date, 
                next_date - datetime.timedelta(microseconds=1),
                params_dir,
                overwrite,
                dataset_id
            )
            if filename:
                filenames.append(filename)
                
        current_date = next_date
    
    return filenames

def fetch_resampled_from_cache(label: str, 
                              t_from: datetime.datetime = None, 
                              t_to: datetime.datetime = None,
                              epoch_seconds_from: int = None, 
                              epoch_seconds_to: int = None,
                              date_str_from: str = None, 
                              date_str_to: str = None,
                              params_dir: str = None, 
                              columns: typing.List[str] = None, 
                              dataset_id: str = None) -> typing.Optional[pd.DataFrame]:
    """
    Fetch cached resampled data for the specified time range
    
    Parameters:
    -----------
    label : str
        Type of data to fetch
    t_from, t_to : datetime.datetime
        Time range for the data
    epoch_seconds_from, epoch_seconds_to : int
        Alternative time range specification as epoch seconds
    date_str_from, date_str_to : str
        Alternative time range specification as date strings
    params_dir : str
        Parameters directory name
    columns : List[str]
        Specific columns to fetch
    dataset_id : str
        Dataset identifier
    """
    # Resolve time range
    t_from, t_to = util_time.to_t(
        t_from=t_from, t_to=t_to,
        epoch_seconds_from=epoch_seconds_from, epoch_seconds_to=epoch_seconds_to,
        date_str_from=date_str_from, date_str_to=date_str_to,
    )
    
    # For fetching, we need to search for files covering the whole period
    # So we check the cache directory for any files that might contain our data
    
    # Generate base directory path
    if dataset_id:
        base_dir = os.path.join(CACHE_BASE_PATH, label, dataset_id)
    else:
        base_dir = os.path.join(CACHE_BASE_PATH, label)
    
    # Include params in the directory structure if provided
    if params_dir:
        dir_path = os.path.join(base_dir, params_dir)
    else:
        dir_path = base_dir
    
    if not os.path.exists(dir_path):
        logging.info(f"Cache directory {dir_path} does not exist.")
        return None
    
    # Find all files in the directory
    all_files = [f for f in os.listdir(dir_path) if f.endswith('.parquet')]
    
    # Filter files that overlap with our time range
    relevant_dfs = []
    for filename in all_files:
        # Extract time range from filename
        try:
            file_t_str_from, file_t_str_to = filename.split('_', 1)
            file_t_str_to = file_t_str_to.replace('.parquet', '')
            
            file_t_from = pd.to_datetime(file_t_str_from)
            file_t_to = pd.to_datetime(file_t_str_to)
            
            # Check if this file's time range overlaps with our requested range
            if (file_t_from <= t_to) and (file_t_to >= t_from):
                file_path = os.path.join(dir_path, filename)
                df = pd.read_parquet(file_path)
                
                # Filter to the requested columns if specified
                if columns is not None:
                    valid_columns = [c for c in columns if c in df.columns]
                    if valid_columns:
                        df = df[valid_columns]
                
                # Filter to the requested time range
                df = df[(df.index >= t_from) & (df.index <= t_to)]
                
                if len(df) > 0:
                    relevant_dfs.append(df)
        except Exception as e:
            logging.warning(f"Error processing cache file {filename}: {e}")
            continue
    
    if not relevant_dfs:
        return None
    
    # Combine and sort all relevant data
    combined_df = pd.concat(relevant_dfs)
    combined_df = combined_df.sort_index()
    
    # Remove duplicates if any (should not happen with daily files, but just in case)
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    
    return combined_df

def resample_and_cache_data(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        price_col: str = 'close',
        threshold: float = 0.05,
        t_from: datetime.datetime = None,
        t_to: datetime.datetime = None,
        epoch_seconds_from: int = None,
        epoch_seconds_to: int = None,
        date_str_from: str = None,
        date_str_to: str = None,
        calculation_batch_days: int = 7,  # Use larger batches for resampling
        overwrite_cache: bool = True,
        label: str = "resampled"
) -> None:
    """
    Resample data at significant price movements and cache it.
    
    This function:
    1. Seeks raw data files for the range
    2. Caches raw data if not present
    3. Resamples the data based on significant price movements
    4. Caches the resampled results daily
    
    Parameters:
    -----------
    dataset_mode : DATASET_MODE
        Dataset mode (LIVE, REPLAY, etc.)
    export_mode : EXPORT_MODE
        Export mode (OHLC, TICKS, etc.)
    aggregation_mode : AGGREGATION_MODE
        Aggregation mode (MIN_1, MIN_5, etc.)
    price_col : str
        Column to use for detecting significant price movements
    threshold : float
        Threshold for significant price movement (as decimal)
    t_from, t_to : datetime.datetime
        Time range for the data
    epoch_seconds_from, epoch_seconds_to : int
        Alternative time range specification as epoch seconds
    date_str_from, date_str_to : str
        Alternative time range specification as date strings
    calculation_batch_days : int
        Size of batches in days for processing
    overwrite_cache : bool
        Whether to overwrite existing cache files
    label : str
        Label for the cache directory
    """
    # Resolve time range
    t_from, t_to = util_time.to_t(
        t_from=t_from, t_to=t_to,
        epoch_seconds_from=epoch_seconds_from, epoch_seconds_to=epoch_seconds_to,
        date_str_from=date_str_from, date_str_to=date_str_to,
    )
    
    # Get dataset ID for cache path
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{aggregation_mode}"
    
    # Set up calculation parameters
    if calculation_batch_days <= 0:
        calculation_batch_days = 1
    calculation_interval = datetime.timedelta(days=calculation_batch_days)
    
    # Generate parameters directory name
    params_dir = get_resample_params_dir(threshold=threshold)
    
    # Split the range into calculation batches
    calculation_ranges = split_t_range(t_from, t_to, interval=calculation_interval)
    
    for calc_range in calculation_ranges:
        calc_t_from, calc_t_to = calc_range
        logging.info(f"Processing resampling batch {calc_t_from} to {calc_t_to}")
        
        # 1 & 2. Get raw data (fetch and cache if not present)
        raw_df = raw_cache.read_from_cache_or_query_and_cache(
            dataset_mode, export_mode, aggregation_mode,
            t_from=calc_t_from, t_to=calc_t_to,
            overwirte_cache=overwrite_cache
        )
        
        if raw_df is None or len(raw_df) == 0:
            logging.warning(f"No raw data available for {calc_t_from} to {calc_t_to}")
            continue
            
        # 3. Resample the data for this batch
        try:
            resampled_df = resample_at_events(
                raw_df,
                price_col=price_col,
                threshold=threshold
            )
            
            if resampled_df is None or len(resampled_df) == 0:
                logging.warning(f"Resampling returned empty result for {calc_t_from} to {calc_t_to}")
                continue
                
            # 4. Cache resampled data by day
            cache_resampled_data_by_day(
                resampled_df, 
                label, 
                calc_t_from, 
                calc_t_to,
                params_dir,
                overwrite=overwrite_cache,
                dataset_id=dataset_id
            )
            
        except Exception as e:
            logging.error(f"Error resampling data for {calc_t_from} to {calc_t_to}: {e}")
            continue

def load_cached_resampled_data(
        threshold: float = 0.05,
        t_from: datetime.datetime = None,
        t_to: datetime.datetime = None,
        epoch_seconds_from: int = None,
        epoch_seconds_to: int = None,
        date_str_from: str = None,
        date_str_to: str = None,
        columns: typing.List[str] = None,
        dataset_mode: DATASET_MODE = None,
        export_mode: EXPORT_MODE = None,
        aggregation_mode: AGGREGATION_MODE = None,
        label: str = "resampled"
    ) -> pd.DataFrame:
    """Load cached resampled data for a specific time range"""
    # Get dataset ID for cache path if dataset_mode, export_mode, and aggregation_mode are provided
    dataset_id = None
    if dataset_mode is not None and export_mode is not None and aggregation_mode is not None:
        dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{aggregation_mode}"
    
    # Generate parameters directory name
    params_dir = get_resample_params_dir(threshold=threshold)
    
    return fetch_resampled_from_cache(
        label,
        t_from=t_from, 
        t_to=t_to,
        epoch_seconds_from=epoch_seconds_from, 
        epoch_seconds_to=epoch_seconds_to,
        date_str_from=date_str_from, 
        date_str_to=date_str_to,
        params_dir=params_dir,
        columns=columns,
        dataset_id=dataset_id
    )
