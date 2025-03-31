import pandas as pd
import datetime
import pytz
import logging
import typing
import os
import json
import hashlib
from pathlib import Path
import re

from ingest.util import time as util_time

# The base directory for cache
CACHE_BASE_PATH = os.path.expanduser('~/algo_cache/feature_data')
Path(CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

# Cache by day
CACHE_INTERVAL = datetime.timedelta(days=1)
# Timezone for day boundaries 
CACHE_TIMEZONE = pytz.timezone('America/New_York')
# Timestamp column name
TIMESTAMP_INDEX_NAME = 'timestamp'

def _sanitize_for_path(value_str: str) -> str:
    """Convert a string to a filesystem-safe string for directory names"""
    # Remove or replace characters that could cause issues in filenames
    value_str = re.sub(r'[\\/*?:"<>|]', '_', value_str)
    # Replace spaces, commas and dots with underscores
    value_str = re.sub(r'[\s,\.]+', '_', value_str)
    # Remove any trailing or leading underscores
    value_str = value_str.strip('_')
    return value_str

def params_to_dir_name(params: dict) -> str:
    """
    Convert a parameters dictionary to a directory name string.
    
    For list parameters, joins the elements with underscores to make them
    human-readable. For other complex parameters like dicts, still uses a hash.
    """
    if not params:
        return "default"
        
    parts = []
    for key, value in sorted(params.items()):
        # Skip None values
        if value is None:
            continue
            
        # For list values, create a human-readable string with actual values
        if isinstance(value, list):
            # Handle different types of elements
            if all(isinstance(x, (int, float)) for x in value):
                # For numeric lists, join with underscores
                value_str = f"{key}-" + "_".join(str(x) for x in value)
                
                # Truncate if too long (over 50 chars)
                if len(value_str) > 50:
                    value_str = f"{key}-len{len(value)}-" + "_".join(str(x) for x in value[:3]) + "..."
            else:
                # For non-numeric lists or mixed lists, still use length and hash
                value_str = json.dumps(value, sort_keys=True)
                value_hash = hashlib.md5(value_str.encode()).hexdigest()[:6]
                value_str = f"{key}-len{len(value)}-{value_hash}"
                
            parts.append(_sanitize_for_path(value_str))
            
        # For dictionaries or sets, just use a hash
        elif isinstance(value, (dict, set)):
            value_str = json.dumps(sorted(value) if isinstance(value, set) else value, sort_keys=True)
            value_hash = hashlib.md5(value_str.encode()).hexdigest()[:8]
            parts.append(f"{key}-{value_hash}")
            
        # For simple values, use them directly
        else:
            parts.append(f"{key}-{value}")
    
    if not parts:
        return "default"
        
    return "_".join(parts)

def to_filename(basedir: str, label: str, t_from: datetime.datetime, t_to: datetime.datetime, params_dir: str = None, dataset_id: str = None) -> str:
    """
    Generate a filename for the cached data based on time range and parameters.
    
    Parameters:
    -----------
    basedir : str
        Base directory for caching
    label : str
        Type of data (e.g., "features", "targets")
    t_from : datetime.datetime
        Start time of the data
    t_to : datetime.datetime
        End time of the data
    params_dir : str, optional
        Parameters directory name, generated from params_to_dir_name
    dataset_id : str, optional
        Dataset identifier (e.g., "trading-290017.market_data_okx.by_minute_AGGREGATION_MODE.TAKE_LASTEST")
        If provided, this will be included in the path
    """
    t_str_from = t_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_str_to = t_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    
    # Base directory structure with dataset_id if provided
    if dataset_id:
        base_dir = os.path.join(basedir, label, dataset_id)
    else:
        base_dir = os.path.join(basedir, label)
    
    # Include params in the directory structure if provided
    if params_dir:
        dir_path = os.path.join(base_dir, params_dir)
    else:
        dir_path = base_dir
        
    fn = os.path.join(dir_path, f"{t_str_from}_{t_str_to}.parquet")
    
    # Ensure directory exists
    Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)
    return fn

def anchor_to_begin_of_day(t: datetime.datetime) -> datetime.datetime:
    """Anchor a timestamp to the beginning of its day"""
    return CACHE_TIMEZONE.localize(datetime.datetime(year=t.year, month=t.month, day=t.day, hour=0, minute=0, second=0))

def split_t_range(t_from: datetime.datetime, t_to: datetime.datetime, 
                 interval: datetime.timedelta = CACHE_INTERVAL,
                 warm_up: datetime.timedelta = None) -> typing.List[typing.Tuple[datetime.datetime, datetime.datetime]]:
    """
    Split a time range into intervals, with an optional warm-up period for each interval.
    
    Parameters:
    -----------
    t_from : datetime.datetime
        Start time of the entire range
    t_to : datetime.datetime
        End time of the entire range
    interval : datetime.timedelta
        Step size for advancing through the range (default is CACHE_INTERVAL = 1 day)
    warm_up : datetime.timedelta, optional
        Optional warm-up period to include at the beginning of each range
        
    Returns:
    --------
    List of (start_time, end_time) tuples, where:
        - Each range moves forward by 'interval'
        - If warm_up is provided, each start_time is adjusted to include the warm-up period
        - The warm-up period overlaps with the previous interval
    
    Examples:
    ---------
    With interval=1 day, no warm-up:
        [(day1, day2), (day2, day3), (day3, day4)]
    
    With interval=1 day, warm_up=2 days:
        [(day1, day2), (day1-2days, day3), (day2-2days, day4)]
    """
    ret = []
    # Anchor times to beginning of day
    t1 = anchor_to_begin_of_day(t_from - warm_up)
    t2 = anchor_to_begin_of_day(t_from + interval)
    
    # First range doesn't have warm-up (starting from t_from)
    ret.append((t1, t2))
    
    # Process subsequent ranges with warm-up if specified
    while t2 < t_to:
        # Move forward by interval for the end time
        t2 = anchor_to_begin_of_day(t2 + interval)
        
        # Calculate start time, adjusted for warm-up if provided
        if warm_up is not None:
            # Start earlier by warm_up period
            t1_with_warmup = anchor_to_begin_of_day(t2 - interval - warm_up)
        else:
            # Without warm-up, start at previous end
            t1_with_warmup = anchor_to_begin_of_day(t2 - interval)
        
        # Add the range to our list
        ret.append((t1_with_warmup, t2))
    
    # Adjust the last range if it exceeds t_to
    last = (ret[-1][0], min(ret[-1][1], t_to))
    ret[-1] = last
    
    return ret

def is_exact_cache_interval(t_from: datetime.datetime, t_to: datetime.datetime) -> bool:
    """Check if time range is exactly one cache interval (day) starting at zero hour"""
    t_from_plus_interval = anchor_to_begin_of_day(t_from + CACHE_INTERVAL)
    if t_to != t_from_plus_interval:
        return False

    def at_begin_of_day(t: datetime.datetime) -> bool:
        return t.hour == 0 and t.minute == 0 and t.second == 0

    return at_begin_of_day(t_from) and at_begin_of_day(t_to)

def cache_daily_df(df: pd.DataFrame, label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                  params_dir: str = None, overwrite=True, dataset_id: str = None):
    """Cache a DataFrame that covers one-day range exactly"""
    if not is_exact_cache_interval(t_from, t_to):
        logging.info(f"{t_from}-{t_to} does not match {CACHE_INTERVAL=} thus will not be cached.")
        return None

    if len(df) == 0:
        logging.info(f"df for {t_from}-{t_to} is empty thus will not be cached.")
        return None

    filename = to_filename(CACHE_BASE_PATH, label, t_from, t_to, params_dir, dataset_id)
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

def fetch_from_daily_cache(label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                          params_dir: str = None, columns: typing.List[str] = None, 
                          dataset_id: str = None) -> typing.Optional[pd.DataFrame]:
    """Fetch cached data for a specific day"""
    if not is_exact_cache_interval(t_from, t_to):
        logging.info(f"{t_from} to {t_to} does not match {CACHE_INTERVAL=} thus not read from cache.")
        return None
    
    filename = to_filename(CACHE_BASE_PATH, label, t_from, t_to, params_dir, dataset_id)
    if not os.path.exists(filename):
        logging.info(f"{filename=} does not exist in local cache.")
        return None
    
    df = pd.read_parquet(filename)
    if len(df) == 0:
        return None

    if columns is None:
        return df
    else:
        columns = [c for c in columns if c in df.columns]
        return df[columns]

def read_from_cache_generic(label: str, params_dir: str = None, 
                           t_from: datetime.datetime = None, t_to: datetime.datetime = None,
                           epoch_seconds_from: int = None, epoch_seconds_to: int = None,
                           date_str_from: str = None, date_str_to: str = None,
                           columns: typing.List[str] = None,
                           dataset_id: str = None) -> pd.DataFrame:
    """Read cached data for a specified time range"""
    t_from, t_to = util_time.to_t(
        t_from=t_from, t_to=t_to,
        epoch_seconds_from=epoch_seconds_from, epoch_seconds_to=epoch_seconds_to,
        date_str_from=date_str_from, date_str_to=date_str_to,
    )

    t_ranges = split_t_range(t_from, t_to)
    df_concat = None
    df_list = []
    # Concat every 10 days to free memory
    concat_interval = 10

    def concat_batch():
        nonlocal df_concat, df_list
        if not df_list:
            return
        df_batch = pd.concat(df_list)
        if df_concat is None:
            df_concat = df_batch
        else:
            df_concat = pd.concat([df_concat, df_batch])
        df_list = []

    for i, t_range in enumerate(t_ranges):
        df = fetch_from_daily_cache(label, t_range[0], t_range[1], params_dir, columns, dataset_id)
        if df is None:
            continue
        df_list.append(df)

        if len(df_list) > 0 and len(df_list) % concat_interval == 0:
            concat_batch()

    concat_batch()
    return df_concat

def cache_data_by_day(df: pd.DataFrame, label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                     params_dir: str = None, overwrite=False, warm_up_period_days=1, 
                     dataset_id: str = None) -> None:
    """Cache a DataFrame, splitting it into daily pieces"""
    if len(df) == 0:
        logging.info(f"df is empty for {label} thus will be skipped.")
        return

    def _split_df_by_day() -> typing.List[pd.DataFrame]:
        # Get timestamp column from index or regular column
        if isinstance(df.index, pd.DatetimeIndex) or (df.index.nlevels > 1 and TIMESTAMP_INDEX_NAME in df.index.names):
            timestamps = df.index.get_level_values(TIMESTAMP_INDEX_NAME) if df.index.nlevels > 1 else df.index
            return [group[1] for group in df.groupby(timestamps.date)]
        elif TIMESTAMP_INDEX_NAME in df.columns:
            return [group[1] for group in df.groupby(df[TIMESTAMP_INDEX_NAME].dt.date)]
        else:
            logging.error(f"DataFrame doesn't have timestamp index or column for {label}")
            return [df]  # Return whole df as one group

    df_dailys = _split_df_by_day()
    for i, df_daily in enumerate(df_dailys):
        if i < warm_up_period_days:
            continue

        # Find time range for this daily df
        if isinstance(df_daily.index, pd.DatetimeIndex) or (df_daily.index.nlevels > 1 and TIMESTAMP_INDEX_NAME in df_daily.index.names):
            timestamps = df_daily.index.get_level_values(TIMESTAMP_INDEX_NAME) if df_daily.index.nlevels > 1 else df_daily.index
        elif TIMESTAMP_INDEX_NAME in df_daily.columns:
            timestamps = df_daily[TIMESTAMP_INDEX_NAME]
        else:
            logging.error(f"Cannot determine timestamps for daily data in {label}")
            continue
            
        t_begin = anchor_to_begin_of_day(timestamps.min())
        t_end = anchor_to_begin_of_day(t_begin + CACHE_INTERVAL)
        cache_daily_df(df_daily, label, t_begin, t_end, params_dir, overwrite=overwrite, dataset_id=dataset_id)
        del df_daily
