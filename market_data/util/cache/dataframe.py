"""
DataFrame utilities for caching.

This module provides functions for caching and reading pandas DataFrames,
with support for daily caching and parameter-based directory structures.
"""

import pandas as pd
import logging
import os
import datetime
from pathlib import Path
from typing import List, Optional

from .path import to_filename
from .time import (
    CACHE_INTERVAL,
    CACHE_TIMEZONE,
    is_exact_cache_interval,
    split_t_range,
    anchor_to_begin_of_day
)
from ...util.time import TimeRange
from ...ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE

logger = logging.getLogger(__name__)

# Timestamp column name
TIMESTAMP_INDEX_NAME = 'timestamp'

def cache_daily_df(df: pd.DataFrame, label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                  params_dir: str = None, overwrite=True, dataset_id: str = None,
                  dataset_mode: DATASET_MODE = None, export_mode: EXPORT_MODE = None, 
                  aggregation_mode: AGGREGATION_MODE = None, cache_base_path: str = None):
    """Cache a DataFrame that covers one-day range exactly"""
    assert cache_base_path is not None, "cache_base_path must be provided"
    if not is_exact_cache_interval(t_from, t_to):
        logger.info(f"{t_from}-{t_to} does not match {CACHE_INTERVAL=} thus will not be cached.")
        return None

    if len(df) == 0:
        logger.info(f"df for {t_from}-{t_to} is empty thus will not be cached.")
        return None

    filename = to_filename(cache_base_path, label, t_from, t_to, params_dir, dataset_id,
                          dataset_mode, export_mode, aggregation_mode)
    if os.path.exists(filename):
        logger.info(f"{filename} already exists.")
        if overwrite:
            logger.info(f"and would overwrite it.")
            df.to_parquet(filename)
        return None
    else:
        logger.info(f"{filename} does not exist.")
        df.to_parquet(filename)
        return None

def fetch_from_daily_cache(label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                          params_dir: str = None, columns: List[str] = None, 
                          dataset_id: str = None, dataset_mode: DATASET_MODE = None, 
                          export_mode: EXPORT_MODE = None, aggregation_mode: AGGREGATION_MODE = None,
                          cache_base_path: str = None) -> Optional[pd.DataFrame]:
    """Read a DataFrame that covers one-day range exactly"""
    if not is_exact_cache_interval(t_from, t_to):
        logger.info(f"{t_from}-{t_to} does not match {CACHE_INTERVAL=} thus will not be read from cache.")
        return None

    if cache_base_path is None:
        cache_base_path = os.path.expanduser('~/algo_cache')
        Path(cache_base_path).mkdir(parents=True, exist_ok=True)

    filename = to_filename(cache_base_path, label, t_from, t_to, params_dir, dataset_id,
                          dataset_mode, export_mode, aggregation_mode)
    if os.path.exists(filename):
        logger.info(f"{filename} exists.")
        return pd.read_parquet(filename, columns=columns)
    else:
        logger.info(f"{filename} does not exist.")
        return None

def read_from_cache_generic(label: str, params_dir: str = None, 
                           time_range: TimeRange = None,
                           columns: List[str] = None,
                           dataset_id: str = None, dataset_mode: DATASET_MODE = None, 
                           export_mode: EXPORT_MODE = None, aggregation_mode: AGGREGATION_MODE = None,
                           cache_base_path: str = None) -> pd.DataFrame:
    """Read cached data for a specified time range"""
    assert cache_base_path is not None, "cache_base_path must be provided"
    t_from, t_to = time_range.to_datetime() if time_range else (None, None)
    
    # Split the range into daily pieces
    daily_ranges = split_t_range(t_from, t_to, interval=CACHE_INTERVAL)
    
    # Read each daily piece
    dfs = []
    for d_from, d_to in daily_ranges:
        df = fetch_from_daily_cache(
            label, d_from, d_to, params_dir, columns,
            dataset_id, dataset_mode, export_mode, aggregation_mode,
            cache_base_path
        )
        if df is not None:
            dfs.append(df)
    
    # Concatenate all pieces
    if dfs:
        return pd.concat(dfs)
    else:
        return pd.DataFrame()

def cache_data_by_day(df: pd.DataFrame, label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                     params_dir: str = None, overwrite=False, warm_up_period_days=1, 
                     dataset_id: str = None, dataset_mode: DATASET_MODE = None, 
                     export_mode: EXPORT_MODE = None, aggregation_mode: AGGREGATION_MODE = None,
                     cache_base_path: str = None) -> None:
    """Cache a DataFrame, splitting it into daily pieces"""
    if len(df) == 0:
        logger.info(f"df is empty for {label} thus will be skipped.")
        return

    def _split_df_by_day() -> List[pd.DataFrame]:
        # Get timestamp column from index or regular column
        if isinstance(df.index, pd.DatetimeIndex) or (df.index.nlevels > 1 and TIMESTAMP_INDEX_NAME in df.index.names):
            timestamps = df.index.get_level_values(TIMESTAMP_INDEX_NAME) if df.index.nlevels > 1 else df.index
            return [group[1] for group in df.groupby(timestamps.date)]
        elif TIMESTAMP_INDEX_NAME in df.columns:
            return [group[1] for group in df.groupby(df[TIMESTAMP_INDEX_NAME].dt.date)]
        else:
            logger.error(f"DataFrame doesn't have timestamp index or column for {label}")
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
            logger.error(f"Cannot determine timestamps for daily data in {label}")
            continue
            
        t_begin = anchor_to_begin_of_day(timestamps.min())
        t_end = anchor_to_begin_of_day(t_begin + CACHE_INTERVAL)
        cache_daily_df(df_daily, label, t_begin, t_end, params_dir, overwrite=overwrite, dataset_id=dataset_id,
                       dataset_mode=dataset_mode, export_mode=export_mode, aggregation_mode=aggregation_mode,
                       cache_base_path=cache_base_path)
        del df_daily 