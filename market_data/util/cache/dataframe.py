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
                  aggregation_mode: AGGREGATION_MODE = None):
    """Cache a DataFrame that covers one-day range exactly"""
    if not is_exact_cache_interval(t_from, t_to):
        logger.info(f"{t_from}-{t_to} does not match {CACHE_INTERVAL=} thus will not be cached.")
        return None

    if len(df) == 0:
        logger.info(f"df for {t_from}-{t_to} is empty thus will not be cached.")
        return None

    filename = to_filename(CACHE_BASE_PATH, label, t_from, t_to, params_dir, dataset_id,
                          dataset_mode, export_mode, aggregation_mode)
    if os.path.exists(filename):
        logger.info(f"{filename} already exists.")
        if overwrite:
            logger.info(f"and would overwrite it.")
            df.to_parquet(filename)
        else:
            logger.info(f"and would not write it.")
    else:
        df.to_parquet(filename)
    
    return filename

def fetch_from_daily_cache(label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                          params_dir: str = None, columns: List[str] = None, 
                          dataset_id: str = None, dataset_mode: DATASET_MODE = None, 
                          export_mode: EXPORT_MODE = None, aggregation_mode: AGGREGATION_MODE = None) -> Optional[pd.DataFrame]:
    """Fetch cached data for a specific day"""
    if not is_exact_cache_interval(t_from, t_to):
        logger.info(f"{t_from} to {t_to} does not match {CACHE_INTERVAL=} thus not read from cache.")
        return None
    
    filename = to_filename(CACHE_BASE_PATH, label, t_from, t_to, params_dir, dataset_id,
                          dataset_mode, export_mode, aggregation_mode)
    if not os.path.exists(filename):
        logger.info(f"{filename=} does not exist in local cache.")
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
                           time_range: TimeRange = None,
                           columns: List[str] = None,
                           dataset_id: str = None, dataset_mode: DATASET_MODE = None, 
                           export_mode: EXPORT_MODE = None, aggregation_mode: AGGREGATION_MODE = None) -> pd.DataFrame:
    """Read cached data for a specified time range"""
    t_from, t_to = time_range.to_datetime() if time_range else (None, None)

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
        df = fetch_from_daily_cache(label, t_range[0], t_range[1], params_dir, columns, dataset_id,
                                   dataset_mode, export_mode, aggregation_mode)
        if df is None:
            continue
        df_list.append(df)

        if len(df_list) > 0 and len(df_list) % concat_interval == 0:
            concat_batch()

    concat_batch()
    return df_concat

def cache_data_by_day(df: pd.DataFrame, label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                     params_dir: str = None, overwrite=False, warm_up_period_days=1, 
                     dataset_id: str = None, dataset_mode: DATASET_MODE = None, 
                     export_mode: EXPORT_MODE = None, aggregation_mode: AGGREGATION_MODE = None) -> None:
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
                       dataset_mode=dataset_mode, export_mode=export_mode, aggregation_mode=aggregation_mode)
        del df_daily 