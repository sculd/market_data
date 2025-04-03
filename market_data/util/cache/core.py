"""
Core caching functions for reading and writing data.

This module provides the fundamental functions for caching data,
including reading from cache, querying data, and writing to cache.
"""

import pandas as pd
import logging
import os
import typing
from typing import Optional, List, Union, Tuple
from datetime import datetime
import warnings

from ...ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from ...ingest.bq import candle, orderbook1l
from ...util.time import TimeRange
from .paths import get_cache_path, get_full_table_id
from .time import split_t_range, anchor_to_begin_of_day
from .dataframe import TIMESTAMP_INDEX_NAME, cache_daily_df, fetch_from_daily_cache

logger = logging.getLogger(__name__)

# Default label for market data
LABEL_MARKET_DATA = "market_data"

# Note: The functions query_and_cache, read_from_cache_or_query_and_cache, and validate_df
# have been moved to ingest/bq/cache.py as they are specific to raw data caching logic.

def read_from_cache(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    label: str = LABEL_MARKET_DATA,
    resample_interval_str: str = None,
    columns: List[str] = None
) -> Optional[pd.DataFrame]:
    """
    Read data from cache if available.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        label: Label for the cached data
        resample_interval_str: Optional resampling interval string (e.g., '1H', '15T')
        columns: Optional list of columns to load
        
    Returns:
        DataFrame with the data, or None if not in cache
    """
    t_from, t_to = time_range.to_datetime()
    t_id = get_full_table_id(dataset_mode, export_mode)
    t_ranges = split_t_range(t_from, t_to)
    
    df_concat: pd.DataFrame = None
    df_list = []
    # concat every 30 minutes to free up memory
    concat_interval = 30

    def concat_batch():
        nonlocal df_concat, df_list
        if len(df_list) == 0:
            return
        df_batch = pd.concat(df_list)
        if df_concat is None:
            df_concat = df_batch
        else:
            df_concat = pd.concat([df_concat, df_batch])
        for df in df_list:
            del df
        df_list = []

    for t_range in t_ranges:
        df = fetch_from_daily_cache(t_id, label, aggregation_mode, t_range[0], t_range[1], columns)
        if df is None:
            continue
        df_list.append(df)

        if len(df_list) > 0 and len(df_list) % concat_interval == 0:
            concat_batch()

    concat_batch()

    if df_concat is not None and resample_interval_str is not None:
        df_concat = df_concat.reset_index().groupby('symbol').apply(
            lambda x: x.set_index(TIMESTAMP_INDEX_NAME).resample(resample_interval_str).asfreq().ffill()).drop(columns='symbol').reset_index()
    
    return df_concat 