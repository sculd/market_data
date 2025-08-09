"""
DataFrame utilities for caching.

This module provides functions for caching and reading pandas DataFrames,
with support for daily caching and parameter-based directory structures.
"""

import pandas as pd
import logging
import os
import datetime
from typing import List, Callable
import multiprocessing
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

from market_data.util.cache.path import to_filename
from market_data.util.cache.time import (
    CACHE_INTERVAL,
    is_exact_cache_interval,
    split_t_range,
)
from market_data.util.time import TimeRange
from market_data.ingest.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE

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


def _cache_at_top_level(time_range: TimeRange, cache_func: Callable):
    try:
        cache_func(time_range=time_range)
        return (True, time_range, None)
        
    except Exception as e:
        return (False, time_range, str(e))


def cache_multiprocess(cache_func: Callable, time_ranges: List[TimeRange], workers: int = 10):
    worker_func = partial(
        _cache_at_top_level,
        cache_func=cache_func,
    )

    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(worker_func, time_ranges)

    # Collect results and report progress
    successful_batches = 0
    failed_batches = 0
    
    for success, calc_range, error_msg in results:
        calc_t_from, calc_t_to = calc_range.to_datetime()
        if success:
            successful_batches += 1
            print(f"  ✅ Completed batch: {calc_t_from.date()} to {calc_t_to.date()}")
        else:
            failed_batches += 1
            print(f"  ❌ Failed batch: {calc_t_from.date()} to {calc_t_to.date()}: {error_msg}")
    
    print(f"\n  Parallel processing summary:")
    print(f"    Successful batches: {successful_batches}")
    print(f"    Failed batches: {failed_batches}")
    
    return successful_batches, failed_batches


def read_multithreaded(read_func: Callable, time_range: TimeRange, max_workers: int = 10):
    t_from, t_to = time_range.to_datetime()
    daily_ranges = split_t_range(t_from, t_to, interval=datetime.timedelta(days=1))

    d_from_to_df = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(read_func, d_from, d_to): (d_from, d_to) 
                  for d_from, d_to in daily_ranges}
        
        # Collect results
        for future in as_completed(futures):
            d_from, df = future.result()
            if df is not None and not df.empty:
                d_from_to_df[d_from] = df

    dfs = [d_from_to_df[d_from] for d_from, _ in daily_ranges if d_from in d_from_to_df]
    # Concatenate all dataframes
    if dfs:
        result_df = pd.concat(dfs)
    else:
        result_df = pd.DataFrame()

    return result_df
