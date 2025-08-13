import datetime
import logging
import os
import typing

import pandas as pd

import market_data.util.cache.common
import market_data.util.cache.path
from market_data.util.cache.time import is_exact_cache_interval, split_t_range
from market_data.util.time import TimeRange


def read_daily_from_local_cache(
    folder_path: str,
    t_from: datetime.datetime,
    t_to: datetime.datetime,
    columns: typing.List[str] = None,
) -> typing.Optional[pd.DataFrame]:
    if not is_exact_cache_interval(t_from, t_to):
        logging.info(f"{t_from} to {t_to} does not match {market_data.util.cache.common.cache_interval=} thus not read from cache.")
        return None
    filename = market_data.util.cache.path.to_local_filename(folder_path, t_from, t_to)
    if not os.path.exists(filename):
        return None

    df = pd.read_parquet(filename)
    if len(df) == 0:
        return None

    if columns is None:
        return df
    else:
        columns = [c for c in columns if c in df.columns]
        return df[columns]


def read_from_local_cache(
    folder_path: str,
    resample_interval_str = None,
    time_range: TimeRange = None,
    columns: typing.List[str] = None,
) -> pd.DataFrame:
    t_from, t_to = time_range.to_datetime()
    t_ranges = split_t_range(t_from, t_to, interval=datetime.timedelta(days=1))
    df_concat: pd.DataFrame = None
    df_list = []
    # concat every 10 files to free up memory more frequently
    concat_interval = 10

    def concat_batch():
        nonlocal df_concat, df_list
        if len(df_list) == 0:
            return
        
        # Memory-efficient concatenation: avoid creating intermediate copies
        if df_concat is None:
            # First batch - just concatenate the list
            df_concat = pd.concat(df_list, copy=False)
        else:
            # Subsequent batches - use list concatenation to minimize copies
            df_batch = pd.concat(df_list, copy=False)
            df_concat = pd.concat([df_concat, df_batch], copy=False)
            # Explicitly delete the batch to free memory immediately
            del df_batch
        
        # Clear the list and explicitly delete references
        for df in df_list:
            del df
        df_list.clear()  # More explicit than df_list = []
        
        # Force garbage collection for large datasets
        import gc
        gc.collect()

    for t_range in t_ranges:
        df = read_daily_from_local_cache(folder_path, t_range[0], t_range[1])
        if df is None:
            continue
        df_list.append(df)

        if len(df_list) > 0 and len(df_list) % concat_interval == 0:
            concat_batch()

    concat_batch()

    if df_concat is not None and resample_interval_str is not None:
        df_concat = df_concat.reset_index().groupby('symbol').apply(
            lambda x: x.set_index(market_data.util.cache.common.timestamp_index_name).resample(resample_interval_str).asfreq().ffill()).drop(columns='symbol').reset_index()

    return df_concat if columns is None else df_concat[columns]
