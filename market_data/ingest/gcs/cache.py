import datetime
import logging
import os
import typing

import pandas as pd

import market_data.ingest.gcs.util
import market_data.util.cache.read
import market_data.util.cache.path
from market_data.ingest.common import DATASET_MODE, EXPORT_MODE, CacheContext
from market_data.util.cache.time import split_t_range
from market_data.util.time import TimeRange

_label_market_data = "market_data"


def _get_gcsblobname(
    cache_context: CacheContext,
    t: datetime.datetime,
) -> str:
    t_str = t.strftime("%Y-%m-%d")
    return os.path.join(
        _label_market_data, 
        cache_context.dataset_mode.name.lower(), 
        cache_context.export_mode.name.lower(), 
        f"{t_str}.parquet")


def _convert_raw_to_by_minute(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_sorted_df = raw_df
    if 'timestamp' in raw_sorted_df.index.names:
        raw_sorted_df = raw_sorted_df.reset_index()
    raw_sorted_df = raw_sorted_df.sort_values(['symbol', 'timestamp', 'ingestion_timestamp'], ascending=[True, True, False])
    
    columns_to_drop = []
    if 'timestamp_seconds' in raw_sorted_df.columns:
        columns_to_drop.append('timestamp_seconds')
    if 'ingestion_timestamp' in raw_sorted_df.columns:
        columns_to_drop.append('ingestion_timestamp')
    # Keep only the latest row per (symbol, timestamp)
    minute_df = raw_sorted_df.drop_duplicates(subset=['symbol', 'timestamp'], keep='first') \
                  .sort_values(['timestamp', 'symbol']) \
                  .drop(columns=columns_to_drop) \
                  .set_index(['timestamp'])

    return minute_df


def cache(
    cache_context: CacheContext,
    time_range: TimeRange = None,
    overwrite_cache = False,
    skip_first_day = False,
) -> pd.DataFrame:
    t_from, t_to = time_range.to_datetime()
    t_ranges = split_t_range(t_from, t_to)

    folder_path = cache_context.get_market_data_path()

    for i, (t_from, t_to) in enumerate(t_ranges):
        if skip_first_day and i == 0:
            continue

        local_filename = market_data.util.cache.path.to_local_filename(folder_path, t_from, t_to)
        if overwrite_cache or not os.path.exists(local_filename):
            blob_name = _get_gcsblobname(cache_context, t_from)
            blob_exist = market_data.ingest.gcs.util.if_blob_exist(blob_name)
            if not blob_exist:
                logging.info(f"For gcs, {blob_name=} does not exist.")
                continue
            market_data.ingest.gcs.util.download_gcs_blob(blob_name, local_filename)
        else:
            logging.info(f"For local cache, {local_filename=} exists.")

        if cache_context.dataset_mode == DATASET_MODE.OKX and cache_context.export_mode == EXPORT_MODE.RAW:
            logging.info(f"For okx raw data, convert to by minute.")
            by_minute_folder_path = cache_context.with_params({"export_mode": EXPORT_MODE.BY_MINUTE}).get_market_data_path()
            by_minute_local_filename = market_data.util.cache.path.to_local_filename(by_minute_folder_path, t_from, t_to)
            if overwrite_cache or not os.path.exists(by_minute_local_filename):
                raw_df = market_data.util.cache.read.read_daily_from_local_cache(
                    folder_path,
                    t_from,
                    t_to,
                )
                by_minute_df = _convert_raw_to_by_minute(raw_df)
                by_minute_df.to_parquet(by_minute_local_filename)
            else:
                logging.info(f"For local cache, {by_minute_local_filename=} exists.")


def read_from_local_cache_or_query_and_cache(
    cache_context: CacheContext,
    time_range: TimeRange,
    resample_interval_str = None,
    columns: typing.List[str] = None,
    overwrite_cache = False,
) -> pd.DataFrame:
    folder_path = cache_context.get_market_data_path()
    df = market_data.util.cache.read.read_from_local_cache(
        folder_path,
        resample_interval_str=resample_interval_str,
        time_range = time_range,
        columns = columns,
    )
    if df is not None:
        return df

    cache(
        cache_context,
        time_range = time_range,
        overwrite_cache = overwrite_cache,
    )
    return market_data.util.cache.read.read_from_local_cache(
        folder_path,
        resample_interval_str=resample_interval_str,
        time_range = time_range,
    )

