import datetime
import logging
import os
import typing

import pandas as pd
import pytz

import market_data.ingest.bq.candle as candle
import market_data.ingest.bq.common as common
import market_data.ingest.bq.orderbook1l as orderbook1l
from market_data.ingest.common import AGGREGATION_MODE, CacheContext
from market_data.ingest.gcs.util import (download_gcs_blob, if_blob_exist,
                                         upload_file_to_gcs)
from market_data.util.cache.path import get_cache_base_path
from market_data.util.cache.time import (anchor_to_begin_of_day,
                                         is_exact_cache_interval,
                                         split_t_range)
from market_data.util.time import TimeRange

# the cache will be stored per day.
_cache_interval = datetime.timedelta(days=1)
_cache_base_path = get_cache_base_path()
try:
    os.mkdir(_cache_base_path)
except FileExistsError:
    pass
_label_market_data = "market_data"

# the timezone of dividing range into days.
_cache_timezone = pytz.timezone('America/New_York')

_timestamp_index_name = 'timestamp'


def _to_gcsblobname(label: str, t_id: str, t_from: datetime.datetime, t_to: datetime.datetime) -> str:
    t_str_from = t_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_str_to = t_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    return os.path.join(label, f"{t_id}", f"{t_str_from}_{t_str_to}.parquet")


def to_filename(basedir: str, label: str, t_id: str, aggregation_mode: AGGREGATION_MODE, t_from: datetime.datetime, t_to: datetime.datetime) -> str:
    t_str_from = t_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_str_to = t_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    fn = os.path.join(basedir, label, f"{t_id}_{str(aggregation_mode)}", f"{t_str_from}_{t_str_to}.parquet")
    dir = os.path.dirname(fn)
    try:
        os.makedirs(dir, exist_ok=True)
    except FileExistsError:
        pass

    return fn


def _cache_daily_df(df: pd.DataFrame, label: str, aggregation_mode: AGGREGATION_MODE, t_id: str, t_from: datetime.datetime, t_to: datetime.datetime, overwrite=True):
    """
    Cache a DataFrame that covers one-day range exactly.
    """
    if not is_exact_cache_interval(t_from, t_to):
        logging.info(f"{t_from}-{t_to} does not match {_cache_interval=} thus will not be cached.")
        return None

    if len(df) == 0:
        logging.info(f"df for {t_from}-{t_to} is empty thus will not be cached.")
        return None

    filename = to_filename(_cache_base_path, label, t_id, aggregation_mode, t_from, t_to)
    if os.path.exists(filename):
        logging.info(f"{filename} already exists.")
        if overwrite:
            logging.info(f"and would overwrite it.")
            df.to_parquet(filename)
        else:
            logging.info(f"and would not write it.")
    else:
        df.to_parquet(filename)

    blob_name = _to_gcsblobname(label, t_id, t_from, t_to)
    upload_file_to_gcs(filename, blob_name, rewrite=overwrite)


def cache_df(
        df: pd.DataFrame,
        cache_context: CacheContext,
        label: str = _label_market_data,
        overwrite = False,
        warm_up_period_days = 1,
) -> None:
    """
    Cache a DataFrame.

    Chop it into daily pieces and cache them one-by-one.
    skip_first_day is present as the first day of data is often incomplete due to warm-up period.
    """
    if len(df) == 0:
        logging.info(f"df is empty for {label} thus will be skipped.")
        return
    t_id = common.get_full_table_id(cache_context.dataset_mode, cache_context.export_mode)

    def _split_df_by_day() -> typing.List[pd.DataFrame]:
        dfs = [group[1] for group in df.groupby(df.index.get_level_values(_timestamp_index_name).date)]
        return dfs

    df_dailys = _split_df_by_day()
    for i, df_daily in enumerate(df_dailys):
        if i < warm_up_period_days:
            continue

        timestamps = df_daily.index.get_level_values(_timestamp_index_name).unique()
        t_begin = anchor_to_begin_of_day(timestamps[0])
        t_end = anchor_to_begin_of_day(t_begin + _cache_interval)
        _cache_daily_df(df_daily, label, cache_context.aggregation_mode, t_id, t_begin, t_end, overwrite=overwrite)
        del df_daily


def _fetch_from_daily_cache(
        t_id: str,
        label: str,
        aggregation_mode: AGGREGATION_MODE,
        t_from: datetime.datetime,
        t_to: datetime.datetime,
        columns: typing.List[str] = None,
) -> typing.Optional[pd.DataFrame]:
    if not is_exact_cache_interval(t_from, t_to):
        logging.info(f"{t_from} to {t_to} does not match {_cache_interval=} thus not read from cache.")
        return None
    filename = to_filename(_cache_base_path, label, t_id, aggregation_mode, t_from, t_to)
    if not os.path.exists(filename):
        blob_name = _to_gcsblobname(label, t_id, t_from, t_to)
        blob_exist = if_blob_exist(blob_name)
        logging.info(f"{filename=} does not exist in local cache. For gcs, {blob_exist=}.")
        if blob_exist:
            download_gcs_blob(blob_name, filename)
        else:
            return None
    df = pd.read_parquet(filename)
    if len(df) == 0:
        return None

    if columns is None:
        return df
    else:
        columns = [c for c in columns if c in df.columns]
        return df[columns]


def read_from_cache(
        cache_context: CacheContext,
        time_range: TimeRange,
        label: str = _label_market_data,
        resample_interval_str = None,
        columns: typing.List[str] = None,
        ) -> pd.DataFrame:
    t_from, t_to = time_range.to_datetime()

    t_id = common.get_full_table_id(cache_context.dataset_mode, cache_context.export_mode)
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

    for i, t_range in enumerate(t_ranges):
        df = _fetch_from_daily_cache(t_id, label, cache_context.aggregation_mode, t_range[0], t_range[1])
        if df is None:
            continue
        df_list.append(df)

        if len(df_list) > 0 and len(df_list) % concat_interval == 0:
            concat_batch()

    concat_batch()

    if df_concat is not None and resample_interval_str is not None:
        df_concat = df_concat.reset_index().groupby('symbol').apply(
            lambda x: x.set_index(_timestamp_index_name).resample(resample_interval_str).asfreq().ffill()).drop(columns='symbol').reset_index()

    return df_concat if columns is None else df_concat[columns]


def query_and_cache(
        cache_context: CacheContext,
        time_range: TimeRange,
        label: str = _label_market_data,
        resample_interval_str = None,
        overwrite_cache = False,
        skip_first_day = False,
        ) -> pd.DataFrame:
    t_from, t_to = time_range.to_datetime()

    t_id = common.get_full_table_id(cache_context.dataset_mode, cache_context.export_mode)
    if cache_context.export_mode == common.EXPORT_MODE.BY_MINUTE:
        fetch_function = candle.fetch_minute_candle

    elif cache_context.export_mode == common.EXPORT_MODE.ORDERBOOK_LEVEL1:
        fetch_function = orderbook1l.fetch_orderbook1l
    else:
        raise Exception(f"{cache_context.dataset_mode=}, {cache_context.export_mode=} is not available for ingestion")

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

    for i, t_range in enumerate(t_ranges):
        if skip_first_day and i == 0:
            continue
        df_cache = _fetch_from_daily_cache(t_id, label, cache_context.aggregation_mode, t_range[0], t_range[1])
        if overwrite_cache or df_cache is None:
            df = fetch_function(t_id, cache_context.aggregation_mode, t_from=t_range[0], t_to=t_range[1])
            _cache_daily_df(df, label, cache_context.aggregation_mode, t_id, t_range[0], t_range[1])
        else:
            df = df_cache

        if df is None:
            continue
        if len(df) == 0:
            continue
        if _timestamp_index_name in df.columns:
            df = df.set_index(_timestamp_index_name)

        df_list.append(df)

        if len(df_list) > 0 and len(df_list) % concat_interval == 0:
            concat_batch()

    concat_batch()


    if df_concat is not None and resample_interval_str is not None:
        df_concat = df_concat.reset_index().groupby('symbol').apply(
            lambda x: x.set_index(_timestamp_index_name).resample(resample_interval_str).asfreq().ffill()).drop(columns='symbol').reset_index()
    return df_concat


def read_from_cache_or_query_and_cache(
        cache_context: CacheContext,
        time_range: TimeRange,
        label: str = _label_market_data,
        resample_interval_str = None,
        overwrite_cache = False,
        ) -> pd.DataFrame:
    df = read_from_cache(
            cache_context,
            time_range,
            label = label,
            resample_interval_str=resample_interval_str,
    )
    if df is not None:
        return df

    return query_and_cache(
            cache_context,
            time_range,
            label = label,
            resample_interval_str=resample_interval_str,
            overwrite_cache = overwrite_cache,
    )


def validate_df(
        cache_context: CacheContext,
        time_range: TimeRange,
        label: str = _label_market_data,
        ) -> None:
    t_id = common.get_full_table_id(cache_context.dataset_mode, cache_context.export_mode)
    t_from, t_to = time_range.to_datetime()
    t_ranges = split_t_range(t_from, t_to)
    for t_range in t_ranges:
        t_from, t_to = t_range[0], t_range[-1]
        filename = to_filename(_cache_base_path, label, t_id, cache_context.aggregation_mode, t_from, t_to)
        if not os.path.exists(filename):
            blob_name = _to_gcsblobname(label, t_id, t_from, t_to)
            blob_exist = if_blob_exist(blob_name)
            logging.info(f"{filename=} does not exist in local cache. For gcs, {blob_exist=}.")
            if blob_exist:
                download_gcs_blob(blob_name, filename)
                print(f'after download from gcs, file exist: {os.path.exists(filename)}')
            else:
                print(f'{filename} does not exist in the gcs bucket either ({blob_name} does not exist')


