import pandas as pd
import datetime
import pytz
import logging
import typing
import os

import market_data.util.cache.cache_common
import market_data.util.cache.cache_read
import market_data.ingest.gcs.util
from market_data.ingest.common import DATASET_MODE, EXPORT_MODE
from market_data.util.cache.time import split_t_range
from market_data.util.time import TimeRange


# the cache will be stored per day.
_cache_interval = datetime.timedelta(days=1)

_label_market_data = "market_data"

# the timezone of dividing range into days.
_cache_timezone = pytz.timezone('America/New_York')


def get_gcsblobname(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    t: datetime.datetime,
) -> str:
    t_str = t.strftime("%Y-%m-%d")
    return os.path.join(
        _label_market_data, 
        dataset_mode.name.lower(), 
        export_mode.name.lower(), 
        f"{t_str}.parquet")


def query_and_cache(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    label: str = _label_market_data,
    resample_interval_str = None,
    time_range: TimeRange = None,
    overwirte_cache = False,
    skip_first_day = False,
    columns: typing.List[str] = None,
) -> pd.DataFrame:
    t_from, t_to = time_range.to_datetime()
    t_ranges = split_t_range(t_from, t_to)

    base_label = market_data.util.cache.cache_common.get_label(dataset_mode, export_mode)
    folder_path = os.path.join(market_data.util.cache.cache_common.cache_base_path, label, base_label)


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

    for i, (t_from, t_to) in enumerate(t_ranges):
        if skip_first_day and i == 0:
            continue

        local_filename = market_data.util.cache.cache_common.to_local_filename(folder_path, t_from, t_to)
        df_cache = market_data.util.cache.cache_read.read_daily_from_local_cache(
            folder_path,
            t_from,
            t_to,
            columns=columns,
        )

        if overwirte_cache or df_cache is None:
            blob_name = get_gcsblobname(dataset_mode, export_mode, t_from)
            blob_exist = market_data.ingest.gcs.util.if_blob_exist(blob_name)
            if not blob_exist:
                logging.info(f"For gcs, {blob_name=} does not exist.")
                continue
            market_data.ingest.gcs.util.download_gcs_blob(blob_name, local_filename)

            df = market_data.util.cache.cache_read.read_daily_from_local_cache(
                folder_path,
                t_from,
                t_to,
                columns=columns,
            )
        else:
            df = df_cache

        if df is None:
            continue
        if len(df) == 0:
            continue

        if market_data.util.cache.cache_common.timestamp_index_name in df.columns:
            df = df.set_index(market_data.util.cache.cache_common.timestamp_index_name)

        df_list.append(df)

        if len(df_list) > 0 and len(df_list) % concat_interval == 0:
            concat_batch()

    concat_batch()

    if df_concat is not None and resample_interval_str is not None:
        df_concat = df_concat.reset_index().groupby('symbol').apply(
            lambda x: x.set_index(market_data.util.cache.cache_common.timestamp_index_name).resample(resample_interval_str).asfreq().ffill()).drop(columns='symbol').reset_index()

    return df_concat


def read_from_local_cache_or_query_and_cache(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    label: str = _label_market_data,
    resample_interval_str = None,
    time_range: TimeRange = None,
    columns: typing.List[str] = None,
    overwirte_cache = False,
    skip_first_day = False,
) -> pd.DataFrame:
    base_label = market_data.util.cache.cache_common.get_label(dataset_mode, export_mode)
    folder_path = os.path.join(market_data.util.cache.cache_common.cache_base_path, label, base_label)
    df = market_data.util.cache.cache_read.read_from_local_cache(
            folder_path,
            resample_interval_str=resample_interval_str,
            time_range = time_range,
    )
    if df is not None:
        return df

    return query_and_cache(
            dataset_mode,
            export_mode,
            label = label,
            resample_interval_str=resample_interval_str,
            time_range = time_range,
            columns=columns,
    )

