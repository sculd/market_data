import datetime
import pandas as pd
import os
from pathlib import Path

import market_data.ingest.bq.common as bq_common
import market_data.ingest.common as common
import market_data.ingest.bq.candle as candle
import market_data.ingest.bq.orderbook1l as orderbook1l
import market_data.util.time as util_time
from market_data.util.cache.path import get_cache_base_path
from market_data.util.cache.time import split_t_range


# the cache will be stored per day.
_cache_interval = datetime.timedelta(days=1)
_cache_base_path = get_cache_base_path()
try:
    os.mkdir(_cache_base_path)
except FileExistsError:
    pass

_label_market_data = "market_data"


def query_and_cache(
        dataset_mode: common.DATASET_MODE,
        export_mode: common.EXPORT_MODE,
        aggregation_mode: common.AGGREGATION_MODE,
        label: str = _label_market_data,
        resample_interval_str = None,
        t_from: datetime.datetime = None,
        t_to: datetime.datetime = None,
        epoch_seconds_from: int = None,
        epoch_seconds_to: int = None,
        date_str_from: str = None,
        date_str_to: str = None,
        overwirte_cache = False,
        skip_first_day = False,
        ) -> pd.DataFrame:
    t_from, t_to = util_time.to_t(
        t_from=t_from,
        t_to=t_to,
        epoch_seconds_from=epoch_seconds_from,
        epoch_seconds_to=epoch_seconds_to,
        date_str_from=date_str_from,
        date_str_to=date_str_to,
    )

    t_id = common.get_full_table_id(dataset_mode, export_mode)
    if export_mode == common.EXPORT_MODE.BY_MINUTE:
        fetch_function = candle.fetch_minute_candle

    elif export_mode == common.EXPORT_MODE.ORDERBOOK_LEVEL1:
        fetch_function = orderbook1l.fetch_orderbook1l
    else:
        raise Exception(f"{dataset_mode=}, {export_mode=} is not available for ingestion")

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
        df_cache = _fetch_from_daily_cache(t_id, label, aggregation_mode, t_range[0], t_range[1])
        if overwirte_cache or df_cache is None:
            df = fetch_function(t_id, aggregation_mode, t_from=t_range[0], t_to=t_range[1])
            _cache_daily_df(df, label, aggregation_mode, t_id, t_range[0], t_range[1])
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
