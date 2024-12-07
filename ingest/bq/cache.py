import pandas as pd
import datetime
import pytz
import logging
import typing
import os

from . import candle
from . import orderbook1l
from . import common
from .common import AGGREGATION_MODE
from ..util import time as util_time
from ..gcs.util import get_gcsblobname, download_gcs_blob, upload_file_to_public_gcs_bucket, if_blob_exist
from google.cloud import storage


# the cache will be stored per day.
_cache_interval = datetime.timedelta(days=1)
_cache_base_path = os.path.expanduser('~/algo_cache')
try:
    os.mkdir(_cache_base_path)
except FileExistsError:
    pass
_label_market_data = "market_data"

# the timezone of dividing range into days.
_cache_timezone = pytz.timezone('America/New_York')

_timestamp_index_name = 'timestamp'


def to_filename(basedir: str, label: str, t_id: str, aggregation_mode: AGGREGATION_MODE, t_from: datetime.datetime, t_to: datetime.datetime) -> str:
    t_str_from = t_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_str_to = t_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    fn = os.path.join(basedir, label, f"{t_id}_{aggregation_mode}", f"{t_str_from}_{t_str_to}.parquet")
    dir = os.path.dirname(fn)
    try:
        os.makedirs(dir, exist_ok=True)
    except FileExistsError:
        pass

    return fn


def anchor_to_begin_of_day(t: datetime.datetime) -> datetime.datetime:
    return _cache_timezone.localize(datetime.datetime(year=t.year, month=t.month, day=t.day, hour=0, minute=0, second=0))


def split_t_range(t_from: datetime.datetime, t_to: datetime.datetime) -> typing.List[typing.Tuple[datetime.datetime, datetime.datetime]]:
    """
    split an interval into a list of daily intervals.
    """
    ret = []
    t1, t2 = anchor_to_begin_of_day(t_from), anchor_to_begin_of_day(t_from + _cache_interval)
    ret.append((t1, t2))
    while t2 < t_to:
        t1, t2 = t2, t2 + _cache_interval
        t1, t2 = anchor_to_begin_of_day(t1), anchor_to_begin_of_day(t2)
        ret.append((t1, t2))

    last = (ret[-1][0], min(ret[-1][1], t_to))
    ret[-1] = last
    return ret


def is_exact_cache_interval(t_from: datetime.datetime, t_to: datetime.datetime) -> bool:
    """
    returns if the range is one-day interval anchored at the zero hour.
    """
    t_from_plus_a_day = anchor_to_begin_of_day(t_from + datetime.timedelta(days=1))
    if t_to != t_from_plus_a_day:
        return False

    def at_begin_of_day(t: datetime.datetime) -> bool:
        return t.hour == 0 and t.minute == 0 and t.second == 0

    if not at_begin_of_day(t_from):
        return False

    if not at_begin_of_day(t_to):
        return False

    return True


def _cache_df_daily(df: pd.DataFrame, label: str, aggregation_mode: AGGREGATION_MODE, t_id: str, t_from: datetime.datetime, t_to: datetime.datetime, overwrite=True):
    if not is_exact_cache_interval(t_from, t_to):
        logging.info(f"{t_from}-{t_to} does not match {_cache_interval=} thus will not be cached.")
        return None

    if len(df) == 0:
        logging.info(f"df for {t_from}-{t_to} is empty thus will not be cached.")
        return None

    filename = to_filename(_cache_base_path, _label_market_data, t_id, aggregation_mode, t_from, t_to)
    if os.path.exists(filename):
        logging.info(f"{filename} already exists.")
        if overwrite:
            logging.info(f"and would overwrite it.")
            df.to_parquet(filename)
        else:
            logging.info(f"and would not write it.")
    else:
        df.to_parquet(filename)

    blob_name = get_gcsblobname(label, t_id, t_from, t_to)
    upload_file_to_public_gcs_bucket(filename, blob_name, rewrite=overwrite)


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
        blob_name = get_gcsblobname(label, t_id, t_from, t_to)
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
        dataset_mode: common.DATASET_MODE,
        export_mode: common.EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        label: str = _label_market_data,
        resample_interval_str = None,
        t_from: datetime.datetime = None,
        t_to: datetime.datetime = None,
        epoch_seconds_from: int = None,
        epoch_seconds_to: int = None,
        date_str_from: str = None,
        date_str_to: str = None,
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
    t_ranges = split_t_range(t_from, t_to)
    df_concat = None
    for t_range in t_ranges:
        df = _fetch_from_daily_cache(t_id, label, aggregation_mode, t_range[0], t_range[1])
        if df is None:
            continue
        if df_concat is None:
            df_concat = df.copy()
        else:
            df_concat = pd.concat([df_concat, df])
        del df

    if df_concat is not None and resample_interval_str is not None:
        df_concat = df_concat.reset_index().groupby('symbol').apply(
            lambda x: x.set_index(_timestamp_index_name).resample(resample_interval_str).asfreq().ffill()).drop(columns='symbol').reset_index()
    return df_concat


def query_and_cache(
        dataset_mode: common.DATASET_MODE,
        export_mode: common.EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
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
    df_concat = None
    for i, t_range in enumerate(t_ranges):
        if skip_first_day and i == 0:
            continue
        df_cache = _fetch_from_daily_cache(t_id, label, aggregation_mode, t_range[0], t_range[1])
        if overwirte_cache or df_cache is None:
            df = fetch_function(t_id, aggregation_mode, t_from=t_range[0], t_to=t_range[1])
            _cache_df_daily(df, label, aggregation_mode, t_id, t_range[0], t_range[1])
        else:
            df = df_cache

        if df is None:
            continue
        if len(df) == 0:
            continue
        if _timestamp_index_name in df.columns:
            df = df.set_index(_timestamp_index_name)
        if df_concat is None:
            df_concat = df.copy()
        else:
            df_concat = pd.concat([df_concat, df])
        del df

    if df_concat is not None and resample_interval_str is not None:
        df_concat = df_concat.reset_index().groupby('symbol').apply(
            lambda x: x.set_index(_timestamp_index_name).resample(resample_interval_str).asfreq().ffill()).drop(columns='symbol').reset_index()
    return df_concat


