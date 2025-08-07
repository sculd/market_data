import pandas as pd
import datetime
import pytz
import os

from market_data.ingest.common import DATASET_MODE, EXPORT_MODE
from market_data.util.cache.path import get_cache_base_path

_DATASET_ID_EQUITY = 'equity'
_DATASET_ID_FOREX = 'forex'
_DATASET_ID_BINANCE = 'binance'
_DATASET_ID_BINANCE_ORDERBOOK = 'binance'
_DATASET_ID_GEMINI = 'gemini'
_DATASET_ID_OKCOIN = 'okcoin'
_DATASET_ID_KRAKEN = 'kraken'
_DATASET_ID_OKX = 'okx'
_DATASET_ID_CEX = 'cex'
_DATASET_ID_BITHUMB = 'bithumb'
_DATASET_ID_FOREX_IBKR = 'forex_ibkr'
_DATASET_ID_STOCK_HIGH_VOLATILITY = 'stock_high_volatility'
_DATASET_ID_FUTURE_FIRSTDATA_ABSOLUTE = 'future_firstdata_absolute'
_DATASET_ID_FUTURE_FIRSTDATA_RATIO = 'future_firstdata_ratio'
_DATASET_ID_FUTURE_FIRSTDATA_NOADJUST = 'future_firstdata_noadjust'

_TABLE_ID_BY_MINUTE = 'by_minute'
_TABLE_ID_RAW = 'raw'
_TABLE_ID_ORDERBOOK_LEVEL10 = 'orderbook_level10'
_TABLE_ID_ORDERBOOK_LEVEL1 = 'orderbook_level1'
_TABLE_ID_ORDERBOOK = 'orderbook'
_TABLE_ID_ORDERBOOK_LIQUIDITY_IMBALANCE = 'liquidity_imbalance'


# the cache will be stored per day.
cache_interval = datetime.timedelta(days=1)
cache_base_path = get_cache_base_path()
try:
    os.mkdir(cache_base_path)
except FileExistsError:
    pass

# the timezone of dividing range into days.
cache_timezone = pytz.timezone('America/New_York')

timestamp_index_name = 'timestamp'


def to_local_filename(folder_path: str, t_from: datetime.datetime, t_to: datetime.datetime) -> str:
    t_str_from = t_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_str_to = t_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    fn = os.path.join(folder_path, f"{t_str_from}_{t_str_to}.parquet")
    dir = os.path.dirname(fn)
    try:
        os.makedirs(dir, exist_ok=True)
    except FileExistsError:
        pass

    return fn


def get_label(dataset_mode, export_mode):
    ds_id = _DATASET_ID_EQUITY
    if dataset_mode is DATASET_MODE.EQUITY:
        ds_id = _DATASET_ID_EQUITY
    elif dataset_mode is DATASET_MODE.FOREX:
        ds_id = _DATASET_ID_FOREX
    elif dataset_mode is DATASET_MODE.BINANCE:
        ds_id = _DATASET_ID_BINANCE
    elif dataset_mode is DATASET_MODE.GEMINI:
        ds_id = _DATASET_ID_GEMINI
    elif dataset_mode is DATASET_MODE.OKCOIN:
        ds_id = _DATASET_ID_OKCOIN
    elif dataset_mode is DATASET_MODE.KRAKEN:
        ds_id = _DATASET_ID_KRAKEN
    elif dataset_mode is DATASET_MODE.OKX:
        ds_id = _DATASET_ID_OKX
    elif dataset_mode is DATASET_MODE.CEX:
        ds_id = _DATASET_ID_CEX
    elif dataset_mode is DATASET_MODE.BITHUMB:
        ds_id = _DATASET_ID_BITHUMB
    elif dataset_mode is DATASET_MODE.FOREX_IBKR:
        ds_id = _DATASET_ID_FOREX_IBKR
    elif dataset_mode is DATASET_MODE.STOCK_HIGH_VOLATILITY:
        ds_id = _DATASET_ID_STOCK_HIGH_VOLATILITY

    t_id = _TABLE_ID_BY_MINUTE
    if export_mode is EXPORT_MODE.BY_MINUTE:
        t_id = _TABLE_ID_BY_MINUTE
    elif export_mode is EXPORT_MODE.RAW:
        t_id = _TABLE_ID_RAW
    elif export_mode is EXPORT_MODE.ORDERBOOK_LEVEL10:
        t_id = _TABLE_ID_ORDERBOOK_LEVEL10
    elif export_mode is EXPORT_MODE.ORDERBOOK_LEVEL1:
        t_id = _TABLE_ID_ORDERBOOK_LEVEL1
    elif export_mode is EXPORT_MODE.ORDERBOOK:
        t_id = _TABLE_ID_ORDERBOOK
    elif export_mode is EXPORT_MODE.ORDERBOOK_LIQUIDITY_IMBALANCE:
        t_id = _TABLE_ID_ORDERBOOK_LIQUIDITY_IMBALANCE
    return '{d}.{t}'.format(d=ds_id, t=t_id)


