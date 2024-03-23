# Lint as: python3

import os, datetime , logging
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas as pd

_client = None


_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')

_DATASET_ID_EQUITY = 'market_data'
_DATASET_ID_FOREX = 'market_data_forex'
_DATASET_ID_BINANCE = 'market_data_binance'
_DATASET_ID_BINANCE_ORDERBOOK = 'market_data_binance'
_DATASET_ID_GEMINI = 'market_data_gemini'
_DATASET_ID_OKCOIN = 'market_data_okcoin'
_DATASET_ID_KRAKEN = 'market_data_kraken'
_DATASET_ID_OKX = 'market_data_okx'
_DATASET_ID_CEX = 'market_data_cex'
_DATASET_ID_BITHUMB = 'market_data_bithumb'

_TABLE_ID_BY_MINUTE = 'by_minute'
_TABLE_ID_RAW = 'raw'
_TABLE_ID_ORDERBOOK_LEVEL10 = 'orderbook_level10'
_TABLE_ID_ORDERBOOK_LEVEL1 = 'orderbook_level1'
_TABLE_ID_ORDERBOOK = 'orderbook'
_TABLE_ID_ORDERBOOK_LIQUIDITY_IMBALANCE = 'liquidity_imbalance'

_WRITE_TIMEOUT_SECONDS = 5

_WRITE_QUEUE_SIZE_THRESHOLD_DEFAULT = 500
_WRITE_QUEUE_SIZE_THRESHOLD_PER_MINUTE_NO_OVERWRITE = 100
_WRITE_QUEUE_SIZE_THRESHOLD_LIQUIDITY_IMBALANCE = 100
_LOCK_TIMEOUT_SECONDS = 10

from enum import Enum


def get_big_query_client():
  global _client
  if _client is None:
      _client = bigquery.Client(project = _PROJECT_ID)
  return _client

def does_table_exist(table_name):
  client = get_big_query_client()
  try:
    table = client.get_table(table_name)
    if table:
      print("Found Table")
      return True
  except NotFound as error:
    print("Table does not exist")
  return False


def run_query(query_str, timestamp_columnname) -> pd.DataFrame:
    print(query_str)

    bq_query_job = get_big_query_client().query(query_str)
    df = bq_query_job.to_dataframe().set_index(timestamp_columnname)
    df.index = df.index.tz_convert('America/New_York')

    logging.debug(f'fetched {len(df)} rows')
    del bq_query_job
    return df


class EXPORT_MODE(Enum):
    BY_MINUTE = 1
    RAW = 2
    ORDERBOOK_LEVEL10 = 3
    ORDERBOOK = 4
    ORDERBOOK_LIQUIDITY_IMBALANCE = 5
    ORDERBOOK_LEVEL1 = 6

class DATASET_MODE(Enum):
    EQUITY = 1
    FOREX = 2
    BINANCE = 3
    GEMINI = 4
    OKCOIN = 5
    KRAKEN = 6
    OKX = 7
    CEX = 8
    BITHUMB = 9

def get_full_table_id(dataset_mode, export_mode):
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
    return '{p}.{d}.{t}'.format(p=_PROJECT_ID, d=ds_id, t=t_id)
