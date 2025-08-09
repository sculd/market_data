# Lint as: python3

import os, datetime , logging
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas as pd
import market_data.util.cache.cache_common

_client = None


_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')

_WRITE_TIMEOUT_SECONDS = 5

_WRITE_QUEUE_SIZE_THRESHOLD_DEFAULT = 500
_WRITE_QUEUE_SIZE_THRESHOLD_PER_MINUTE_NO_OVERWRITE = 100
_WRITE_QUEUE_SIZE_THRESHOLD_LIQUIDITY_IMBALANCE = 100
_LOCK_TIMEOUT_SECONDS = 10

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
    df = bq_query_job.to_dataframe()
    df = df.drop_duplicates([timestamp_columnname, 'symbol'])
    del bq_query_job
    logging.debug(f'fetched {len(df)} rows')
    if len(df) == 0:
        return df

    df = df.set_index(timestamp_columnname)
    df.index = df.index.tz_convert('America/New_York')

    return df


def get_full_table_id(dataset_mode, export_mode):
    return '{p}.{l}'.format(
        p=_PROJECT_ID, 
        l=market_data.util.cache.cache_common.get_label(dataset_mode, export_mode),
        )


