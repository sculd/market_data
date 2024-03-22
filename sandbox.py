import pandas as pd, numpy as np
import datetime, time
import logging
import typing

import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), 'credential.json')
from google.cloud import bigquery
import util.time

_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')


_query_minute_candle_template = """
    WITH LATEST AS (
    SELECT timestamp, max(ingestion_timestamp) AS max_ingestion_timestamp
    FROM `{t_id}` 
    WHERE TRUE
    AND timestamp >= "{t_str_from}"
    AND timestamp < "{t_str_to}"
    GROUP BY timestamp
    )

    SELECT *
    FROM `trading-290017.market_data_okx.by_minute` AS T JOIN 
        LATEST ON T.timestamp = LATEST.timestamp AND T.ingestion_timestamp = LATEST.max_ingestion_timestamp
    WHERE TRUE
    AND T.timestamp >= "{t_str_from}"
    AND T.timestamp < "{t_str_to}"
    ORDER BY T.timestamp ASC
"""

_client = None

def get_big_query_client():
  global _client
  if _client is None:
      _client = bigquery.Client(project = _PROJECT_ID)
  return _client


def _fetch_minute_candle_datetime(t_id, t_from, t_to) -> pd.DataFrame:
    t_str_from = t_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_str_to = t_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    logging.debug(
        f'fetching prices from {t_from} to {t_to}')

    query_str = _query_minute_candle_template.format(
        t_id=t_id, t_str_from=t_str_from, t_str_to=t_str_to
    )
    print(query_str)

    bq_query_job = get_big_query_client().query(query_str)
    df = bq_query_job.to_dataframe().set_index('timestamp')
    df.index = df.index.tz_convert('America/New_York')

    logging.debug(f'fetched {len(df)} rows')
    del bq_query_job
    return df


def _to_t(
        t_from: datetime.datetime = None,
        t_to: datetime.datetime = None,
        epoch_seconds_from: int = None,
        epoch_seconds_to: int = None,
        date_str_from: str = None,
        date_str_to: str = None,
        ) -> typing.Tuple[datetime.datetime, datetime.datetime]:
    if t_from is not None and t_to is not None:
        pass
    elif epoch_seconds_from is not None and epoch_seconds_to is not None:
        t_from = util.time.epoch_seconds_to_datetime(epoch_seconds_from)
        t_to = util.time.epoch_seconds_to_datetime(epoch_seconds_to)
    elif date_str_from is not None and date_str_to is not None:
        t_from = util.time.date_str_et_to_t(date_str_from)
        t_to = util.time.date_str_et_to_t(date_str_to)
    else:
        raise Exception("the time range must be specified.")

    return t_from, t_to


def fetch_minute_candle(
        t_id: str,
        t_from: datetime.datetime = None,
        t_to: datetime.datetime = None,
        epoch_seconds_from: int = None,
        epoch_seconds_to: int = None,
        date_str_from: str = None,
        date_str_to: str = None,
        ) -> pd.DataFrame:
    t_from, t_to = _to_t(
        t_from=t_from,
        t_to=t_to,
        epoch_seconds_from=epoch_seconds_from,
        epoch_seconds_to=epoch_seconds_to,
        date_str_from=date_str_from,
        date_str_to=date_str_to,
    )

    return _fetch_minute_candle_datetime(t_id, t_from, t_to)


import ingest.bq.util
t_id = ingest.bq.util.get_full_table_id(ingest.bq.util.DATASET_MODE.OKX, ingest.bq.util.EXPORT_MODE.BY_MINUTE)
df = fetch_minute_candle(t_id, date_str_from='2024-03-15', date_str_to='2024-03-16')

print(df)

print('done')