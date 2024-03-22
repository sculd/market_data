import pandas as pd, numpy as np
import datetime
import logging
import typing

import util.time

import ingest.bq.util

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


def _fetch_minute_candle_datetime(t_id, t_from, t_to) -> pd.DataFrame:
    t_str_from = t_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_str_to = t_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    logging.debug(
        f'fetching prices from {t_from} to {t_to}')

    query_str = _query_minute_candle_template.format(
        t_id=t_id, t_str_from=t_str_from, t_str_to=t_str_to
    )
    print(query_str)

    bq_query_job = ingest.bq.util.get_big_query_client().query(query_str)
    df = bq_query_job.to_dataframe().set_index('timestamp')
    df.index = df.index.tz_convert('America/New_York')

    logging.debug(f'fetched {len(df)} rows')
    del bq_query_job
    return df


def _to_filename_prefix(t_id: str, t_from: datetime.datetime, t_to: datetime.datetime) -> str:
    t_str_from = t_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_str_to = t_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    return f'{t_id}_{t_str_from}_{t_str_to}'


def _split_t_range(t_from: datetime.datetime, t_to: datetime.datetime, interval: datetime.timedelta = datetime.timedelta(days=10)) -> typing.List[typing.Tuple[datetime.datetime, datetime.datetime]]:
    ret = []
    t1, t2 = t_from, t_from + interval
    ret.append((t1, t2))
    while t2 < t_to:
        t1, t2 = t2, t2 + interval
        ret.append((t1, t2))

    last = (ret[-1][0], min(ret[-1][1], t_to))
    ret[-1] = last
    return ret


def fetch_minute_candle(
        t_id: str,
        t_from: datetime.datetime = None,
        t_to: datetime.datetime = None,
        epoch_seconds_from: int = None,
        epoch_seconds_to: int = None,
        date_str_from: str = None,
        date_str_to: str = None,
        ) -> pd.DataFrame:
    t_from, t_to = util.time.to_t(
        t_from=t_from,
        t_to=t_to,
        epoch_seconds_from=epoch_seconds_from,
        epoch_seconds_to=epoch_seconds_to,
        date_str_from=date_str_from,
        date_str_to=date_str_to,
    )

    return _fetch_minute_candle_datetime(t_id, t_from, t_to)
