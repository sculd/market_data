import pandas as pd, numpy as np
import datetime
import logging
import typing

import util.time

import ingest.bq.util

_query_template = """
    WITH LATEST AS (
    SELECT timestamp, max(ingestion_timestamp) AS max_ingestion_timestamp
    FROM `{t_id}` 
    WHERE TRUE
    AND timestamp >= "{t_str_from}"
    AND timestamp < "{t_str_to}"
    GROUP BY timestamp
    )

    SELECT T.timestamp, T.symbol, open, high, low, close, volume
    FROM `{t_id}` AS T JOIN 
        LATEST ON T.timestamp = LATEST.timestamp AND T.ingestion_timestamp = LATEST.max_ingestion_timestamp
    WHERE TRUE
    AND T.timestamp >= "{t_str_from}"
    AND T.timestamp < "{t_str_to}"
    ORDER BY T.timestamp ASC
"""


def _fetch_minute_candle_datetime(t_id, t_from, t_to) -> pd.DataFrame:
    logging.debug(
        f'fetching prices from {t_from} to {t_to}')

    query_str = _query_template.format(
        t_id=t_id,
        t_str_from=util.time.t_to_bq_t_str(t_from),
        t_str_to=util.time.t_to_bq_t_str(t_to),
    )
    df = ingest.bq.util.run_query(query_str, timestamp_columnname="timestamp")
    return df


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

