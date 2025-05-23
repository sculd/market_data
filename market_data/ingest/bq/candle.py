import pandas as pd
import datetime
import logging
import pytz

from ...util import time as util_time

from . import common
from .common import AGGREGATION_MODE


_query_template_take_latest = """
    WITH LATEST AS (
    SELECT timestamp, symbol, max(ingestion_timestamp) AS max_ingestion_timestamp
    FROM `{t_id}` 
    WHERE TRUE
    AND timestamp >= "{t_str_from}"
    AND timestamp < "{t_str_to}"
    GROUP BY timestamp, symbol
    )

    SELECT T.timestamp, T.symbol, open, high, low, close, volume
    FROM `{t_id}` AS T JOIN 
        LATEST ON T.timestamp = LATEST.timestamp AND T.symbol = LATEST.symbol 
        AND IFNULL(T.ingestion_timestamp, PARSE_TIMESTAMP("%c", "Thu Dec 25 07:30:00 2008")) = IFNULL(LATEST.max_ingestion_timestamp, PARSE_TIMESTAMP("%c", "Thu Dec 25 07:30:00 2008"))
    WHERE TRUE
    AND T.timestamp >= "{t_str_from}"
    AND T.timestamp < "{t_str_to}"
    ORDER BY T.timestamp ASC
"""

_query_template_collect_all_updates = """
    SELECT timestamp, symbol, open, high, low, close, volume
    FROM `{t_id}` 
    WHERE TRUE
    AND timestamp >= "{t_str_from}"
    AND timestamp < "{t_str_to}"
    ORDER BY timestamp ASC
"""


def _get_query_template(aggregation_mode: AGGREGATION_MODE) -> str:
    if aggregation_mode == AGGREGATION_MODE.TAKE_LASTEST:
        return _query_template_take_latest
    elif aggregation_mode == AGGREGATION_MODE.COLLECT_ALL_UPDATES:
        return _query_template_collect_all_updates
    else:
        raise Exception(f"{aggregation_mode=} is not valid")


def _fetch_minute_candle_datetime(t_id: str, aggregation_mode: AGGREGATION_MODE, t_from: datetime.datetime, t_to: datetime.datetime) -> pd.DataFrame:
    logging.debug(
        f'fetching minute candles from {t_from} to {t_to}')

    query_template = _get_query_template(aggregation_mode)
    query_str = query_template.format(
        t_id=t_id,
        t_str_from=util_time.t_to_bq_t_str(t_from),
        t_str_to=util_time.t_to_bq_t_str(t_to),
    )
    df = common.run_query(query_str, timestamp_columnname="timestamp")
    df = df.reset_index().drop_duplicates(["timestamp", "symbol"]).set_index("timestamp")
    return df


def fetch_minute_candle(
        t_id: str,
        aggregation_mode: AGGREGATION_MODE,
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

    return _fetch_minute_candle_datetime(t_id, aggregation_mode, t_from, t_to)

