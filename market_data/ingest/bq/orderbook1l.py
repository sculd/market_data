import datetime
import logging

import pandas as pd

import market_data.ingest.bq.common as common
import market_data.util.time as util_time
from market_data.ingest.common import AGGREGATION_MODE

_query_template_take_latest = """
    WITH LATEST AS (
    SELECT timestamp_floored_by_minute, symbol, max(ingestion_timestamp) AS max_ingestion_timestamp
    FROM `{t_id}` 
    WHERE TRUE
    AND timestamp >= "{t_str_from}"
    AND timestamp < "{t_str_to}"
    GROUP BY timestamp_floored_by_minute, symbol
    )

    SELECT T.timestamp_floored_by_minute as timestamp, T.symbol, price_ask, qty_ask, price_bid, qty_bid
    FROM `{t_id}` AS T JOIN 
        LATEST ON T.timestamp_floored_by_minute = LATEST.timestamp_floored_by_minute AND T.symbol = LATEST.symbol AND T.ingestion_timestamp = LATEST.max_ingestion_timestamp
    WHERE TRUE
    AND T.timestamp >= "{t_str_from}"
    AND T.timestamp < "{t_str_to}"
    ORDER BY T.timestamp ASC
"""


_query_template_collect_all_updates = """
    SELECT timestamp_floored_by_minute as timestamp, symbol, price_ask, qty_ask, price_bid, qty_bid
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


def _fetch_minute_orderbook1l_datetime(t_id: str, aggregation_mode: AGGREGATION_MODE, t_from: datetime.datetime, t_to: datetime.datetime) -> pd.DataFrame:
    logging.debug(
        f'fetching prices from {t_from} to {t_to}')

    query_template = _get_query_template(aggregation_mode)
    query_str = query_template.format(
        t_id=t_id,
        t_str_from=util_time.t_to_bq_t_str(t_from),
        t_str_to=util_time.t_to_bq_t_str(t_to),
    )
    df = common.run_query(query_str, timestamp_columnname="timestamp")
    return df


def fetch_orderbook1l(
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

    return _fetch_minute_orderbook1l_datetime(t_id, aggregation_mode, t_from, t_to)

