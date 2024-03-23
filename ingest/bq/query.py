import pandas as pd, numpy as np
import datetime
import logging
import typing
from enum import Enum

import util.time

import ingest.bq.util
import ingest.bq.candle
import ingest.bq.orderbook1l
from ingest.bq.util import AGGREGATION_MODE


def fetch(
        dataset_mode: ingest.bq.util.DATASET_MODE,
        export_mode: ingest.bq.util.EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
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

    t_id = ingest.bq.util.get_full_table_id(dataset_mode, export_mode)
    if export_mode == ingest.bq.util.EXPORT_MODE.BY_MINUTE:
        return ingest.bq.candle.fetch_minute_candle(t_id, aggregation_mode, t_from=t_from, t_to=t_to)
    elif export_mode == ingest.bq.util.EXPORT_MODE.ORDERBOOK_LEVEL1:
        return ingest.bq.orderbook1l.fetch_orderbook1l(t_id, aggregation_mode, t_from=t_from, t_to=t_to)
    else:
        raise Exception(f"{dataset_mode=}, {export_mode=} is not available for ingestion")
