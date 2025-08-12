import datetime

import pandas as pd

import market_data.ingest.bq.candle as candle
import market_data.ingest.bq.common as common
import market_data.ingest.bq.orderbook1l as orderbook1l
import market_data.util.time as util_time
from market_data.ingest.common import AGGREGATION_MODE


def fetch(
        dataset_mode: common.DATASET_MODE,
        export_mode: common.EXPORT_MODE,
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

    t_id = common.get_full_table_id(dataset_mode, export_mode)
    if export_mode == common.EXPORT_MODE.BY_MINUTE:
        return candle.fetch_minute_candle(t_id, aggregation_mode, t_from=t_from, t_to=t_to)
    elif export_mode == common.EXPORT_MODE.ORDERBOOK_LEVEL1:
        return orderbook1l.fetch_orderbook1l(t_id, aggregation_mode, t_from=t_from, t_to=t_to)
    else:
        raise Exception(f"{dataset_mode=}, {export_mode=} is not available for ingestion")
