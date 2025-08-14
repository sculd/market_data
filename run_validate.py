import datetime
import logging
import os
import sys

from market_data.ingest.common import CacheContext
from market_data.util.time import TimeRange

if os.path.exists('./credential.json'):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), 'credential.json')
    os.environ["GOOGLE_CLOUD_PROJECT"] = "trading-290017"
else:
    print('the credential.json file does not exist')


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

import market_data.ingest.bq.cache
from market_data.ingest.common import (AGGREGATION_MODE, DATASET_MODE,
                                       EXPORT_MODE)

if __name__ == '__main__':
    market_data.ingest.bq.cache.validate_df(
        CacheContext(DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST),
        TimeRange(date_str_from='2024-01-01', date_str_to='2024-03-21'),
        label=market_data.ingest.bq.cache._label_market_data,
    )