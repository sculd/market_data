import pandas as pd, numpy as np
import datetime, time
import logging
import typing

import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), 'credential.json')

import ingest.bq.util
import ingest.bq.orderbook1l
import ingest.bq.cache
import util.time

_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')




#'''

df = ingest.bq.cache.fetch_and_cache(
    ingest.bq.util.DATASET_MODE.OKX, ingest.bq.util.EXPORT_MODE.BY_MINUTE, ingest.bq.util.AGGREGATION_MODE.TAKE_LASTEST,
    date_str_from='2024-01-01', date_str_to='2024-03-21')

#df = ingest.bq.cache.fetch_and_cache(ingest.bq.util.DATASET_MODE.BITHUMB, ingest.bq.util.EXPORT_MODE.ORDERBOOK_LEVEL1, date_str_from='2024-03-21', date_str_to='2024-03-22')

#df = fetch_minute_candle(t_id, date_str_from='2024-03-15', date_str_to='2024-03-16')
print(df)
#'''

print('done')

