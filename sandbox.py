import logging
import sys
import os
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables from .env file
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), 'credential.json')

import market_data.ingest.bq.common
import market_data.ingest.bq.cache

# Get project ID from environment variable
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
if not _PROJECT_ID:
    logging.warning("GOOGLE_CLOUD_PROJECT environment variable not set. Please check your .env file.")


import market_data.util
import market_data.util.time


'''
df = market_data.ingest.bq.cache.query_and_cache(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    date_str_from='2024-01-01', date_str_to='2024-01-03')

#df = ingest.bq.cache.query_and_cache(ingest.bq.util.DATASET_MODE.BITHUMB, ingest.bq.util.EXPORT_MODE.ORDERBOOK_LEVEL1, date_str_from='2024-03-21', date_str_to='2024-03-22')

#df = fetch_minute_candle(t_id, date_str_from='2024-03-15', date_str_to='2024-03-16')
print(df)
#'''

t_range = market_data.util.time.TimeRange(date_str_from='2024-01-01', date_str_to='2024-01-03')


'''
import market_data.feature.feature
import market_data.feature.cache_feature
features_df = market_data.feature.cache_feature.calculate_and_cache_features(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    params=market_data.feature.feature.FeatureParams(),
    time_range=t_range,
    calculation_batch_days=1,
    overwrite_cache=True
)
#'''

'''
import market_data.feature.target
import market_data.feature.cache_target
features_df = market_data.feature.cache_target.calculate_and_cache_targets(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    params=market_data.feature.target.TargetParams(),
    time_range=t_range,
    calculation_batch_days=1,
    overwrite_cache=True
)
#'''


#'''
import market_data.machine_learning.data
#'''

print('done')

