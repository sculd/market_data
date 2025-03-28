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

import ingest.bq.common
import ingest.bq.cache

# Get project ID from environment variable
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
if not _PROJECT_ID:
    logging.warning("GOOGLE_CLOUD_PROJECT environment variable not set. Please check your .env file.")




#'''

df = ingest.bq.cache.query_and_cache(
    ingest.bq.common.DATASET_MODE.OKX, ingest.bq.common.EXPORT_MODE.BY_MINUTE, ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    date_str_from='2024-03-18', date_str_to='2025-01-01')

#df = ingest.bq.cache.query_and_cache(ingest.bq.util.DATASET_MODE.BITHUMB, ingest.bq.util.EXPORT_MODE.ORDERBOOK_LEVEL1, date_str_from='2024-03-21', date_str_to='2024-03-22')

#df = fetch_minute_candle(t_id, date_str_from='2024-03-15', date_str_to='2024-03-16')
print(df)
#'''

print('done')

