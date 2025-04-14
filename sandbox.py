import logging
import os, sys, datetime, time
from dotenv import load_dotenv
import importlib
import pandas as pd
import market_data.util.time

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

# Get project ID from environment variable
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
if not _PROJECT_ID:
    logging.warning("GOOGLE_CLOUD_PROJECT environment variable not set. Please check your .env file.")


import market_data.ingest.bq.common
import market_data.ingest.bq.cache

import market_data.util
import market_data.util.time

'''
df = market_data.ingest.bq.cache.query_and_cache(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    date_str_from='2025-03-01', date_str_to='2025-04-01')

#df = ingest.bq.cache.query_and_cache(ingest.bq.util.DATASET_MODE.BITHUMB, ingest.bq.util.EXPORT_MODE.ORDERBOOK_LEVEL1, date_str_from='2024-03-21', date_str_to='2024-03-22')

#df = fetch_minute_candle(t_id, date_str_from='2024-03-15', date_str_to='2024-03-16')
print(df)
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2025-01-01', date_str_to='2025-01-03',
    )

import market_data.feature.feature
t1 = time.time()
df_feature = market_data.feature.feature.create_features(df)
t2 = time.time()
print(f'time: {t2 - t1}')
#'''


time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2024-01-03',
    )


import market_data.feature.cache_reader
'''
featuers_df = market_data.feature.cache_reader.read_multi_feature_cache(
    feature_labels_params=["returns", "ema"],
    time_range=time_range,
    dataset_mode=market_data.ingest.bq.common.DATASET_MODE.OKX,
    export_mode=market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE,
    aggregation_mode=market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
)
print(featuers_df)
#'''


# Import the registry module directly
registry_module = importlib.import_module('market_data.feature.registry')

# Access the registry directly
registry = getattr(registry_module, '_FEATURE_REGISTRY', {})

print(f"Total registered features: {len(registry)}")

# Print each registered feature
for label in sorted(registry.keys()):
    feature_class = registry[label]
    print(f"- {label}: {feature_class.__name__}")


time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2024-01-05',
    )

'''
featuers_df = market_data.feature.cache_reader.read_multi_feature_cache(
    feature_labels_params=['returns'],
    time_range=time_range,
    dataset_mode=market_data.ingest.bq.common.DATASET_MODE.OKX, 
    export_mode=market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE,
    aggregation_mode=market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
)
print(featuers_df.shape)
#'''

'''
import market_data.feature.sequential_feature
importlib.reload(market_data.feature.sequential_feature)

t1 = time.time()
df_feature_seq = market_data.feature.sequential_feature.sequentialize_feature(featuers_df)
t2 = time.time()
print(f'time: {t2 - t1}')
print(df_feature_seq.shape)
seq_s_df = df_feature_seq[df_feature_seq.index.get_level_values('symbol') == 'ZRX-USDT-SWAP']
print(seq_s_df[["return_1"]].tail(1).values)
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-04-01',
    )
import market_data.feature.cache_writer
for label in sorted(registry.keys()):
    if label not in ["ema"]:
        continue
    market_data.feature.cache_writer.cache_feature_cache(
        label,
        market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
        time_range=time_range,
    )
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-04-01',
    )
import market_data.feature.cache_writer
for label in sorted(registry.keys()):
    if label not in ["ema"]:
        continue
    market_data.feature.cache_writer.cache_seq_feature_cache(
        label,
        market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
        time_range=time_range,
    )
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-04-01',
    )
import market_data.feature.cache_writer
for label in sorted(registry.keys()):
    if label not in ["ema"]:
        continue
    market_data.feature.cache_writer.cache_feature_cache(
        label,
        market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
        time_range=time_range,
    )
'''


'''
import market_data.feature.cache_feature
market_data.feature.cache_feature.calculate_and_cache_features(
      market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
      time_range=time_range,
    )
#'''

'''
import market_data.target.cache_target
market_data.target.cache_target.calculate_and_cache_targets(
      market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
      time_range=time_range,
    )
#'''

'''
import market_data.machine_learning.cache_resample
market_data.machine_learning.cache_resample.calculate_and_cache_resampled(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
)
#'''

#'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-04-01',
    )


import market_data.machine_learning.cache_ml_data
market_data.machine_learning.cache_ml_data.calculate_and_cache_ml_data(
    market_data.ingest.bq.common.DATASET_MODE.OKX, market_data.ingest.bq.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.bq.common.AGGREGATION_MODE.TAKE_LASTEST,
    time_range=time_range,
)
#'''


print('done')

