import datetime
import importlib
import logging
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

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

import market_data.ingest.common
import market_data.machine_learning.resample.calc
import market_data.util
import market_data.util.time
from market_data.feature.label import FeatureLabel, FeatureLabelCollection
from market_data.feature.param import SequentialFeatureParam
from market_data.ingest.common import AGGREGATION_MODE, DATASET_MODE, EXPORT_MODE, CacheContext
from market_data.machine_learning.ml_data.cache import (
    calculate_and_cache_ml_data,
    load_cached_and_select_columns_ml_data,
    load_cached_ml_data,
)
from market_data.machine_learning.ml_data.calc import calculate as calculate_ml_data
from market_data.machine_learning.ml_data.param import MlDataParam
from market_data.machine_learning.resample.calc import CumSumResampleParams
from market_data.machine_learning.target_resample.cache import calculate_and_cache_targets_resampled
from market_data.target.param import TargetParams, TargetParamsBatch

cache_context = market_data.ingest.common.CacheContext(
    DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST)

'''
df = market_data.ingest.bq.cache.query_and_cache(
    DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST,
    date_str_from='2025-03-01', date_str_to='2025-04-01')

#df = ingest.bq.cache.query_and_cache(ingest.bq.util.DATASET_MODE.BITHUMB, ingest.bq.util.EXPORT_MODE.ORDERBOOK_LEVEL1, date_str_from='2024-03-21', date_str_to='2024-03-22')

#df = fetch_minute_candle(t_id, date_str_from='2024-03-15', date_str_to='2024-03-16')
print(df)
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2025-01-01', date_str_to='2025-01-03',
    )

#'''

time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-07-01',
    )

target_params_batch = TargetParamsBatch(
        target_params_list=[TargetParams(forward_period=int(period), tp_value=float(tp), sl_value=float(tp)) 
            for period in [5, 10, 30]
            for tp in [0.015, 0.03, 0.05]]
        )

'''
s = calculate_and_cache_targets_resampled(
    cache_context=cache_context,
    time_range=time_range,
    target_params_batch=target_params_batch,
    resample_params=CumSumResampleParams(price_col = 'close', threshold = 0.1),
)
print(s)
#'''


'''
ml_data = calculate_ml_data(
    CacheContext(market_data.ingest.common.DATASET_MODE.OKX, market_data.ingest.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.common.AGGREGATION_MODE.TAKE_LATEST),
    time_range=time_range,
    feature_collection = FeatureLabelCollection(),
    target_params_batch=target_params_batch,
    resample_params=CumSumResampleParams(price_col = 'close', threshold = 0.1),
)
print(ml_data.shape)
#'''


'''
calculate_and_cache_ml_data(
    CacheContext(market_data.ingest.common.DATASET_MODE.OKX, market_data.ingest.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.common.AGGREGATION_MODE.TAKE_LATEST),
    time_range=time_range,
    feature_collection = FeatureLabelCollection(),
    target_params_batch=target_params_batch,
    resample_params=CumSumResampleParams(price_col = 'close', threshold = 0.1),
)
#'''


#'''
ml_data_param = MlDataParam(
    feature_collection=FeatureLabelCollection(),
    target_params_batch=target_params_batch,
    resample_params=CumSumResampleParams(price_col = 'close', threshold = 0.1),
)
ml_data = load_cached_and_select_columns_ml_data(
    CacheContext(market_data.ingest.common.DATASET_MODE.OKX, market_data.ingest.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.common.AGGREGATION_MODE.TAKE_LATEST),
    time_range=time_range,
    ml_data_param = ml_data_param,
)
print(ml_data.shape)

ml_data_param = MlDataParam(
    feature_collection=FeatureLabelCollection().with_feature_label(FeatureLabel("bollinger")),
    target_params_batch=target_params_batch,
    resample_params=CumSumResampleParams(price_col = 'close', threshold = 0.1),
)
ml_data = load_cached_and_select_columns_ml_data(
    CacheContext(market_data.ingest.common.DATASET_MODE.OKX, market_data.ingest.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.common.AGGREGATION_MODE.TAKE_LATEST),
    time_range=time_range,
    ml_data_param = ml_data_param,
)
print(ml_data.shape)
#'''

'''
feature_labels = market_data.feature.registry.list_registered_features('all')
feature_collection = FeatureLabelCollection()
for feature_label in feature_labels:
    feature_collection = feature_collection.with_feature_label(FeatureLabel(feature_label))

    
calculate_and_cache_ml_data(
    CacheContext(market_data.ingest.common.DATASET_MODE.OKX, market_data.ingest.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.common.AGGREGATION_MODE.TAKE_LATEST),
    time_range=time_range,
    feature_collection = feature_collection,
    target_params_batch=target_params_batch,
    resample_params=CumSumResampleParams(price_col = 'close', threshold = 0.1),
)
#'''


'''
ml_data = calculate_and_cache_ml_data(
    CacheContext(market_data.ingest.common.DATASET_MODE.OKX, market_data.ingest.common.EXPORT_MODE.BY_MINUTE, market_data.ingest.common.AGGREGATION_MODE.TAKE_LATEST),
    time_range=time_range,
    feature_collection = feature_collection,
    target_params_batch=target_params_batch,
    resample_params=CumSumResampleParams(price_col = 'close', threshold = 0.1),
    overwrite_cache = False,
)

print(ml_data.shape)
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2024-01-03',
    )


import market_data.feature.cache_reader
from market_data.feature.label import FeatureLabel, FeatureLabelCollection

feature_label_collection = FeatureLabelCollection().with_feature_label(FeatureLabel("returns")).with_feature_label(FeatureLabel("ema"))
featuers_df = market_data.feature.cache_reader.read_multi_features(
    feature_label_collection=feature_label_collection,
    time_range=time_range,
    cache_context=cache_context,
)
print(featuers_df)
#'''

'''
feature_label_collection = FeatureLabelCollection().with_feature_label(FeatureLabel("returns")).with_feature_label(FeatureLabel("ema"))
featuers_df = market_data.feature.cache_reader.read_multi_features(
    feature_label_collection=feature_label_collection,
    time_range=time_range,
    cache_context=cache_context,
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
    market_data.feature.cache_writer.cache_feature(
        label,
        DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST,
        time_range=time_range,
    )
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-04-01',
    )
import market_data.feature.cache_writer
for label in sorted(registry.keys()):
    if label in ["bollinger"]:
        continue
    market_data.feature.cache_writer.cache_sequential(
        FeatureLabel(label),
        DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST,
        time_range=time_range,
    )
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-04-01',
    )
import market_data.feature.cache_writer
for label in sorted(registry.keys()):
    if label not in ["bollinger"]:
        continue
    market_data.feature.cache_writer.cache_feature(
        label,
        DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST,
        time_range=time_range,
    )
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-02', date_str_to='2024-01-04',
    )
import market_data.machine_learning.ml_data
ml_data_df = market_data.machine_learning.ml_data.prepare_sequential_ml_data(
    DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST,
    time_range=time_range,
)
print(ml_data_df)
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-04-01',
    )
import market_data.machine_learning.ml_data.cache
market_data.machine_learning.ml_data.cache.calculate_and_cache_sequential_ml_data(
    DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST,
    time_range=time_range,
)
#'''


'''
import market_data.feature.cache_feature
market_data.feature.cache_feature.calculate_and_cache_features(
      DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST,
      time_range=time_range,
    )
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-01-01',
    )

import market_data.target.cache_target
market_data.target.cache_target.calculate_and_cache_targets(
      DATASET_MODE.FOREX_IBKR, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.COLLECT_ALL_UPDATES,
      time_range=time_range,
      params = market_data.target.cache_target.TargetParamsBatch(
        target_params_list=[market_data.target.calc.TargetParams(forward_period=period, tp_value=tp, sl_value=tp) 
            for period in market_data.target.calc.DEFAULT_FORWARD_PERIODS 
            for tp in [0.01, 0.015, 0.02, 0.03]]
      ),
    )
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-04-01',
    )

import market_data.machine_learning.resample.cache
market_data.machine_learning.resample.cache.calculate_and_cache_resampled(
    DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST,
    time_range=time_range,
    params=market_data.machine_learning.resample.calc.ResampleParams(price_col = 'close', threshold = 0.07,),
)
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2024-01-03',
    )

from market_data.feature.registry import list_registered_features
feature_labels = list_registered_features(security_type="all")


import market_data.machine_learning.ml_data
from market_data.target.calc import TargetParamsBatch

ml_data_df = market_data.machine_learning.ml_data.calculate(
    cache_context,
    time_range=time_range,
    feature_label_params = feature_labels,
    resample_params = market_data.machine_learning.resample.calc.ResampleParams(
        threshold=0.05,
    ),
    target_params_batch = TargetParamsBatch(),
)
print(ml_data_df)
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2024-01-04',
    )

import market_data.ingest.missing_data_finder

cache_context = market_data.ingest.common.CacheContext(
    DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST
)
from market_data.feature.label import FeatureLabel

feature_label_obj = FeatureLabel('returns', None)
missing = market_data.ingest.missing_data_finder.check_missing_feature_resampled_data(
    cache_context=cache_context,
    time_range=time_range,
    feature_label_obj=feature_label_obj,
    resample_params=market_data.machine_learning.resample.calc.ResampleParams(
        threshold=0.05,
    ),
    #seq_param=None,
)
print(missing)
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2025-04-01', date_str_to='2025-04-05',
    )

import market_data.machine_learning.ml_data.cache
market_data.machine_learning.ml_data.cache.calculate_and_cache_ml_data(
    DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST,
    time_range=time_range,
    resample_params=market_data.machine_learning.resample.calc.ResampleParams(price_col = 'close', threshold = 0.07,),
)
#'''

'''
time_range = market_data.util.time.TimeRange(
    date_str_from='2024-01-01', date_str_to='2025-04-01',
    )

import market_data.machine_learning.ml_data.cache
market_data.machine_learning.ml_data.cache.calculate_and_cache_ml_data(
    DATASET_MODE.OKX, EXPORT_MODE.BY_MINUTE, AGGREGATION_MODE.TAKE_LATEST,
    time_range=time_range,
    seq_param = SequentialFeatureParam(),
)
#'''


print('done')

