import pandas as pd
import datetime
import logging
import typing
import math
import os
from pathlib import Path
from typing import Optional, List, Union, Tuple
import numpy as np
from dataclasses import dataclass, field

from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
from market_data.feature.feature import create_features, FeatureParams
from market_data.util.cache.time import (
    split_t_range,
)
from market_data.util.cache.dataframe import (
    cache_data_by_day,
    read_from_cache_generic,
)
from market_data.util.cache.path import (
    params_to_dir_name,
    get_cache_base_path
)
from market_data.util.cache.core import calculate_and_cache_data

# Global paths configuration - use configurable base path
FEATURE_CACHE_BASE_PATH = os.path.join(get_cache_base_path(), 'feature_data')
Path(FEATURE_CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

def calculate_and_cache_features(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        params: FeatureParams = None,
        time_range: TimeRange = None,
        calculation_batch_days: int = 1,
        warm_up_days: Optional[int] = None,
        overwrite_cache: bool = True,
        ) -> None:
    """
    Calculate and cache features for a specified time range.
    
    Parameters:
    -----------
    dataset_mode : DATASET_MODE
        Dataset mode (LIVE, REPLAY, etc.)
    export_mode : EXPORT_MODE
        Export mode (OHLC, TICKS, etc.)
    aggregation_mode : AGGREGATION_MODE
        Aggregation mode (MIN_1, MIN_5, etc.)
    params : FeatureParams, optional
        Feature calculation parameters. If None, uses default parameters.
    time_range : TimeRange, optional
        Time range for calculation. If None, must provide individual time parameters.
    calculation_batch_days : int, optional
        Number of days to calculate for in each batch, default 1
    warm_up_days : int, optional
        Number of warm-up days for calculation, default None (auto-calculated)
    overwrite_cache : bool, optional
        If True, overwrite existing cache files, default True
    """
    # Create default params if None
    params = params or FeatureParams()
    
    # If warm_up_days not provided, calculate based on feature parameters
    if warm_up_days is None:
        warm_up_days = params.get_warm_up_days()
    
    # Get the params directory name using the method from FeatureParams
    params_dir = params.get_params_dir()
    
    # Use the generic calculate_and_cache_data function
    calculate_and_cache_data(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        params=params,
        time_range=time_range,
        calculation_batch_days=calculation_batch_days,
        warm_up_days=warm_up_days,
        overwrite_cache=overwrite_cache,
        label="features",
        calculate_batch_fn=create_features,
        cache_base_path=FEATURE_CACHE_BASE_PATH,
        params_dir=params_dir
    )

def load_cached_features(
        params: FeatureParams = None,
        time_range: TimeRange = None,
        columns: typing.List[str] = None,
        dataset_mode: DATASET_MODE = None,
        export_mode: EXPORT_MODE = None,
        aggregation_mode: AGGREGATION_MODE = None
    ) -> pd.DataFrame:
    """
    Load cached features for a specific time range
    
    Parameters:
    -----------
    params : FeatureParams, optional
        Feature calculation parameters. If None, uses default parameters.
    time_range : TimeRange, optional
        Time range for feature calculation. If None, must provide individual time parameters.
    columns : typing.List[str], optional
        Columns to load from cache. If None, all columns are loaded.
    dataset_mode : DATASET_MODE, optional
        Dataset mode for cache path. If None, uses default dataset mode.
    export_mode : EXPORT_MODE, optional
        Export mode for cache path. If None, uses default export mode.
    aggregation_mode : AGGREGATION_MODE, optional
        Aggregation mode for cache path. If None, uses default aggregation mode.
    """
    # Create default params if None
    params = params or FeatureParams()
    
    return read_from_cache_generic(
        'features', params_dir=params.get_params_dir(), time_range=time_range, columns=columns,
        dataset_mode=dataset_mode, export_mode=export_mode, aggregation_mode=aggregation_mode,
        cache_base_path=FEATURE_CACHE_BASE_PATH
    )
