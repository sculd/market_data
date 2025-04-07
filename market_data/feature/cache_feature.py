import pandas as pd
import datetime
import logging
import typing
import math
import os
from pathlib import Path
from typing import Optional, List, Union, Tuple

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
    params_to_dir_name
)
from market_data.util.cache.core import calculate_and_cache_data

# The base directory for feature cache
FEATURE_CACHE_BASE_PATH = os.path.expanduser('~/algo_cache/feature_data')
Path(FEATURE_CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

def _get_feature_params_dir(params: FeatureParams = None) -> str:
    params = params or FeatureParams()
    params_dict = {
        'rp': params.return_periods,
        'ep': params.ema_periods,
        'btc': params.add_btc_features
    }
    return params_to_dir_name(params_dict)

def _get_recommended_warm_up_days(params: FeatureParams) -> int:
    """
    Calculate the recommended warm-up period based on feature parameters.
    
    Uses the maximum window size from return_periods, ema_periods, and volatility_windows,
    plus a buffer to ensure sufficient historical data for feature calculations.
    
    Returns:
        int: Recommended number of warm-up days
    """
    # Find the maximum window period from all feature types
    max_window = max(
        max(params.return_periods),
        max(params.ema_periods),
        max(params.volatility_windows) if params.volatility_windows else 0
    )
    
    # Convert to days (assuming periods are in minutes for 24/7 markets)
    days_needed = math.ceil(max_window / (24 * 60))
    
    return days_needed

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
        warm_up_days = _get_recommended_warm_up_days(params)
    
    # Get the params directory name
    params_dir = _get_feature_params_dir(params)
    
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
    return read_from_cache_generic(
        'features', params_dir=_get_feature_params_dir(params), time_range=time_range, columns=columns,
        dataset_mode=dataset_mode, export_mode=export_mode, aggregation_mode=aggregation_mode,
        cache_base_path=FEATURE_CACHE_BASE_PATH
    )
