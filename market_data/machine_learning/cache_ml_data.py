import pandas as pd
import logging
from typing import Optional, List
import os
import datetime
from pathlib import Path
from dataclasses import asdict

from market_data.feature.target import TargetParams, DEFAULT_FORWARD_PERIODS, DEFAULT_TP_VALUES, DEFAULT_SL_VALUES
from market_data.feature.feature import FeatureParams, DEFAULT_RETURN_PERIODS, DEFAULT_EMA_PERIODS
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from market_data.util.time import TimeRange
from market_data.machine_learning.resample import ResampleParams
from market_data.machine_learning.ml_data import prepare_ml_data
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

logger = logging.getLogger(__name__)

# The base directory for cache
CACHE_BASE_PATH = os.path.expanduser('~/algo_cache/ml_data')
Path(CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

def _get_resample_params_dir(
    resample_params: ResampleParams = None,
    feature_params: FeatureParams = None,
    target_params: TargetParams = None
) -> str:
    """
    Convert all ML data parameters to a directory name string.
    
    Uses the default values when None is passed to ensure consistent directory paths.
    """
    resample_params = resample_params or ResampleParams()
    feature_params = feature_params or FeatureParams(
        return_periods=DEFAULT_RETURN_PERIODS,
        ema_periods=DEFAULT_EMA_PERIODS,
        add_btc_features=True
    )
    target_params = target_params or TargetParams(
        forward_periods=DEFAULT_FORWARD_PERIODS,
        tp_values=DEFAULT_TP_VALUES,
        sl_values=DEFAULT_SL_VALUES
    )
    
    # Flatten all parameters into a single dictionary with prefixed keys
    params_dict = {}
    
    # Add resample parameters with 'r_' prefix
    for key, value in asdict(resample_params).items():
        params_dict[f'r_{key}'] = value
        
    # Add feature parameters with 'f_' prefix
    for key, value in asdict(feature_params).items():
        params_dict[f'f_{key}'] = value
        
    # Add target parameters with 't_' prefix
    for key, value in asdict(target_params).items():
        params_dict[f't_{key}'] = value
        
    return params_to_dir_name(params_dict)

def _calculate_daily_ml_data(
    date: datetime.datetime,
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    feature_params: FeatureParams,
    target_params: TargetParams,
    resample_params: ResampleParams,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache ML data for a single day.
    
    Args:
        date: The date to calculate ML data for
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        feature_params: Feature calculation parameters
        target_params: Target calculation parameters
        resample_params: Resampling parameters
        overwrite_cache: Whether to overwrite existing cache files
    """
    # Create time range for the specific day
    t_from = date
    t_to = date + datetime.timedelta(days=1)
    time_range = TimeRange(t_from, t_to)
    
    # Prepare ML data for the day
    ml_data_df = prepare_ml_data(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        time_range=time_range,
        feature_params=feature_params,
        target_params=target_params,
        resample_params=resample_params
    )
    
    if ml_data_df is None or len(ml_data_df) == 0:
        logger.warning(f"No ML data available for {date}")
        return
    
    # Cache the data
    logger.info(f"Caching ML data for {date}")
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{aggregation_mode}"
    params_dir = _get_resample_params_dir(resample_params, feature_params, target_params)
    
    cache_data_by_day(
        df=ml_data_df,
        label="ml_data",
        t_from=t_from,
        t_to=t_to,
        params_dir=params_dir,
        overwrite=overwrite_cache,
        dataset_id=dataset_id,
        cache_base_path=CACHE_BASE_PATH,
        warm_up_period_days=0,
    )
    
    logger.info(f"Successfully cached ML data for {date} with {len(ml_data_df)} rows")

def calculate_and_cache_ml_data(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_params: FeatureParams = None,
    target_params: TargetParams = None,
    resample_params: ResampleParams = None,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache ML data by preparing the data and caching it daily.
    
    This function:
    1. Splits the time range into individual days
    2. For each day, prepares ML data using prepare_ml_data
    3. Caches each daily piece
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_params: Feature calculation parameters. If None, uses default parameters.
        target_params: Target calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        overwrite_cache: Whether to overwrite existing cache files
    """
    # Use default parameters if none provided
    feature_params = feature_params or FeatureParams(
        return_periods=DEFAULT_RETURN_PERIODS,
        ema_periods=DEFAULT_EMA_PERIODS,
        add_btc_features=True
    )
    target_params = target_params or TargetParams(
        forward_periods=DEFAULT_FORWARD_PERIODS,
        tp_values=DEFAULT_TP_VALUES,
        sl_values=DEFAULT_SL_VALUES
    )
    resample_params = resample_params or ResampleParams()
    
    # Get time range
    t_from, t_to = time_range.to_datetime()
    
    # Calculate number of days
    current_date = t_from
    
    # Process each day
    while current_date < t_to:
        logger.info(f"Processing day {current_date}")
        
        _calculate_daily_ml_data(
            date=current_date,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            feature_params=feature_params,
            target_params=target_params,
            resample_params=resample_params,
            overwrite_cache=overwrite_cache
        )

        current_date += datetime.timedelta(days=1)

def load_cached_ml_data(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_params: FeatureParams = None,
    target_params: TargetParams = None,
    resample_params: ResampleParams = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load cached ML data for a specific time range.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_params: Feature calculation parameters. If None, uses default parameters.
        target_params: Target calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        columns: Optional list of columns to load. If None, loads all columns.
        
    Returns:
        DataFrame with ML data, or empty DataFrame if no data is available
    """
    # Use default parameters if none provided
    feature_params = feature_params or FeatureParams()
    target_params = target_params or TargetParams()
    resample_params = resample_params or ResampleParams()
    
    # Get dataset ID for cache path
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{aggregation_mode}"
    
    # Get parameters directory name
    params_dir = _get_resample_params_dir(resample_params, feature_params, target_params)
    
    # Read from cache
    return read_from_cache_generic(
        label="ml_data",
        params_dir=params_dir,
        time_range=time_range,
        columns=columns,
        dataset_id=dataset_id,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        cache_base_path=CACHE_BASE_PATH
    )
