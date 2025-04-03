import pandas as pd
import logging
from typing import Optional, List
import os
import datetime
from pathlib import Path

from feature.target import TargetParams, DEFAULT_FORWARD_PERIODS, DEFAULT_TP_VALUES, DEFAULT_SL_VALUES
from feature.feature import FeatureParams, DEFAULT_RETURN_PERIODS, DEFAULT_EMA_PERIODS
from ingest.bq.cache import read_from_cache_or_query_and_cache
from ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from ingest.util.time import TimeRange
from feature.cache_feature import load_cached_features, calculate_and_cache_features
from feature.cache_target import load_cached_targets, calculate_and_cache_targets
from machine_learning.cache_resample import load_cached_resampled_data, resample_and_cache_data
from machine_learning.resample import ResampleParams
from feature.cache_util import (
    split_t_range,
    params_to_dir_name,
    to_filename,
    cache_data_by_day,
    read_from_cache_generic
)

logger = logging.getLogger(__name__)

# The base directory for cache
CACHE_BASE_PATH = os.path.expanduser('~/algo_cache/ml_data')
Path(CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

def get_resample_params_dir(params: ResampleParams = None) -> str:
    """
    Convert resampling parameters to a directory name string.
    
    Uses the default values when None is passed to ensure consistent directory paths.
    """
    params = params or ResampleParams()
    params_dict = {
        'th': params.threshold,
    }
    return params_to_dir_name(params_dict)

def calculate_and_cache_data(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_params: FeatureParams = None,
    target_params: TargetParams = None,
    resample_params: ResampleParams = None,
    overwrite_cache: bool = False
) -> None:
    """
    Calculate and cache all required data for machine learning.
    
    This function:
    1. Ensures raw data is present (caches if not)
    2. Ensures feature data is present (calculates and caches if not)
    3. Ensures target data is present (calculates and caches if not)
    4. Ensures resampled data is present (calculates and caches if not)
    5. Joins feature, target and resampled data into a final DataFrame
    6. Caches the final ML data DataFrame
    
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
    
    # 1. Ensure raw data is present
    raw_df = read_from_cache_or_query_and_cache(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        time_range=time_range,
        overwirte_cache=overwrite_cache
    )
    
    if raw_df is None or len(raw_df) == 0:
        logger.error("No raw data available")
        return
    
    # 2. Ensure feature data is present
    features_df = load_cached_features(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        time_range=time_range
    )
    
    if features_df is None:
        logger.info("Calculating and caching features")
        calculate_and_cache_features(
            raw_df,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            params=feature_params,
            overwrite_cache=overwrite_cache
        )
        features_df = load_cached_features(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range
        )
    
    if features_df is None or len(features_df) == 0:
        logger.error("No feature data available")
        return
    
    # 3. Ensure target data is present
    targets_df = load_cached_targets(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        time_range=time_range
    )
    
    if targets_df is None:
        logger.info("Calculating and caching targets")
        calculate_and_cache_targets(
            raw_df,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            params=target_params,
            overwrite_cache=overwrite_cache
        )
        targets_df = load_cached_targets(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range
        )
    
    if targets_df is None or len(targets_df) == 0:
        logger.error("No target data available")
        return
    
    # 4. Ensure resampled data is present
    resampled_df = load_cached_resampled_data(
        params=resample_params,
        time_range=time_range,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode
    )
    
    if resampled_df is None:
        logger.info("Calculating and caching resampled data")
        resample_and_cache_data(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            params=resample_params,
            overwrite_cache=overwrite_cache
        )
        resampled_df = load_cached_resampled_data(
            params=resample_params,
            time_range=time_range,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode
        )
    
    if resampled_df is None or len(resampled_df) == 0:
        logger.error("No resampled data available")
        return
    
    # 5. Join feature, target and resampled data
    # First join features and targets
    combined_df = features_df.join(targets_df, how='inner')
    
    # Then filter to resampled timestamps
    resampled_timestamps = resampled_df.index
    ml_data_df = combined_df[combined_df.index.isin(resampled_timestamps)]
    
    if len(ml_data_df) == 0:
        logger.error("No data after joining features, targets and resampled timestamps")
        return
    
    # 6. Cache the final ML data DataFrame
    logger.info("Caching final ML data DataFrame")
    t_from, t_to = time_range.to_datetime()
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{aggregation_mode}"
    params_dir = get_resample_params_dir(resample_params)
    
    cache_data_by_day(
        df=ml_data_df,
        label="ml_data",
        t_from=t_from,
        t_to=t_to,
        params_dir=params_dir,
        overwrite=overwrite_cache,
        dataset_id=dataset_id
    )
    
    logger.info(f"Successfully prepared and cached ML data with {len(ml_data_df)} rows")
    return ml_data_df

def load_cached_data(
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
    Load cached data for machine learning, ensuring all required data is present.
    
    This function:
    1. Loads cached feature data
    2. Loads cached target data
    3. Loads cached resampled data
    4. Joins feature, target and resampled data
    
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
        DataFrame with features and targets, resampled at significant price movements
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
    
    # 1. Load feature data
    features_df = load_cached_features(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        time_range=time_range,
        columns=columns
    )
    
    if features_df is None or len(features_df) == 0:
        logger.error("No feature data available")
        return pd.DataFrame()
    
    # 2. Load target data
    targets_df = load_cached_targets(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        time_range=time_range,
        columns=columns
    )
    
    if targets_df is None or len(targets_df) == 0:
        logger.error("No target data available")
        return pd.DataFrame()
    
    # 3. Load resampled data
    resampled_df = load_cached_resampled_data(
        params=resample_params,
        time_range=time_range,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode
    )
    
    if resampled_df is None or len(resampled_df) == 0:
        logger.error("No resampled data available")
        return pd.DataFrame()
    
    # 4. Join feature, target and resampled data
    # First join features and targets
    combined_df = features_df.join(targets_df, how='inner')
    
    # Then filter to resampled timestamps
    resampled_timestamps = resampled_df.index
    final_df = combined_df[combined_df.index.isin(resampled_timestamps)]
    
    if len(final_df) == 0:
        logger.error("No data after joining features, targets and resampled timestamps")
        return pd.DataFrame()
    
    logger.info(f"Successfully loaded ML data with {len(final_df)} rows")
    return final_df
