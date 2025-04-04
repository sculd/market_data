import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from pathlib import Path

from market_data.feature.target import TargetParams
from market_data.feature.feature import FeatureParams
from market_data.ingest.bq.cache import read_from_cache_or_query_and_cache
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
from market_data.feature.cache_feature import load_cached_features, calculate_and_cache_features
from market_data.feature.cache_target import load_cached_targets, calculate_and_cache_targets
from market_data.machine_learning.cache_resample import load_cached_resampled_data, resample_and_cache_data
from market_data.machine_learning.resample import ResampleParams

logger = logging.getLogger(__name__)

def prepare_ml_data(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_params: FeatureParams = None,
    target_params: TargetParams = None,
    resample_params: ResampleParams = None,
    overwrite_cache: bool = False
) -> pd.DataFrame:
    """
    Prepare machine learning data by ensuring all required data is present and properly joined.
    
    This function:
    1. Ensures raw data is present (caches if not)
    2. Ensures feature data is present (calculates and caches if not)
    3. Ensures target data is present (calculates and caches if not)
    4. Ensures resampled data is present (calculates and caches if not)
    5. Joins feature, target and resampled data
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_params: Feature calculation parameters. If None, uses default parameters.
        target_params: Target calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        overwrite_cache: Whether to overwrite existing cache files
        
    Returns:
        DataFrame with features and targets, resampled at significant price movements
    """
    # Use default parameters if none provided
    feature_params = feature_params or FeatureParams()
    target_params = target_params or TargetParams()
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
        return pd.DataFrame()
    
    # 2. Ensure feature data is present
    features_df = load_cached_features(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        time_range=time_range
    )
    
    if features_df is None:
        logger.info("Calculating and caching features")
        features_df = calculate_and_cache_features(
            raw_df,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            params=feature_params,
            overwrite_cache=overwrite_cache
        )
    
    if features_df is None or len(features_df) == 0:
        logger.error("No feature data available")
        return pd.DataFrame()
    
    # 3. Ensure target data is present
    targets_df = load_cached_targets(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        time_range=time_range
    )
    
    if targets_df is None:
        logger.info("Calculating and caching targets")
        targets_df = calculate_and_cache_targets(
            raw_df,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            params=target_params,
            overwrite_cache=overwrite_cache
        )
    
    if targets_df is None or len(targets_df) == 0:
        logger.error("No target data available")
        return pd.DataFrame()
    
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
        return pd.DataFrame()
    
    # 5. Join feature, target and resampled data
    # First join features and targets
    combined_df = features_df.join(targets_df, how='inner')
    
    # Then filter to resampled timestamps
    resampled_timestamps = resampled_df.index
    final_df = combined_df[combined_df.index.isin(resampled_timestamps)]
    
    if len(final_df) == 0:
        logger.error("No data after joining features, targets and resampled timestamps")
        return pd.DataFrame()
    
    logger.info(f"Successfully prepared ML data with {len(final_df)} rows")
    return final_df
