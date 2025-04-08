"""
Machine Learning Data Construction Module

This module provides functions for constructing machine learning data
by loading cached features and targets and joining them with resampled data.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
from market_data.feature.feature import FeatureParams
from market_data.target.target import TargetParams
from market_data.machine_learning.resample import ResampleParams
from market_data.feature.cache_reader import read_multi_feature_cache
from market_data.target.cache_target import load_cached_targets
from market_data.machine_learning.cache_resample import load_cached_resampled_data

logger = logging.getLogger(__name__)

def construct_ml_data(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_labels_params: List[Union[str, Tuple[str, Any]]],
    target_params: TargetParams = None,
    resample_params: ResampleParams = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Construct machine learning data by loading cached features and targets
    and joining them with resampled data without caching the results.
    
    Args:
        dataset_mode: Dataset mode (e.g., OKX)
        export_mode: Export mode (e.g., BY_MINUTE)
        aggregation_mode: Aggregation mode (e.g., TAKE_LASTEST)
        time_range: Time range to construct data for
        feature_labels_params: List of either:
            - feature labels (str) - will use default parameters
            - (label, params) tuples - will use the provided parameters
        target_params: Target parameters (if None, uses default parameters)
        resample_params: Resampling parameters (if None, uses default parameters)
        columns: Specific columns to retrieve from the cache (if None, all columns are retrieved)
        
    Returns:
        DataFrame with features and targets joined with resampled data
        
    Example:
        ```
        from market_data.feature.impl.returns import ReturnParams
        from market_data.target.impl.price_movement import PriceMovementParams
        from market_data.machine_learning.resample import ResampleParams
        from market_data.util.time import TimeRange
        
        ml_data = construct_ml_data(
            dataset_mode=DATASET_MODE.OKX,
            export_mode=EXPORT_MODE.BY_MINUTE,
            aggregation_mode=AGGREGATION_MODE.TAKE_LASTEST,
            time_range=TimeRange(t_from="2023-01-01", t_to="2023-01-31"),
            feature_labels_params=[
                "volatility",  # Using default parameters
                ("returns", ReturnParams(periods=[1, 5, 10])),
                "market_regime"
            ],
            target_params=PriceMovementParams(horizons=[10, 30, 60]),
            resample_params=ResampleParams(threshold=0.05)
        )
        ```
    """
    # Use default parameters if none provided
    target_params = target_params or TargetParams()
    resample_params = resample_params or ResampleParams()
    
    # Load resampled data
    logger.info(f"Loading resampled data for time range: {time_range}")
    resampled_df = load_cached_resampled_data(
        params=resample_params,
        time_range=time_range,
        columns=columns,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode
    )
    
    if resampled_df is None or len(resampled_df) == 0:
        logger.error("No resampled data available")
        return pd.DataFrame()
    
    # Remove OHLCV columns from resampled data since they will come from features
    if all(col in resampled_df.columns for col in ["open", "high", "low", "close", "volume"]):
        resampled_df = resampled_df.drop(["open", "high", "low", "close", "volume"], axis=1)

    # Load features data
    logger.info(f"Loading feature data for {len(feature_labels_params)} feature types")
    features_df = read_multi_feature_cache(
        feature_labels_params=feature_labels_params,
        time_range=time_range,
        columns=columns,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode
    )
    
    if features_df is None or len(features_df) == 0:
        logger.error("No feature data available")
        return pd.DataFrame()

    # Join features with resampled data
    logger.info("Joining features with resampled data")
    feature_resampled_df = resampled_df.reset_index().set_index(["timestamp", "symbol"]).join(
        features_df.reset_index().set_index(["timestamp", "symbol"])
    )
    
    if len(feature_resampled_df) == 0:
        logger.error("No data after joining features with resampled timestamps")
        return pd.DataFrame()
    
    # Load targets data
    logger.info("Loading target data")
    targets_df = load_cached_targets(
        params=target_params,
        time_range=time_range,
        columns=columns,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
    )
    
    if targets_df is None or len(targets_df) == 0:
        logger.warning("No target data available, returning only features")
        return feature_resampled_df.reset_index().set_index("timestamp")

    # Join targets with resampled data
    logger.info("Joining targets with features and resampled data")
    ml_data_df = feature_resampled_df.join(
        targets_df.reset_index().set_index(["timestamp", "symbol"])
    ).reset_index().set_index("timestamp").drop(["symbol"], axis=1)

    
    if len(ml_data_df) == 0:
        logger.error("No data after joining features, targets, and resampled timestamps")
        return pd.DataFrame()
    
    logger.info(f"Successfully constructed ML data with {len(ml_data_df)} rows")
    logger.info(f"ML data columns: {ml_data_df.columns.tolist()}")
    
    return ml_data_df
