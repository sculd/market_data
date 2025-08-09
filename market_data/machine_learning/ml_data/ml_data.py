import pandas as pd
import numpy as np
import os
import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from pathlib import Path

from market_data.target.target import TargetParamsBatch
from market_data.ingest.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
from market_data.target.cache_target import load_cached_targets
from market_data.machine_learning.resample.cache_resample import load_cached_resampled_data
from market_data.machine_learning.resample.resample import ResampleParams
from market_data.machine_learning.feature_resample.cache_feature_resample import load_cached_feature_resampled
from market_data.feature.util import parse_feature_label_param, parse_feature_label_params
from market_data.feature.impl.common import SequentialFeatureParam

logger = logging.getLogger(__name__)

def prepare_ml_data(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_label_params: Optional[List[Union[str, Tuple[str, Any]]]] = None,
    target_params_batch: TargetParamsBatch = None,
    resample_params: ResampleParams = None,
    seq_params: Optional[SequentialFeatureParam] = None,
) -> pd.DataFrame:
    """
    Prepare machine learning data by loading cached feature_resampled data and joining with targets.
    
    This function uses pre-computed feature_resampled data for efficient ML dataset construction.
    The feature_resampled data is much smaller than raw feature data, making runtime construction fast.
    
    This function:
    * Loads cached resampled data to establish the base timestamps
    * Loads cached feature_resampled data (regular or sequential based on seq_params)
    * Joins multiple features efficiently
    * Loads cached target data
    * Joins all data to create the final ML dataset
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_label_params: List of either:
            - feature labels (str) - will use default parameters
            - (label, params) tuples - will use the provided parameters
            If None, uses all available features with default parameters
        target_params_batch: Target calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        seq_params: Sequential feature parameters. If provided, loads sequential features.
        
    Returns:
        DataFrame with features and targets, resampled at significant price movements.
        Features can be regular or sequential based on seq_params.
    """
    # Use default parameters if none provided
    feature_label_params = parse_feature_label_params(feature_label_params)
    target_params_batch = target_params_batch or TargetParamsBatch()
    resample_params = resample_params or ResampleParams()
    
    # Ensure resampled data is present
    resampled_df = load_cached_resampled_data(
        params=resample_params,
        time_range=time_range,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode
    )
    
    if resampled_df is None:
        logger.info("No cached resampled data available")
        return pd.DataFrame()
    
    if resampled_df is None or len(resampled_df) == 0:
        logger.error("No resampled data available")
        return pd.DataFrame()
    
    # Remove OHLCV columns from resampled data as they'll be added with features
    resampled_df_cleaned = resampled_df.reset_index().set_index(["timestamp", "symbol"])
    resampled_df_cleaned = resampled_df_cleaned[["breakout_side"]]
    
    # Initialize the combined feature and resampled DataFrame
    combined_df = resampled_df_cleaned.copy()
    
    data_type = "sequential" if seq_params is not None else "regular"
    logger.info(f"Loading {data_type} feature_resampled data for {len(feature_label_params)} features")
    
    # Load and join features with resampled data one by one
    for feature_label_param in feature_label_params:
        feature_label, feature_params = parse_feature_label_param(feature_label_param)

        logger.info(f"Processing feature: {feature_label}")

        feature_resampled_df = load_cached_feature_resampled(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            time_range=time_range,
            feature_label=feature_label,
            feature_params=feature_params,
            resample_params=resample_params,
            seq_params=seq_params,
        )

        if feature_resampled_df is None or len(feature_resampled_df) == 0:
            logger.warning(f"No data available for feature '{feature_label}', skipping")
            continue
        
        combined_df = combined_df.join(feature_resampled_df)
        logger.info(f"Added feature '{feature_label}' to combined data, now has {len(combined_df.columns)} columns")
    
    if len(combined_df.columns) == 0:
        logger.error("No feature data available after joining all features")
        return pd.DataFrame()
    
    # Ensure target data is present
    targets_df = load_cached_targets(
        params=target_params_batch,
        time_range=time_range,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
    )
    
    if targets_df is None or len(targets_df) == 0:
        logger.error("No target data available")
        return pd.DataFrame()

    # Join targets with the combined feature data
    targets_df_indexed = targets_df.reset_index().set_index(["timestamp", "symbol"])
    ml_data_df = combined_df.join(targets_df_indexed).reset_index().set_index("timestamp")
    
    if len(ml_data_df) == 0:
        logger.error("No data after joining features, targets and resampled timestamps")
        return pd.DataFrame()
    
    logger.info(f"Successfully prepared {data_type} ML data with {len(ml_data_df)} rows and {len(ml_data_df.columns)} columns")
    return ml_data_df
