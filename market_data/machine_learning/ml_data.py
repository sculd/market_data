import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from pathlib import Path

from market_data.target.target import TargetParams
from market_data.feature.feature import FeatureParams
from market_data.ingest.bq.cache import read_from_cache_or_query_and_cache
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
from market_data.feature.cache_feature import load_cached_features, calculate_and_cache_features
from market_data.target.cache_target import load_cached_targets, calculate_and_cache_targets
from market_data.machine_learning.cache_resample import load_cached_resampled_data, calculate_and_cache_resampled
from market_data.machine_learning.resample import ResampleParams
from market_data.feature.registry import list_registered_features
from market_data.feature.cache_reader import read_multi_feature_cache
from market_data.feature.cache_writer import _create_default_params
from market_data.feature.registry import get_feature_by_label

logger = logging.getLogger(__name__)

def prepare_ml_data(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_labels_params: Optional[List[Union[str, Tuple[str, Any]]]] = None,
    feature_params: FeatureParams = None,
    target_params: TargetParams = None,
    resample_params: ResampleParams = None,
    overwrite_cache: bool = False
) -> pd.DataFrame:
    """
    Prepare machine learning data by ensuring all required data is present and properly joined.
    
    This function:
    * Ensures raw data is present (caches if not)
    * Ensures resampled data is present (calculates and caches if not)
    * Loads features one at a time and joins with resampled data to reduce data size early
    * Ensures target data is present (calculates and caches if not)
    * Joins all data to create the final ML dataset
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_labels_params: List of either:
            - feature labels (str) - will use default parameters
            - (label, params) tuples - will use the provided parameters
            If None, uses all available features with default parameters
        feature_params: Feature calculation parameters (used only when feature_labels_params is None)
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
    
    # If feature_labels_params is None, use all available feature labels with default parameters
    if feature_labels_params is None:
        logger.info("No specific features specified, using all available features with default parameters")
        feature_labels = list_registered_features()
        feature_labels_params = feature_labels
        logger.info(f"Found {len(feature_labels)} registered features: {feature_labels}")
    
    t_from, t_to = time_range.to_datetime()
    # Ensure raw data is present
    raw_df = read_from_cache_or_query_and_cache(
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
        t_from=t_from, t_to=t_to,
        overwirte_cache=overwrite_cache
    )
    
    if raw_df is None or len(raw_df) == 0:
        logger.error("No raw data available")
        return pd.DataFrame()

    # Ensure resampled data is present
    resampled_df = load_cached_resampled_data(
        params=resample_params,
        time_range=time_range,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode
    )
    
    if resampled_df is None:
        logger.info("Calculating and caching resampled data")
        calculate_and_cache_resampled(
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
    
    # Remove OHLCV columns from resampled data as they'll be added with features
    resampled_df_cleaned = resampled_df.drop(["open", "high", "low", "close", "volume"], axis=1, errors='ignore')
    
    # Initialize the combined feature and resampled DataFrame
    combined_df = resampled_df_cleaned.reset_index().set_index(["timestamp", "symbol"])
    
    # Load and join features with resampled data one by one
    for feature_item in feature_labels_params:
        # Handle both string labels and (label, params) tuples
        if isinstance(feature_item, tuple) and len(feature_item) == 2:
            label, params = feature_item
        elif isinstance(feature_item, str):
            label = feature_item
            params = None
        else:
            logger.warning(f"Invalid feature item format: {feature_item}, must be string or (label, params) tuple")
            continue
        
        logger.info(f"Processing feature: {label}")
        
        # Get the feature module
        feature_module = get_feature_by_label(label)
        if feature_module is None:
            logger.warning(f"Feature module '{label}' not found, skipping.")
            continue
        
        # Create default params if needed
        if params is None:
            params = _create_default_params(feature_module, label)
            if params is None:
                logger.warning(f"Failed to create default parameters for feature '{label}', skipping")
                continue
            logger.info(f"Using default parameters for feature '{label}': {params}")
        
        # Load the feature data
        feature_df = read_multi_feature_cache(
            feature_labels_params=[(label, params)],
            time_range=time_range,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode
        )
        
        if feature_df is None or len(feature_df) == 0:
            logger.warning(f"No data available for feature '{label}', skipping")
            continue
        
        # Join this feature with the combined DataFrame
        feature_df_indexed = feature_df.reset_index().set_index(["timestamp", "symbol"])
        combined_df = combined_df.join(feature_df_indexed)
        
        logger.info(f"Added feature '{label}' to combined data, now has {len(combined_df.columns)} columns")
    
    if len(combined_df.columns) == 0:
        logger.error("No feature data available after joining all features")
        return pd.DataFrame()
    
    # Ensure target data is present
    targets_df = load_cached_targets(
        params=target_params,
        time_range=time_range,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode,
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

    # Join targets with the combined feature data
    targets_df_indexed = targets_df.reset_index().set_index(["timestamp", "symbol"])
    ml_data_df = combined_df.join(targets_df_indexed).reset_index().set_index("timestamp")
    
    if len(ml_data_df) == 0:
        logger.error("No data after joining features, targets and resampled timestamps")
        return pd.DataFrame()
    
    logger.info(f"Successfully prepared ML data with {len(ml_data_df)} rows and {len(ml_data_df.columns)} columns")
    return ml_data_df
