import pandas as pd
import numpy as np
import os
import datetime
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
from market_data.feature.util import _create_default_params, parse_feature_label_param, parse_feature_label_params
from market_data.feature.impl.common import SequentialFeatureParam
from market_data.feature.sequential_feature import create_sequences_numba

logger = logging.getLogger(__name__)

def prepare_ml_data(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_label_params: Optional[List[Union[str, Tuple[str, Any]]]] = None,
    target_params: TargetParams = None,
    resample_params: ResampleParams = None,
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
        target_params: Target calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        
    Returns:
        DataFrame with features and targets, resampled at significant price movements
    """
    # Use default parameters if none provided
    feature_label_params = parse_feature_label_params(feature_label_params)
    target_params = target_params or TargetParams()
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
    for feature_label_param in feature_label_params:
        feature_label, params = parse_feature_label_param(feature_label_param)

        logger.info(f"Processing feature: {feature_label}")
        
        # Load the feature data
        feature_df = read_multi_feature_cache(
            feature_labels_params=[(feature_label, params)],
            time_range=time_range,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode
        )
        
        if feature_df is None or len(feature_df) == 0:
            logger.warning(f"No data available for feature '{feature_label}', skipping")
            continue
        
        # Join this feature with the combined DataFrame
        feature_df_indexed = feature_df.reset_index().set_index(["timestamp", "symbol"])
        combined_df = combined_df.join(feature_df_indexed)
        
        logger.info(f"Added feature '{feature_label}' to combined data, now has {len(combined_df.columns)} columns")
    
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


def prepare_sequential_ml_data(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_label_params: Optional[List[Union[str, Tuple[str, Any]]]] = None,
    target_params: TargetParams = None,
    resample_params: ResampleParams = None,
    seq_params: Optional[SequentialFeatureParam] = None,
) -> pd.DataFrame:
    """
    Prepare sequential machine learning data more efficiently.
    
    This function:
    1. First loads resampled data to identify the timestamps and symbols
    2. For each feature and resampled timestamp, creates a small DataFrame with sequence data
    3. Concatenates these DataFrames within each feature, then joins across features
    4. Joins with targets to create the final ML dataset
    
    This approach uses pandas' efficient join operations to combine data.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_label_params: List of feature labels and parameters
        target_params: Target calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        seq_params: Sequential feature parameters. If None, uses default parameters.
        
    Returns:
        DataFrame with sequential features and targets
    """
    # Use default parameters if none provided
    feature_label_params = parse_feature_label_params(feature_label_params)
    target_params = target_params or TargetParams()
    resample_params = resample_params or ResampleParams()
    seq_params = seq_params or SequentialFeatureParam()
    
    sequence_window = seq_params.sequence_window
    logger.info(f"Preparing sequential ML data with sequence window: {sequence_window}")
    
    # First, get the resampled data to identify required timestamps and symbols
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
    
    # Remove OHLCV columns from resampled data
    resampled_df = resampled_df.drop(["open", "high", "low", "close", "volume"], axis=1, errors='ignore')
    
    # Create a mapping of symbols to their resampled timestamps
    symbol_to_resampled_timestamps = {}
    for symbol, ts in resampled_df.groupby('symbol'): 
        symbol_to_resampled_timestamps[symbol] = ts.index

    # Extend time range to include historical data for sequencing
    t_from, t_to = time_range.to_datetime()
    
    # For minute-level data, calculate lookback period
    # Add a buffer to ensure we have enough data
    extended_t_from = t_from - pd.Timedelta(minutes=sequence_window)
    extended_time_range = TimeRange(extended_t_from, t_to)
    
    logger.info(f"Loading features with extended time range: {extended_t_from} to {t_to}")
    
    # Dictionary to store feature DataFrames
    feature_dfs = {}
    
    # Process each feature to create a DataFrame with sequence data
    for feature_label_param in feature_label_params:
        feature_label, params = parse_feature_label_param(feature_label_param)
        
        logger.info(f"Processing feature: {feature_label}")
        
        # Load feature data with extended time range
        feature_df = read_multi_feature_cache(
            feature_labels_params=[(feature_label, params)],
            time_range=extended_time_range,
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode
        )
        
        if feature_df is None or len(feature_df) == 0:
            logger.warning(f"No data available for feature '{feature_label}', skipping")
            continue
        
        # Sort by timestamp for each symbol
        feature_df = feature_df.sort_index()
        
        # List to store small DataFrames for each timestamp-symbol pair
        sequence_dfs = []
        
        # Process each symbol group for this feature
        for symbol, symbol_data in feature_df.groupby(level='symbol'):
            # Skip symbols not in resampled data
            if symbol not in symbol_to_resampled_timestamps:
                continue
                
            # Get the timestamps for this symbol from resampled data
            resampled_timestamps = symbol_to_resampled_timestamps[symbol]
            
            # Process each resampled timestamp for this symbol
            for ts in resampled_timestamps:
                # Calculate the start time for our window
                start_ts = ts - pd.Timedelta(minutes=sequence_window)
                
                # Get data points within our window
                sequence_data = symbol_data[
                    (symbol_data.index.get_level_values('timestamp') > start_ts) & 
                    (symbol_data.index.get_level_values('timestamp') <= ts)
                ]
                
                # Sort by timestamp to ensure correct order
                sequence_data = sequence_data.sort_index(level='timestamp')
                
                # Check if we have enough data points
                if len(sequence_data) < sequence_window:
                    logger.debug(f"Not enough history for timestamp {ts}, symbol {symbol}, feature {feature_label}. Got {len(sequence_data)} points, need {sequence_window}.")
                    continue
                
                # If we have more points than needed, take the most recent ones
                if len(sequence_data) > sequence_window:
                    sequence_data = sequence_data.iloc[-sequence_window:]
                
                # Extract the sequence of values for this feature
                sequence_values = sequence_data.values
                
                # Ensure we have the right number of values
                if len(sequence_values) != sequence_window:
                    logger.debug(f"Expected {sequence_window} values but got {len(sequence_values)} for {ts}, {symbol}, {feature_label}")
                    continue
                
                # Create a small DataFrame with this sequence
                # Use MultiIndex with timestamp and symbol
                
                sequence_df = pd.DataFrame({
                    c: [sequence_data[c].values] for c in sequence_data.columns
                }, index=pd.MultiIndex.from_tuples([(ts, symbol)], names=['timestamp', 'symbol']))
                
                # Add to our list of sequence DataFrames
                sequence_dfs.append(sequence_df)
        
        # Concatenate all sequence DataFrames for this feature
        if sequence_dfs:
            feature_dfs[feature_label] = pd.concat(sequence_dfs).sort_index(level='timestamp')
            logger.info(f"Created {len(feature_dfs[feature_label])} sequences for feature {feature_label}")
        else:
            logger.warning(f"No valid sequences created for feature {feature_label}")
    
    # Check if we have any feature DataFrames
    if not feature_dfs:
        logger.error("No feature sequences could be created")
        return pd.DataFrame()
    
    # Start with the first feature DataFrame
    feature_keys = list(feature_dfs.keys())
    combined_feature_df = feature_dfs[feature_keys[0]].copy()
    
    # Join with other feature DataFrames
    for feature_label in feature_keys[1:]:
        combined_feature_df = combined_feature_df.join(feature_dfs[feature_label], how='inner')
    
    logger.info(f"Created combined feature DataFrame with {len(combined_feature_df)} rows and {len(combined_feature_df.columns)} columns")
    
    # Load targets
    logger.info("Loading target data")
    targets_df = load_cached_targets(
        params=target_params,
        time_range=time_range,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode
    )
    targets_df = targets_df.reset_index().set_index(["timestamp", "symbol"])
    
    if targets_df is None or len(targets_df) == 0:
        logger.error("No target data available")
        return pd.DataFrame()
    
    # Join with targets
    ml_data_df = combined_feature_df.join(targets_df, how='inner').reset_index().set_index("timestamp")
    logger.info(f"Successfully prepared sequential ML data with {len(ml_data_df)} rows and {len(ml_data_df.columns)} columns")
    return ml_data_df
