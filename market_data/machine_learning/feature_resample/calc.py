import pandas as pd
import logging
from typing import Any, Optional

from market_data.ingest.common import CacheContext
from market_data.util.time import TimeRange
from market_data.machine_learning.resample.cache import load_cached_resampled_data
from market_data.machine_learning.resample.param import ResampleParam
from market_data.feature.cache_reader import read_multi_feature_cache
from market_data.feature.label import FeatureLabel, FeatureLabelCollection
from market_data.feature.impl.common import SequentialFeatureParam

logger = logging.getLogger(__name__)

def prepare_feature_resampled(
    cache_context: CacheContext,
    time_range: TimeRange,
    feature_label_obj: FeatureLabel,
    resample_params: ResampleParam = None,
) -> pd.DataFrame:
    """
    Prepare feature data by joining a single feature with resampled data.
    
    This function:
    * Loads cached resampled data (returns empty DataFrame if not available)
    * Loads the specified feature data
    * Joins the feature data with resampled data to create a dataset aligned with significant price movements
    
    Args:
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        time_range: TimeRange object specifying the time range
        feature_label_obj: FeatureLabel object containing the feature label and parameters
        resample_params: Resampling parameters. If None, uses default parameters.
        
    Returns:
        DataFrame with the feature joined with resampled data, indexed by [timestamp, symbol].
        Returns empty DataFrame if resampled data or feature data is not available.
    """
    resample_params = resample_params or ResampleParam()
    
    # Load resampled data
    resampled_df = load_cached_resampled_data(
        cache_context=cache_context,
        params=resample_params,
        time_range=time_range
    )
    
    if resampled_df is None or len(resampled_df) == 0:
        logger.error("No resampled data available")
        return pd.DataFrame()
    
    # Remove all columns from resampled data, keeping only the index (timestamp, symbol)
    # We only need the timestamps and symbols for joining - feature data comes from feature_df
    resampled_df_cleaned = resampled_df.reset_index().set_index(["timestamp", "symbol"])
    resampled_df_cleaned = resampled_df_cleaned[[]]
    
    # Load the feature data
    logger.info(f"Processing feature: {feature_label_obj.feature_label}")
    
    feature_label_collection = FeatureLabelCollection().with_feature_label(feature_label_obj)
    feature_df = read_multi_feature_cache(
        feature_label_collection=feature_label_collection,
        cache_context=cache_context,
        time_range=time_range
    )
    
    if feature_df is None or len(feature_df) == 0:
        logger.warning(f"No data available for feature '{feature_label_obj.feature_label}', skipping")
        return pd.DataFrame()
    
    # Join this feature with the resampled DataFrame
    feature_df_indexed = feature_df.reset_index().set_index(["timestamp", "symbol"])
    feature_resampled_df = resampled_df_cleaned.join(feature_df_indexed)
    
    if len(feature_resampled_df) == 0:
        logger.error("No feature data available after resampling")
        return pd.DataFrame()
    
    logger.info(f"Successfully resampled {feature_label_obj.feature_label} with {len(feature_resampled_df)} rows")
    return feature_resampled_df


def prepare_sequential_feature_resampled(
    cache_context: CacheContext,
    time_range: TimeRange,
    feature_label_obj: FeatureLabel,
    resample_params: ResampleParam = None,
    seq_params: Optional[SequentialFeatureParam] = None,
) -> pd.DataFrame:
    """
    Prepare sequential feature data by creating sequences aligned with resampled timestamps.
    
    This function:
    1. First loads resampled data to identify the timestamps and symbols
    2. For each resampled timestamp, creates a sequence of historical feature data
    3. Returns a DataFrame where each row contains a sequence of feature values
    
    This approach creates time-series sequences that can be used for sequential machine learning models.
    
    Args:
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        time_range: TimeRange object specifying the time range
        feature_label_obj: FeatureLabel object containing the feature label and parameters
        resample_params: Resampling parameters. If None, uses default parameters.
        seq_params: Sequential feature parameters. If None, uses default parameters.
        
    Returns:
        DataFrame with sequential feature data, indexed by [timestamp, symbol].
        Each row contains sequence arrays for the feature columns.
    """
    resample_params = resample_params or ResampleParam()
    seq_params = seq_params or SequentialFeatureParam()
    
    sequence_window = seq_params.sequence_window
    logger.info(f"Preparing sequential feature data with sequence window: {sequence_window}")
    
    # First, get the resampled data to identify required timestamps and symbols
    resampled_df = load_cached_resampled_data(
        cache_context=cache_context,
        params=resample_params,
        time_range=time_range
    )
    
    if resampled_df is None or len(resampled_df) == 0:
        logger.error("No resampled data available")
        return pd.DataFrame()
    
    # Remove all columns from resampled data, keeping only the index (timestamp, symbol)
    # We only need the timestamps and symbols for sequencing - feature data comes from feature_df
    resampled_df_cleaned = resampled_df.reset_index().set_index(["timestamp", "symbol"])
    resampled_df_cleaned = resampled_df_cleaned[[]]
    
    # Create a mapping of symbols to their resampled timestamps
    symbol_to_resampled_timestamps = {}
    for symbol, symbol_group in resampled_df.groupby('symbol'): 
        symbol_to_resampled_timestamps[symbol] = symbol_group.index.get_level_values('timestamp').unique()
 
    # Extend time range to include historical data for sequencing
    t_from, t_to = time_range.to_datetime()
    
    # For minute-level data, calculate lookback period
    # Add a buffer to ensure we have enough data
    extended_t_from = t_from - pd.Timedelta(minutes=sequence_window * 2)  # Add buffer
    extended_time_range = TimeRange(extended_t_from, t_to)
    
    logger.info(f"Loading features with extended time range: {extended_t_from} to {t_to}")
    
    logger.info(f"Processing feature: {feature_label_obj.feature_label}")
    
    # Load feature data with extended time range
    feature_label_collection = FeatureLabelCollection().with_feature_label(feature_label_obj)
    feature_df = read_multi_feature_cache(
        feature_label_collection=feature_label_collection,
        cache_context=cache_context,
        time_range=extended_time_range
    )
    
    if feature_df is None or len(feature_df) == 0:
        logger.warning(f"No data available for feature '{feature_label_obj.feature_label}', skipping")
        return pd.DataFrame()
    
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
                logger.debug(f"Not enough history for timestamp {ts}, symbol {symbol}, feature {feature_label_obj.feature_label}. Got {len(sequence_data)} points, need {sequence_window}.")
                continue
            
            # If we have more points than needed, take the most recent ones
            if len(sequence_data) > sequence_window:
                sequence_data = sequence_data.iloc[-sequence_window:]
            
            # Ensure we have the right number of values
            if len(sequence_data) != sequence_window:
                logger.debug(f"Expected {sequence_window} values but got {len(sequence_data)} for {ts}, {symbol}, {feature_label_obj.feature_label}")
                continue
            
            # Create a small DataFrame with this sequence
            # Convert each column to a sequence array
            sequence_dict = {}
            for col in sequence_data.columns:
                sequence_dict[f"{col}"] = [sequence_data[col].values]
            
            sequence_df = pd.DataFrame(
                sequence_dict,
                index=pd.MultiIndex.from_tuples([(ts, symbol)], names=['timestamp', 'symbol'])
            )
            
            # Add to our list of sequence DataFrames
            sequence_dfs.append(sequence_df)
    
    if len(sequence_dfs) == 0:
        logger.error("No feature sequences could be created")
        return pd.DataFrame()
    
    # Concatenate all sequences
    result_df = pd.concat(sequence_dfs).sort_index()
    
    logger.info(f"Successfully prepared sequential feature data with {len(result_df)} rows and {len(result_df.columns)} columns")
    return result_df

