import pandas as pd
import logging
from typing import Any, Optional

from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
from market_data.machine_learning.cache_resample import load_cached_resampled_data
from market_data.machine_learning.resample import ResampleParams
from market_data.feature.cache_reader import read_multi_feature_cache
from market_data.feature.util import parse_feature_label_param
from market_data.feature.impl.common import SequentialFeatureParam

logger = logging.getLogger(__name__)

def prepare_feature_resampled(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_label: str,
    feature_params: Any,
    resample_params: ResampleParams = None,
) -> pd.DataFrame:
    """
    Prepare feature data by joining a single feature with resampled data.
    
    This function:
    * Loads cached resampled data (returns empty DataFrame if not available)
    * Loads the specified feature data
    * Joins the feature data with resampled data to create a dataset aligned with significant price movements
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        feature_label: Name of the feature to load (e.g., 'bollinger_bands', 'rsi')
        feature_params: Parameters for the feature calculation. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        
    Returns:
        DataFrame with the feature joined with resampled data, indexed by [timestamp, symbol].
        Returns empty DataFrame if resampled data or feature data is not available.
    """
    # Parse feature label and parameters (handles None parameters by creating defaults)
    feature_label_param = parse_feature_label_param((feature_label, feature_params,))
    resample_params = resample_params or ResampleParams()
    
    # Load resampled data
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
    resampled_df_cleaned = resampled_df_cleaned.reset_index().set_index(["timestamp", "symbol"])
    
    # Load the feature data
    logger.info(f"Processing feature: {feature_label_param}")
    
    feature_df = read_multi_feature_cache(
        feature_labels_params=[feature_label_param],
        time_range=time_range,
        dataset_mode=dataset_mode,
        export_mode=export_mode,
        aggregation_mode=aggregation_mode
    )
    
    if feature_df is None or len(feature_df) == 0:
        logger.warning(f"No data available for feature '{feature_label}', skipping")
        return pd.DataFrame()
    
    # Join this feature with the resampled DataFrame
    feature_df_indexed = feature_df.reset_index().set_index(["timestamp", "symbol"])
    feature_resampled_df = resampled_df_cleaned.join(feature_df_indexed)
    
    if len(feature_resampled_df) == 0:
        logger.error("No feature data available after resampling")
        return pd.DataFrame()
    
    logger.info(f"Successfully resampled {feature_label} with {len(feature_resampled_df)} rows")
    return feature_resampled_df


def prepare_sequential_feature_resampled(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    feature_label: str,
    feature_params: Any,
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
        target_params_batch: Target calculation parameters. If None, uses default parameters.
        resample_params: Resampling parameters. If None, uses default parameters.
        seq_params: Sequential feature parameters. If None, uses default parameters.
        
    Returns:
        DataFrame with sequential features and targets
    """
    # Use default parameters if none provided
    feature_label_param = parse_feature_label_param((feature_label, feature_params,))
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
    
    sequence_df = pd.concat(sequence_dfs).sort_index(level='timestamp')
    
    # Check if we have any feature DataFrames
    if len(sequence_df) == 0:
        logger.error("No feature sequences could be created")
        return pd.DataFrame()
    
    logger.info(f"Successfully prepared sequential ML data with {len(sequence_df)} rows and {len(sequence_df.columns)} columns")
    return sequence_df

