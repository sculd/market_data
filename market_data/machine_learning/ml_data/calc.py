import logging

import pandas as pd

from market_data.ingest.common import CacheContext
from market_data.machine_learning.feature_resample.cache import load_cached_feature_resampled
from market_data.machine_learning.resample.cache import load_cached_resampled_data
from market_data.machine_learning.target_resample.cache import load_cached_targets_resampled
from market_data.util.time import TimeRange
from market_data.machine_learning.ml_data.param import MlDataParam

logger = logging.getLogger(__name__)

def calculate(
    cache_context: CacheContext,
    time_range: TimeRange,
    ml_data_param: MlDataParam,
) -> pd.DataFrame:
    """
    Prepare machine learning data by loading cached feature_resampled data and joining with targets.
    
    This function uses pre-computed feature_resampled data for efficient ML dataset construction.
    The feature_resampled data is much smaller than raw feature data, making runtime construction fast.
    
    This function:
    * Loads cached resampled data to establish the base timestamps
    * Loads cached feature_resampled data (regular or sequential based on seq_param)
    * Joins multiple features efficiently
    * Loads cached target data
    * Joins all data to create the final ML dataset
    
    Args:
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        time_range: TimeRange object specifying the time range
        ml_data_param: MlDataParam object containing all parameters
        
    Returns:
        DataFrame with features and targets, resampled at significant price movements.
        Features can be regular or sequential based on seq_param.
    """
    # Extract parameters from MlDataParam
    feature_collection = ml_data_param.feature_collection
    target_params_batch = ml_data_param.target_params_batch
    resample_params = ml_data_param.resample_params
    seq_param = ml_data_param.seq_param
    
    # Ensure resampled data is present
    resampled_df = load_cached_resampled_data(
        cache_context=cache_context,
        params=resample_params,
        time_range=time_range
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
    
    data_type = "sequential" if seq_param is not None else "regular"
    logger.info(f"Loading {data_type} feature_resampled data for {len(feature_collection.feature_labels)} features")
    
    # Load and join features with resampled data one by one
    for feature_label_obj in feature_collection.feature_labels:

        feature_label = feature_label_obj.feature_label
        
        logger.info(f"Processing feature: {feature_label}")

        feature_resampled_df = load_cached_feature_resampled(
            cache_context=cache_context,
            time_range=time_range,
            feature_label_obj=feature_label_obj,
            resample_params=resample_params,
            seq_param=seq_param,
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
    targets_resampled_df = load_cached_targets_resampled(
        cache_context=cache_context,
        time_range=time_range,
        target_params_batch=target_params_batch,
        resample_params=resample_params,
    )
    
    if targets_resampled_df is None or len(targets_resampled_df) == 0:
        logger.error("No resampled target data available")
        return pd.DataFrame()

    # Join targets with the combined feature data
    ml_data_df = combined_df.join(targets_resampled_df).reset_index().set_index("timestamp")
    
    if len(ml_data_df) == 0:
        logger.error("No data after joining features, targets and resampled timestamps")
        return pd.DataFrame()
    
    logger.info(f"Successfully prepared {data_type} ML data with {len(ml_data_df)} rows and {len(ml_data_df.columns)} columns")
    return ml_data_df
