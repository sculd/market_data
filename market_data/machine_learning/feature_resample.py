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
    feature_label_params = parse_feature_label_param((feature_label, feature_params,))
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
    logger.info(f"Processing feature: {feature_label_params}")
    
    feature_df = read_multi_feature_cache(
        feature_labels_params=[feature_label_params],
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
