import logging

import pandas as pd

from market_data.ingest.common import CacheContext
from market_data.target.cache import load_cached_targets
from market_data.target.param import TargetParamsBatch
from market_data.machine_learning.resample.cache import load_cached_resampled_data
from market_data.machine_learning.resample.calc import CumSumResampleParams
from market_data.machine_learning.resample.param import ResampleParam
from market_data.util.time import TimeRange

logger = logging.getLogger(__name__)


def calculate(
    cache_context: CacheContext,
    time_range: TimeRange,
    target_params_batch: TargetParamsBatch,
    resample_params: ResampleParam = None,
) -> pd.DataFrame:
    resample_params = resample_params or CumSumResampleParams()
    
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
    
    logger.info(f"Loading targets")
    
    targets_df = load_cached_targets(
        cache_context=cache_context,
        params=target_params_batch,
        time_range=time_range
    )
    
    if targets_df is None or len(targets_df) == 0:
        logger.warning(f"No data available for targets_df, skipping")
        return pd.DataFrame()
    
    # Join this feature with the resampled DataFrame
    targets_df_indexed = targets_df.reset_index().set_index(["timestamp", "symbol"])
    targets_resampled_df = resampled_df_cleaned.join(targets_df_indexed)
    
    if len(targets_resampled_df) == 0:
        logger.error("No feature data available after resampling")
        return pd.DataFrame()
    
    logger.info(f"Successfully resampled target with {len(targets_resampled_df)} rows")
    return targets_resampled_df
