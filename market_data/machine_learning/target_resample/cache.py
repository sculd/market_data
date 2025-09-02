import logging
from typing import List, Optional

import pandas as pd

import market_data.util.cache.read
import market_data.util.cache.write
from market_data.ingest.common import CacheContext
from market_data.machine_learning.resample.calc import CumSumResampleParams
from market_data.machine_learning.resample.param import ResampleParam
from market_data.machine_learning.target_resample.calc import calculate
from market_data.target.param import TargetParamsBatch
from market_data.util.cache.parallel_processing import read_multithreaded
from market_data.util.cache.time import split_t_range
from market_data.util.time import TimeRange

logger = logging.getLogger(__name__)


def _calculate_and_cache_daily_targets_resampled(
    time_range: TimeRange,
    cache_context: CacheContext,
    target_params_batch: TargetParamsBatch,
    resample_params: ResampleParam,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache targets resampled data for a single day.
    """
    folder_path = cache_context.get_target_resampled_path(
        target_params_batch.get_params_dir(), 
        resample_params.get_params_dir())
    
    targets_resampled_df = calculate(
        cache_context=cache_context,
        time_range=time_range,
        target_params_batch=target_params_batch,
        resample_params=resample_params
    )
    
    if targets_resampled_df is None or len(targets_resampled_df) == 0:
        logger.warning(f"No targets resampled data available for {time_range}")
        return False
    
    # Cache the data
    logger.info(f"Caching targets resampled data for {time_range}")
    
    market_data.util.cache.write.split_and_cache_daily_df(
        df=targets_resampled_df,
        folder_path=folder_path,
        overwrite=overwrite_cache,
        warm_up_period_days=0,
    )
    
    logger.info(f"Successfully cached targets resampled data for {time_range} with {len(targets_resampled_df)} rows")
    return True


def calculate_and_cache_targets_resampled(
    cache_context: CacheContext,
    time_range: TimeRange,
    target_params_batch: TargetParamsBatch,
    resample_params: ResampleParam = None,
    overwrite_cache: bool = True
) -> bool:
    resample_params = resample_params or CumSumResampleParams()
    
    # Split the range into calculation batches
    t_from, t_to = time_range.to_datetime()
    calculation_ranges = split_t_range(t_from, t_to)
    
    success = True
    for calc_range in calculation_ranges:
        calc_t_from, calc_t_to = calc_range
        calc_time_range = TimeRange(t_from=calc_t_from, t_to=calc_t_to)

        try:
            s = _calculate_and_cache_daily_targets_resampled(
                time_range=calc_time_range,
                cache_context=cache_context,
                target_params_batch=target_params_batch,
                resample_params=resample_params,
                overwrite_cache=overwrite_cache
            )
            success = success and s
        except Exception as e:
            logger.error(f"Error resampling data for {calc_t_from} to {calc_t_to}: {e}")
            success = False
            continue

    return success


def load_cached_targets_resampled(
    cache_context: CacheContext,
    time_range: TimeRange,
    target_params_batch: TargetParamsBatch,
    resample_params: ResampleParam = None,
    columns: Optional[List[str]] = None,
        max_workers: int = 10,
) -> pd.DataFrame:
    """
    Load cached targets resampled data for a specific time range.
    
    Returns:
        DataFrame with targets resampled data, or empty DataFrame if no data is available
    """
    resample_params = resample_params or CumSumResampleParams()

    folder_path = cache_context.get_target_resampled_path(
        target_params_batch.get_params_dir(), 
        resample_params.get_params_dir())
        
    # Create worker function that properly handles daily ranges
    def load(d_from, d_to):
        df = market_data.util.cache.read.read_daily_from_local_cache(
                folder_path,
                d_from,
                d_to,
                columns=columns,
        )
        
        return d_from, df

    targets_resampled_df = read_multithreaded(
        read_func=load,
        time_range=time_range,
        max_workers=max_workers
    )
    
    if targets_resampled_df.empty:
        logger.error(f"targets resampled data @ {time_range} is empty")
        return targets_resampled_df
    return targets_resampled_df.sort_values(["timestamp", "symbol"])
