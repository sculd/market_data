import datetime
import logging
import typing
from typing import Callable

import pandas as pd

import market_data.util.cache.read
import market_data.util.cache.write
from market_data.ingest.common import CacheContext
from market_data.ingest.gcs.cache import read_from_local_cache_or_query_and_cache
from market_data.machine_learning.resample.calc import CumSumResampleParams
from market_data.machine_learning.resample.param import ResampleParam
from market_data.util.cache.parallel_processing import read_multithreaded
from market_data.util.cache.time import split_t_range
from market_data.util.time import TimeRange

logger = logging.getLogger(__name__)


def calculate_and_cache_resampled(
    cache_context: CacheContext,
    resample_calculate_func: Callable = None,
    params: ResampleParam = None,
    time_range: TimeRange = None,
    calculation_batch_days: int = 1,
    overwrite_cache: bool = True,
) -> None:
    """
    Resample and cache data for a specified time range.
    
    1. Seeks raw data files for the range
    2. Caches raw data if not present
    3. Resamples data in batches
    4. Caches resampled results daily
    """
    assert resample_calculate_func is not None, "resample_calculate_func must be provided"
    
    params = params or CumSumResampleParams()
    
    # Resolve time range
    t_from, t_to = time_range.to_datetime() if time_range else (None, None)
    
    # Set up calculation parameters
    if calculation_batch_days <= 0:
        calculation_batch_days = 1
    calculation_interval = datetime.timedelta(days=calculation_batch_days)
    
    # Split the range into calculation batches
    calculation_ranges = split_t_range(t_from, t_to, interval=calculation_interval)
    
    for calc_range in calculation_ranges:
        calc_t_from, calc_t_to = calc_range
        logging.info(f"Processing resampling batch {calc_t_from} to {calc_t_to}")
        
        # 3. Resample data for this batch
        try:
            # 1 & 2. Get raw data (fetch and cache if not present)
            raw_df = read_from_local_cache_or_query_and_cache(
                cache_context,
                time_range=TimeRange(t_from=calc_t_from, t_to=calc_t_to),
                overwrite_cache=overwrite_cache
            )
            
            if raw_df is None or len(raw_df) == 0:
                logging.warning(f"No raw data available for {calc_t_from} to {calc_t_to}")
                continue
                
            resampled_df = resample_calculate_func(raw_df, params)
            
            if resampled_df is None or len(resampled_df) == 0:
                logging.warning(f"Resampling returned empty result for {calc_t_from} to {calc_t_to}")
                continue
                
            # 4. Cache resampled data daily
            folder_path = cache_context.get_resampled_path(params.get_params_dir())
            market_data.util.cache.write.cache_locally_df(
                df=resampled_df,
                folder_path=folder_path,
                overwrite=overwrite_cache,
                warm_up_period_days=0,
            )
            
        except Exception as e:
            logging.error(f"Error resampling data for {calc_t_from} to {calc_t_to}: {e}")
            continue

def load_cached_resampled_data(
    cache_context: CacheContext,
    params: ResampleParam = None,
    time_range: TimeRange = None,
    columns: typing.List[str] = None,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Load cached resampled data for a specific time range
    """
    
    def load(d_from, d_to):
        folder_path = cache_context.get_resampled_path(params.get_params_dir())
        df = market_data.util.cache.read.read_daily_from_local_cache(
                folder_path,
                d_from,
                d_to,
                columns=columns,
        )        
        return d_from, df

    return read_multithreaded(
        read_func=load,
        time_range=time_range,
        max_workers=max_workers
    )