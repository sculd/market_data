import pandas as pd
import datetime
import logging
import typing
import os
from pathlib import Path
from dataclasses import asdict


from market_data.ingest.bq.cache import read_from_cache_or_query_and_cache
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.ingest.bq.common import get_full_table_id
from market_data.machine_learning.resample import resample_at_events, ResampleParams
from market_data.util.time import TimeRange

from market_data.util.cache.time import (
    split_t_range,
)
from market_data.util.cache.dataframe import (
    cache_data_by_day,
    read_from_cache_generic,
)

from market_data.util.cache.path import (
    params_to_dir_name
)

# The base directory for cache
RESAMPLE_CACHE_BASE_PATH = os.path.expanduser('~/algo_cache/feature_data')
Path(RESAMPLE_CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

def cache_resampled_data(df: pd.DataFrame, label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
                        params: ResampleParams = None, overwrite=True, dataset_id=None) -> None:
    """Cache a resampled DataFrame, splitting it into daily pieces"""
    params_dir = params_to_dir_name(asdict(params or ResampleParams()))
    return cache_data_by_day(df, label, t_from, t_to, params_dir, overwrite, dataset_id=dataset_id,
                            cache_base_path=RESAMPLE_CACHE_BASE_PATH, warm_up_period_days=0)

def read_resampled_data_from_cache(label: str, 
                                 params: ResampleParams = None,
                                 time_range: TimeRange = None,
                                 columns: typing.List[str] = None,
                                 dataset_id=None) -> pd.DataFrame:
    """Read cached resampled data for a specified time range"""
    params_dir = params_to_dir_name(asdict(params or ResampleParams()))
    t_from, t_to = time_range.to_datetime() if time_range else (None, None)
    return read_from_cache_generic(
        label,
        params_dir=params_dir,
        t_from=t_from, t_to=t_to,
        columns=columns,
        dataset_id=dataset_id,
        cache_base_path=RESAMPLE_CACHE_BASE_PATH
    )

def resample_and_cache_data(
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
        params: ResampleParams = None,
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
    
    Parameters:
    -----------
    params : ResampleParams, optional
        Resampling parameters. If None, uses default parameters.
    time_range : TimeRange, optional
        Time range for resampling. If None, must provide individual time parameters.
    """
    params = params or ResampleParams()
    
    # Resolve time range
    t_from, t_to = time_range.to_datetime() if time_range else (None, None)
    
    # Create label for resampled data cache
    resample_label = "resampled"
    
    # Get dataset ID for cache path
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{aggregation_mode}"
    
    # Set up calculation parameters
    if calculation_batch_days <= 0:
        calculation_batch_days = 1
    calculation_interval = datetime.timedelta(days=calculation_batch_days)
    
    # Split the range into calculation batches
    calculation_ranges = split_t_range(t_from, t_to, interval=calculation_interval)
    
    for calc_range in calculation_ranges:
        calc_t_from, calc_t_to = calc_range
        logging.info(f"Processing resampling batch {calc_t_from} to {calc_t_to}")
        
        # 1 & 2. Get raw data (fetch and cache if not present)
        raw_df = read_from_cache_or_query_and_cache(
            dataset_mode, export_mode, aggregation_mode,
            t_from=calc_t_from, t_to=calc_t_to,
            overwirte_cache=overwrite_cache
        )
        
        if raw_df is None or len(raw_df) == 0:
            logging.warning(f"No raw data available for {calc_t_from} to {calc_t_to}")
            continue
            
        # 3. Resample data for this batch
        try:
            resampled_df = resample_at_events(raw_df, params)
            
            if resampled_df is None or len(resampled_df) == 0:
                logging.warning(f"Resampling returned empty result for {calc_t_from} to {calc_t_to}")
                continue
                
            # 4. Cache resampled data daily
            cache_resampled_data(
                resampled_df, resample_label, 
                calc_t_from, calc_t_to,
                params=params,
                overwrite=overwrite_cache,
                dataset_id=dataset_id
            )
            
        except Exception as e:
            logging.error(f"Error resampling data for {calc_t_from} to {calc_t_to}: {e}")
            continue

def load_cached_resampled_data(
        params: ResampleParams = None,
        time_range: TimeRange = None,
        columns: typing.List[str] = None,
        dataset_mode: DATASET_MODE = None,
        export_mode: EXPORT_MODE = None,
        aggregation_mode: AGGREGATION_MODE = None
    ) -> pd.DataFrame:
    """Load cached resampled data for a specific time range"""
    resample_label = "resampled"
    
    # Get dataset ID for cache path if dataset_mode, export_mode, and aggregation_mode are provided
    dataset_id = None
    if dataset_mode is not None and export_mode is not None and aggregation_mode is not None:
        dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{aggregation_mode}"
    
    return read_resampled_data_from_cache(
        resample_label,
        params=params,
        time_range=time_range,
        columns=columns,
        dataset_id=dataset_id
    )
