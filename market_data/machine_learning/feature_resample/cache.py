import datetime
import logging
from typing import List, Optional

import pandas as pd

import market_data.util.cache.read
import market_data.util.cache.write
from market_data.feature.label import FeatureLabel
from market_data.feature.param import SequentialFeatureParam
from market_data.ingest.common import CacheContext
from market_data.machine_learning.feature_resample.calc import (
    prepare_feature_resampled,
    prepare_sequential_feature_resampled,
)
from market_data.machine_learning.resample.calc import CumSumResampleParams
from market_data.machine_learning.resample.param import ResampleParam
from market_data.util.cache.parallel_processing import read_multithreaded
from market_data.util.cache.time import anchor_to_begin_of_day
from market_data.util.time import TimeRange

logger = logging.getLogger(__name__)


def _calculate_and_cache_daily_feature_resampled(
    date: datetime.datetime,
    cache_context: CacheContext,
    feature_label_obj: FeatureLabel,
    resample_params: ResampleParam,
    seq_param: Optional[SequentialFeatureParam] = None,
    overwrite_cache: bool = True
) -> None:
    """
    Calculate and cache feature resampled data for a single day.
    
    Can handle both regular and sequential features based on seq_param.
    
    Args:
        date: The date to calculate data for
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        feature_label_obj: FeatureLabel object containing the feature label and parameters
        resample_params: Resampling parameters
        seq_param: Sequential feature parameters. If provided, creates sequential features.
        overwrite_cache: Whether to overwrite existing cache files
    """
    # Create time range for the specific day
    t_from = date
    t_to = date + datetime.timedelta(days=1)
    time_range = TimeRange(t_from, t_to)

    # Prepare feature data for the day (sequential or regular)
    if seq_param is not None:
        feature_resampled_df = prepare_sequential_feature_resampled(
            cache_context=cache_context,
            time_range=time_range,
            feature_label_obj=feature_label_obj,
            resample_params=resample_params,
            seq_param=seq_param,
        )
    else:
        feature_resampled_df = prepare_feature_resampled(
            cache_context=cache_context,
            time_range=time_range,
            feature_label_obj=feature_label_obj,
            resample_params=resample_params
        )
        
    if feature_resampled_df is None or len(feature_resampled_df) == 0:
        logger.warning(f"No feature resampled data available for {date}")
        return
    
    # Cache the data
    data_type = "sequential" if seq_param is not None else "regular"
    logger.info(f"Caching {data_type} feature resampled data for {date}")
    
    folder_path = cache_context.get_feature_resampled_path(
        feature_label_obj.feature_label, 
        feature_label_obj.params.get_params_dir(seq_param=seq_param),  
        resample_params.get_params_dir())
    
    market_data.util.cache.write.cache_locally_df(
        df=feature_resampled_df,
        folder_path=folder_path,
        overwrite=overwrite_cache,
        warm_up_period_days=0,
    )
    
    logger.info(f"Successfully cached {data_type} feature resampled data for {date} with {len(feature_resampled_df)} rows")


def calculate_and_cache_feature_resampled(
    cache_context: CacheContext,
    time_range: TimeRange,
    feature_label_obj: FeatureLabel,
    resample_params: ResampleParam = None,
    seq_param: Optional[SequentialFeatureParam] = None,
    overwrite_cache: bool = True
) -> bool:
    """
    Calculate and cache feature resampled data by processing the data daily.
    
    This function:
    1. Splits the time range into individual days
    2. For each day, prepares feature resampled data (sequential or regular based on seq_param)
    3. Caches each daily piece separately for sequential vs regular features
    
    Args:
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        time_range: TimeRange object specifying the time range
        feature_label_obj: FeatureLabel object containing the feature label and parameters
        resample_params: Resampling parameters. If None, uses default parameters.
        seq_param: Sequential feature parameters. If provided, creates sequential features.
        overwrite_cache: Whether to overwrite existing cache files
        
    Returns:
        True if caching was successful, False otherwise
    """
    resample_params = resample_params or CumSumResampleParams()
    t_from, t_to = time_range.to_datetime()
    current_date = t_from
    
    data_type = "sequential" if seq_param is not None else "regular"
    logger.info(f"Starting {data_type} feature resampled data processing for {feature_label_obj.feature_label}")
    
    try:
        # Process each day
        while current_date < t_to:
            logger.info(f"Processing day {current_date}")
            
            _calculate_and_cache_daily_feature_resampled(
                date=current_date,
                cache_context=cache_context,
                feature_label_obj=feature_label_obj,
                resample_params=resample_params,
                seq_param=seq_param,
                overwrite_cache=overwrite_cache
            )

            current_date = anchor_to_begin_of_day(current_date + datetime.timedelta(days=1))

        logger.info(f"Successfully cached {feature_label_obj.feature_label} resampled for {time_range}")
        return True
    except Exception as e:
        logger.error(f"[cache_writer] Error calculating/caching {feature_label_obj.feature_label} resampled: {e}")
        return False


def load_cached_feature_resampled(
    cache_context: CacheContext,
    time_range: TimeRange,
    feature_label_obj: FeatureLabel,
    resample_params: ResampleParam = None,
    seq_param: Optional[SequentialFeatureParam] = None,
    columns: Optional[List[str]] = None,
        max_workers: int = 10,
) -> pd.DataFrame:
    """
    Load cached feature resampled data for a specific time range.
    
    Can load both regular and sequential feature data based on seq_param.
    
    Args:
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        time_range: TimeRange object specifying the time range
        feature_label_obj: FeatureLabel object containing the feature label and parameters
        resample_params: Resampling parameters. If None, uses default parameters.
        seq_param: Sequential feature parameters. If provided, loads sequential features.
        columns: Optional list of columns to load. If None, loads all columns.
        
    Returns:
        DataFrame with feature resampled data (regular or sequential based on seq_param),
        or empty DataFrame if no data is available
    """
    resample_params = resample_params or CumSumResampleParams()

    folder_path = cache_context.get_feature_resampled_path(
        feature_label_obj.feature_label, 
        feature_label_obj.params.get_params_dir(seq_param=seq_param), 
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

    feature_resampled_df = read_multithreaded(
        read_func=load,
        time_range=time_range,
        max_workers=max_workers
    )
    
    if feature_resampled_df.empty:
        logging.error(f"feature resampled data {feature_label_obj} @ {time_range} is empty")
        return feature_resampled_df
    return feature_resampled_df.sort_values(["timestamp", "symbol"])
