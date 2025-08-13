"""
Feature Cache Writer

This module provides functions for writing feature data to cache,
allowing for caching specific features with their parameters.
"""

import logging
import math
from datetime import timedelta
from typing import Any, Optional

import pandas as pd

import market_data.util.cache.read
import market_data.util.cache.write
from market_data.feature.impl.common import SequentialFeatureParam
from market_data.feature.label import FeatureLabel
from market_data.feature.registry import get_feature_by_label
from market_data.feature.sequential_feature import sequentialize_feature
from market_data.ingest.common import CacheContext
from market_data.util.cache.time import split_t_range
from market_data.util.time import TimeRange


logger = logging.getLogger(__name__)

def cache_feature_cache(
        feature_label_obj: FeatureLabel,
        cache_context: CacheContext,
        time_range: TimeRange = None,
        calculation_batch_days: int = 1,
        warm_up_days: Optional[int] = None,
        overwrite_cache: bool = True
    ) -> bool:
    """
    Cache a specific feature with given parameters.
    
    This function allows caching a single feature type with specific parameters,
    which is useful for on-demand caching of features without recalculating all features.
    
    Args:
        feature_label_obj: FeatureLabel object containing feature name and parameters
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode - required
        time_range: TimeRange object specifying the time range to cache
        calculation_batch_days: Number of days to calculate features for in each batch
        warm_up_days: Number of warm-up days for calculation. If None, will attempt to use 
                     params.get_warm_up_days() if available, otherwise defaults to 1 day
        overwrite_cache: Whether to overwrite existing cache files
        
    Returns:
        True if caching was successful, False otherwise
    """
    feature_label, params = feature_label_obj.feature_label, feature_label_obj.params

    # Validate inputs
    if time_range is None:
        raise ValueError("TimeRange must be provided")
    
    # Get params directory
    params_dir = params.get_params_dir()
    
    # Determine warm-up days
    if warm_up_days is None:
        if hasattr(params, 'get_warm_up_days'):
            warm_up_days = params.get_warm_up_days()
            logger.info(f"Using warm-up days {warm_up_days} from {feature_label} params")
        else:
            warm_up_days = 1
            logger.warning(f"Params for {feature_label} does not have get_warm_up_days method, using {warm_up_days} day(s)")
    
    # Get feature module
    feature_cls = get_feature_by_label(feature_label)
    if feature_cls is None:
        raise ValueError(f"Feature class '{feature_label}' not found in registry")
    
    # Create calculation function
    def calculate_batch_fn(raw_df: pd.DataFrame, feature_params: Any) -> pd.DataFrame:
        calculate_fn = getattr(feature_cls, 'calculate', None)
        if calculate_fn is None:
            raise ValueError(f"Feature module {feature_label} does not have a calculate method")
        return calculate_fn(raw_df, feature_params)
    
    try:
        raw_data_folder_path = cache_context.get_market_data_path()
        folder_path = cache_context.get_feature_path(feature_label, params_dir)

        market_data.util.cache.write.calculate_and_cache_data(
            raw_data_folder_path=raw_data_folder_path,
            folder_path=folder_path,
            params=params,
            time_range=time_range,
            calculation_batch_days=calculation_batch_days,
            warm_up_days=warm_up_days,
            overwrite_cache=overwrite_cache,
            calculate_batch_fn=calculate_batch_fn,
        )
        logger.info(f"Successfully cached {feature_label} for {time_range}")
        return True
    except Exception as e:
        logger.error(f"[cache_writer] Error calculating/caching {feature_label}: {e}")
        return False

def cache_seq_feature_cache(
    feature_label_obj: FeatureLabel,
    cache_context: CacheContext,
    time_range: TimeRange,
    seq_params: SequentialFeatureParam = SequentialFeatureParam(),
    warm_up_days: Optional[int] = None,
    overwrite_cache: bool = True
) -> bool:
    """
    Cache sequential features for a specific feature with given parameters.
    
    Args:
        feature_label_obj: FeatureLabel object containing feature name and parameters
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode - required
        time_range: Time range for which to cache the feature
        seq_params: Sequential feature parameters
        calculation_batch_days: Number of days to process in each batch
        warm_up_days: Number of warm-up days for feature calculation
        overwrite_cache: Whether to overwrite existing cache files
        
    Returns:
        bool: True if caching was successful, False otherwise
    """
    try:
        feature_label, params = feature_label_obj.feature_label, feature_label_obj.params

        # Get feature class
        feature_cls = get_feature_by_label(feature_label)
        if not feature_cls:
            logger.error(f"Feature class not found for label: {feature_label}")
            return False
            
        params_dir = params.get_params_dir()
        
        # Calculate warm-up days
        if warm_up_days is None:
            if hasattr(params, 'get_warm_up_days'):
                warm_up_days = params.get_warm_up_days()
            else:
                warm_up_days = 0
                
        # Add sequence window warm-up days (assuming minute granularity)
        seq_warm_up_days = math.ceil(seq_params.sequence_window / (24 * 60))
        warm_up_days = max(warm_up_days, seq_warm_up_days)
        
        # Split time range into daily chunks
        t_from, t_to = time_range.to_datetime()
        t_ranges = split_t_range(t_from, t_to)
        
        for i, t_range in enumerate(t_ranges):
            # Calculate extended range for warm-up
            extended_range = TimeRange(
                t_from=t_range[0] - timedelta(days=warm_up_days),
                t_to=t_range[1]
            )
            
            # Try to read non-sequential feature cache
            try:
                params_dir=params.get_params_dir()
                folder_path = cache_context.get_feature_path(feature_label, params_dir)
                df = market_data.util.cache.read.read_from_local_cache(
                        folder_path,
                        time_range=extended_range,
                )
                
                if df is None or df.empty:
                    logger.warning(f"No feature cache found for {feature_label} on {t_range[0].date()}")
                    continue
                    
                # Create sequential features
                seq_df = sequentialize_feature(df, seq_params)
                if seq_df is None:
                    logger.error(f"Failed to sequentialize {feature_label} for {t_range[0].date()}")
                    continue
                
                # Cache sequential features
                seq_params_dir = f"sequence_window-{seq_params.sequence_window}/{params_dir}"
                seq_folder_path = cache_context.get_feature_path(feature_label, seq_params_dir)
                market_data.util.cache.write.cache_locally_df(
                    df=seq_df,
                    folder_path=seq_folder_path,
                    overwrite=overwrite_cache,
                    warm_up_period_days=warm_up_days,
                )

                logger.info(f"Successfully cached sequential {feature_label} for {t_range[0]}")
                
            except Exception as e:
                logger.error(f"Error processing {feature_label} for {t_range[0]}: {e}")
                continue
                
        return True
        
    except Exception as e:
        logger.error(f"Error caching sequential feature: {e}")
        return False
