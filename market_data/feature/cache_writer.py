"""
Feature Cache Writer

This module provides functions for writing feature data to cache,
allowing for caching specific features with their parameters.
"""

import pandas as pd
import datetime
import logging
import os
from pathlib import Path
from typing import List, Optional, Union, Any, Tuple, Dict, Callable, Type
import numpy as np
import math
from datetime import timedelta

from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from market_data.ingest.bq.cache import read_from_cache_or_query_and_cache
from market_data.util.time import TimeRange
from market_data.util.cache.time import (
    split_t_range,
)
from market_data.util.cache.dataframe import cache_data_by_day, read_from_cache_generic
from market_data.util.cache.path import get_cache_base_path
from market_data.feature.registry import get_feature_by_label
from market_data.feature.util import parse_feature_label_param
from market_data.feature.impl.common import SequentialFeatureParam
from market_data.feature.sequential_feature import sequentialize_feature
from market_data.util.cache.core import calculate_and_cache_data
from market_data.util.cache.missing_data_finder import check_missing_feature_data

# Global paths configuration - use configurable base path
FEATURE_CACHE_BASE_PATH = os.path.join(get_cache_base_path(), 'feature_data')
Path(FEATURE_CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)


logger = logging.getLogger(__name__)

def cache_feature_cache(
        feature_label_param: Union[str, Tuple[str, Any]],
        dataset_mode: DATASET_MODE,
        export_mode: EXPORT_MODE,
        aggregation_mode: AGGREGATION_MODE,
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
        feature_label_param: Either a feature label string or a tuple of (feature_label, parameters).
                             If only a feature label is provided, a default parameter instance will be created.
                             If a tuple is provided, feature_label is a registered feature label and
                             parameters is an instance of the appropriate parameters class or None.
                             If parameters is None, a default parameter instance will be created.
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.) - required
        export_mode: Export mode (OHLC, TICKS, etc.) - required
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.) - required
        time_range: TimeRange object specifying the time range to cache
        calculation_batch_days: Number of days to calculate features for in each batch
        warm_up_days: Number of warm-up days for calculation (if None, auto-calculated)
        overwrite_cache: Whether to overwrite existing cache files
        
    Returns:
        True if caching was successful, False otherwise
    """
    feature_label, params = parse_feature_label_param(feature_label_param)

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
    
    # Define cache path
    cache_path = f"{FEATURE_CACHE_BASE_PATH}/features"
    
    # Get feature module
    feature_module = get_feature_by_label(feature_label)
    if feature_module is None:
        raise ValueError(f"Feature module '{feature_label}' not found in registry")
    
    # Create calculation function
    def calculate_batch_fn(raw_df: pd.DataFrame, feature_params: Any) -> pd.DataFrame:
        calculate_fn = getattr(feature_module, 'calculate', None)
        if calculate_fn is None:
            raise ValueError(f"Feature module {feature_label} does not have a calculate method")
        return calculate_fn(raw_df, feature_params)
    
    try:
        # Use core calculation and caching function
        calculate_and_cache_data(
            dataset_mode=dataset_mode,
            export_mode=export_mode,
            aggregation_mode=aggregation_mode,
            params=params,
            time_range=time_range,
            calculation_batch_days=calculation_batch_days,
            warm_up_days=warm_up_days,
            overwrite_cache=overwrite_cache,
            label=feature_label,
            calculate_batch_fn=calculate_batch_fn,
            cache_base_path=cache_path,
            params_dir=params_dir,
        )
        logger.info(f"Successfully cached {feature_label} for {time_range}")
        return True
    except Exception as e:
        logger.error(f"[cache_writer] Error calculating/caching {feature_label}: {e}")
        return False

def cache_seq_feature_cache(
    feature_label_param: Dict[str, Any],
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    seq_params: SequentialFeatureParam = SequentialFeatureParam(),
    calculation_batch_days: int = 30,
    warm_up_days: Optional[int] = None,
    overwrite_cache: bool = True
) -> bool:
    """
    Cache sequential features for a specific feature with given parameters.
    
    Args:
        feature_label_param: Dictionary containing feature label and parameters
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.) - required
        export_mode: Export mode (OHLC, TICKS, etc.) - required
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.) - required
        time_range: Time range for which to cache the feature
        seq_params: Sequential feature parameters
        calculation_batch_days: Number of days to process in each batch
        warm_up_days: Number of warm-up days for feature calculation
        overwrite_cache: Whether to overwrite existing cache files
        
    Returns:
        bool: True if caching was successful, False otherwise
    """
    try:
        feature_label, params = parse_feature_label_param(feature_label_param)

        # Get feature module
        feature_module = get_feature_by_label(feature_label)
        if not feature_module:
            logger.error(f"Feature module not found for label: {feature_label}")
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
                # Include feature label in cache path
                cache_path = f"{FEATURE_CACHE_BASE_PATH}/features"
                
                df = read_from_cache_generic(
                    dataset_mode=dataset_mode,
                    export_mode=export_mode,
                    aggregation_mode=aggregation_mode,
                    time_range=extended_range,
                    label=feature_label,
                    cache_base_path=cache_path,
                    params_dir=params_dir
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
                cache_data_by_day(
                    df=seq_df,
                    dataset_mode=dataset_mode,
                    export_mode=export_mode,
                    aggregation_mode=aggregation_mode,
                    t_from=t_range[0],
                    t_to=t_range[1],
                    label=feature_label,
                    cache_base_path=cache_path,
                    params_dir=seq_params_dir,
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
