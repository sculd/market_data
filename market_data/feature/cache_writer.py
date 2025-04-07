"""
Feature Cache Writer

This module provides functions for writing feature data to cache,
allowing for caching specific features with their parameters.
"""

import pandas as pd
import datetime
import logging
import inspect
import importlib
from typing import List, Optional, Union, Any, Tuple, Dict, Callable, Type

from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from market_data.ingest.bq.cache import read_from_cache_or_query_and_cache
from market_data.util.time import TimeRange
from market_data.util.cache.dataframe import cache_data_by_day
from market_data.feature.registry import get_feature_by_label
from market_data.feature.cache_feature import FEATURE_CACHE_BASE_PATH
from market_data.util.cache.core import calculate_and_cache_data

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
    
    Example:
        ```
        from market_data.feature.impl.returns import ReturnParams
        
        # Cache returns with specific parameters
        success = cache_feature_cache(
            feature_label_param=("returns", ReturnParams(periods=[1, 5, 10])),
            dataset_mode=DATASET_MODE.OKX,
            export_mode=EXPORT_MODE.BY_MINUTE,
            aggregation_mode=AGGREGATION_MODE.TAKE_LASTEST,
            time_range=TimeRange(t_from="2023-01-01", t_to="2023-01-31")
        )
        
        # Cache returns with default parameters
        success = cache_feature_cache(
            feature_label_param="returns",
            dataset_mode=DATASET_MODE.OKX,
            export_mode=EXPORT_MODE.BY_MINUTE,
            aggregation_mode=AGGREGATION_MODE.TAKE_LASTEST,
            time_range=TimeRange(t_from="2023-01-01", t_to="2023-01-31")
        )
        ```
    """
    # Handle case where only feature_label is provided
    if isinstance(feature_label_param, str):
        feature_label = feature_label_param
        params = None
    else:
        feature_label, params = feature_label_param
    
    # Validate inputs
    assert time_range is not None, "TimeRange must be provided"
    
    # Get feature module
    feature_module = get_feature_by_label(feature_label)
    if feature_module is None:
        logger.error(f"Feature module '{feature_label}' not found in registry")
        return False
    
    # If params is None, try to create a default instance of the appropriate parameter class
    if params is None:
        params = _create_default_params(feature_module, feature_label)
        if params is None:
            logger.error(f"Failed to create default parameters for feature '{feature_label}'")
            return False
        logger.info(f"Created default parameters for feature '{feature_label}': {params}")
    
    # Check if params has get_params_dir method
    if not hasattr(params, 'get_params_dir'):
        logger.error(f"Parameters object for feature '{feature_label}' must have get_params_dir method")
        return False
    
    # Get params directory
    params_dir = params.get_params_dir()
    
    # If warm_up_days not provided, use the get_warm_up_days method if available
    if warm_up_days is None:
        if hasattr(params, 'get_warm_up_days'):
            warm_up_days = params.get_warm_up_days()
            logger.info(f"Using warm-up days {warm_up_days} from {feature_label} params")
        else:
            # Default to 1 day if we can't determine warm-up period
            warm_up_days = 1
            logger.warning(f"Params for {feature_label} does not have get_warm_up_days method, using {warm_up_days} day(s)")
    
    # Cache base path including feature label
    cache_path = f"{FEATURE_CACHE_BASE_PATH}/features"
    
    # Create a calculation function that calls the feature module's calculate method
    def calculate_batch_fn(raw_df: pd.DataFrame, feature_params: Any) -> pd.DataFrame:
        calculate_fn = getattr(feature_module, 'calculate', None)
        if calculate_fn is None:
            raise ValueError(f"Feature module {feature_label} does not have a calculate method")
        return calculate_fn(raw_df, feature_params)
    
    try:
        # Use the core calculate_and_cache_data function
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
            params_dir=params_dir
        )
        return True
    except Exception as e:
        logger.error(f"Error calculating/caching {feature_label}: {e}")
        return False

def _create_default_params(feature_module, feature_label: str) -> Optional[Any]:
    """
    Create a default instance of the parameter class for a feature module.
    
    Args:
        feature_module: The feature module instance
        feature_label: The feature label
        
    Returns:
        An instance of the parameter class, or None if not found
    """
    try:
        # Try different naming patterns to find the params class
        # Common patterns include: ReturnParams, VolatilityParams, etc.
        
        # Pattern 1: Try to find a class attribute with 'Params' in the name
        for name, obj in inspect.getmembers(feature_module):
            if inspect.isclass(obj) and 'Params' in name:
                return obj()
        
        # Pattern 2: Try to import from the impl module directly
        try:
            # Assuming the module structure is like market_data.feature.impl.returns
            module_path = f"market_data.feature.impl.{feature_label}"
            module = importlib.import_module(module_path)
            
            # Look for a class ending with "Params"
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.endswith('Params'):
                    return obj()
        except (ImportError, AttributeError):
            pass
        
        # Pattern 3: Check if the module has a calculate method that accepts None as params
        if hasattr(feature_module, 'calculate'):
            # If calculate method handles None params by creating default ones,
            # we can use None directly
            return None
        
        return None
    except Exception as e:
        logger.warning(f"Error creating default params for {feature_label}: {e}")
        return None
