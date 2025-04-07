"""
Feature Cache Reader

This module provides functions for reading cached feature data,
including a multi-feature cache reader that can load multiple
feature types in a single operation.
"""

import pandas as pd
import logging
import typing
from typing import List, Tuple, Dict, Any, Optional, Union

from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from market_data.util.time import TimeRange
from market_data.util.cache.dataframe import read_from_cache_generic
from market_data.feature.registry import get_feature_by_label
from market_data.feature.cache_feature import FEATURE_CACHE_BASE_PATH

logger = logging.getLogger(__name__)

def read_multi_feature_cache(
        feature_labels_params: List[Tuple[str, Any]],
        time_range: TimeRange = None,
        columns: typing.List[str] = None,
        dataset_mode: DATASET_MODE = None,
        export_mode: EXPORT_MODE = None,
        aggregation_mode: AGGREGATION_MODE = None
    ) -> pd.DataFrame:
    """
    Read cached features for multiple feature types and parameters.
    
    Args:
        feature_labels_params: List of (label, params) tuples
        time_range: Time range to fetch data for
        columns: Specific columns to retrieve
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        
    Returns:
        Combined DataFrame with all requested features
    
    Example:
        ```
        features = read_multi_feature_cache(
            feature_labels_params=[
                ("returns", ReturnParams(periods=[1, 5, 10])),
                ("volatility", VolatilityParams(window=20))
            ],
            time_range=TimeRange(t_from="2023-01-01", t_to="2023-01-31"),
            dataset_mode=DATASET_MODE.OKX,
            export_mode=EXPORT_MODE.BY_MINUTE,
            aggregation_mode=AGGREGATION_MODE.TAKE_LASTEST
        )
        ```
    """
    all_dfs = []
    
    for label, params in feature_labels_params:
        logger.info(f"Reading cached feature: {label}")
        
        # Get the feature module
        feature_module = get_feature_by_label(label)
        if feature_module is None:
            logger.warning(f"Feature module '{label}' not found, skipping cache read.")
            continue
        
        # Get params_dir directly from the params object
        if hasattr(params, 'get_params_dir'):
            params_dir = params.get_params_dir()
        else:
            logger.warning(f"Params object for '{label}' does not have get_params_dir method.")
            continue
        
        # Include feature label in cache path
        cache_path = f"{FEATURE_CACHE_BASE_PATH}/{label}"
        
        # Read from cache
        try:
            df = read_from_cache_generic(
                label=label,
                params_dir=params_dir,
                time_range=time_range,
                columns=columns,
                dataset_mode=dataset_mode,
                export_mode=export_mode,
                aggregation_mode=aggregation_mode,
                cache_base_path=cache_path
            )
            
            if df is not None and not df.empty:
                all_dfs.append(df)
                logger.info(f"Successfully read {len(df)} rows for feature '{label}'")
            else:
                logger.warning(f"No data found in cache for feature '{label}'")
        
        except Exception as e:
            logger.error(f"Error reading cache for feature '{label}': {e}")
    
    if not all_dfs:
        logger.warning("No data found in cache for any of the requested features")
        return pd.DataFrame()
    
    # Combine all feature DataFrames
    try:
        result = pd.concat(all_dfs, axis=1)
        logger.info(f"Combined {len(all_dfs)} feature sets into DataFrame with {len(result)} rows")
        return result
    except Exception as e:
        logger.error(f"Error combining feature DataFrames: {e}")
        return pd.DataFrame() 