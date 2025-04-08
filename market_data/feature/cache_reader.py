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
from market_data.feature.cache_writer import _create_default_params

logger = logging.getLogger(__name__)

def read_multi_feature_cache(
        feature_labels_params: List[Union[str, Tuple[str, Any]]],
        time_range: TimeRange = None,
        columns: typing.List[str] = None,
        dataset_mode: DATASET_MODE = None,
        export_mode: EXPORT_MODE = None,
        aggregation_mode: AGGREGATION_MODE = None
    ) -> pd.DataFrame:
    """
    Read cached features for multiple feature types and parameters.
    
    Args:
        feature_labels_params: List of either:
            - feature labels (str) - will use default parameters
            - (label, params) tuples - will use the provided parameters
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
                "volatility",  # Will use default VolatilityParams
                ("market_regime", MarketRegimeParams(volatility_windows=[240, 1440]))
            ],
            time_range=TimeRange(t_from="2023-01-01", t_to="2023-01-31"),
            dataset_mode=DATASET_MODE.OKX,
            export_mode=EXPORT_MODE.BY_MINUTE,
            aggregation_mode=AGGREGATION_MODE.TAKE_LASTEST
        )
        ```
    """
    all_dfs = []
    
    for feature_item in feature_labels_params:
        # Handle both string labels and (label, params) tuples
        if isinstance(feature_item, tuple) and len(feature_item) == 2:
            label, params = feature_item
        elif isinstance(feature_item, str):
            label = feature_item
            params = None
        else:
            logger.warning(f"Invalid feature item format: {feature_item}, must be string or (label, params) tuple")
            continue
        
        logger.info(f"Reading cached feature: {label}")
        
        # Get the feature module
        feature_module = get_feature_by_label(label)
        if feature_module is None:
            logger.warning(f"Feature module '{label}' not found, skipping cache read.")
            continue
        
        # Use default params if none provided
        if params is None:
            # Use the _create_default_params function from cache_writer
            params = _create_default_params(feature_module, label)
            if params is None:
                logger.warning(f"Failed to create default parameters for feature '{label}', skipping")
                continue
            logger.info(f"Using default parameters for feature '{label}': {params}")
        
        # Get params_dir directly from the params object
        if hasattr(params, 'get_params_dir'):
            params_dir = params.get_params_dir()
        else:
            logger.warning(f"Params object for '{label}' does not have get_params_dir method.")
            continue
        
        # Include feature label in cache path
        cache_path = f"{FEATURE_CACHE_BASE_PATH}/features"
        
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
    
    # Combine all feature DataFrames
    try:
        if not all_dfs:
            logger.warning("No data found in cache for any of the requested features")
            return pd.DataFrame()
            
        # Proper join on timestamp and symbol instead of simple concatenation
        logger.info("Joining feature DataFrames on timestamp and symbol")
        
        # Create multi-index DataFrames for proper joining
        joined_df = None
        for i, df in enumerate(all_dfs):
            # Ensure DataFrame has timestamp and symbol columns
            if not all(col in df.index.names for col in ['timestamp','symbol']):
                logger.warning(f"DataFrame {i} is missing required columns (timestamp, symbol), skipping")
                continue
                
            # Create multi-index DataFrame for joining
            df_indexed = df
            
            if joined_df is None:
                joined_df = df_indexed
            else:
                # Outer join to preserve all timestamps and symbols
                joined_df = joined_df.join(df_indexed, how='outer')
        
        if joined_df is None:
            logger.warning("No valid DataFrames found with timestamp and symbol columns")
            return pd.DataFrame()
        
        logger.info(f"Combined {len(all_dfs)} feature sets into DataFrame with {len(joined_df)} rows")
        return joined_df
    except Exception as e:
        logger.error(f"Error combining feature DataFrames: {e}")
        return pd.DataFrame() 