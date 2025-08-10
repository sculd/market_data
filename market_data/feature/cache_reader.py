"""
Feature Cache Reader

This module provides functions for reading cached feature data,
including a multi-feature cache reader that can load multiple
feature types in a single operation.
"""

import pandas as pd
import logging
import typing
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import market_data.ingest.common
from market_data.ingest.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, CacheContext
from market_data.util.time import TimeRange
from market_data.util.cache.path import get_cache_base_path
import market_data.util.cache.cache_common
import market_data.feature.impl # needed to register features
from market_data.feature.registry import get_feature_by_label
from market_data.feature.util import parse_feature_label_params
from market_data.util.cache.parallel_processing import (
    read_multithreaded,
)

# Global paths configuration - use configurable base path
FEATURE_CACHE_BASE_PATH = os.path.join(get_cache_base_path(), 'feature_data')
Path(FEATURE_CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

def read_multi_feature_cache(
        feature_labels_params: Optional[List[Union[str, Tuple[str, Any]]]] = None,
        time_range: TimeRange = None,
        columns: typing.List[str] = None,
        cache_context: CacheContext = None,
        max_workers: int = 10,
    ) -> pd.DataFrame:
    """
    Read cached features for multiple feature types and parameters.
    
    Args:
        feature_labels_params: List of either:
            - feature labels (str) - will use default parameters
            - (label, params) tuples - will use the provided parameters
        time_range: Time range to fetch data for
        columns: Specific columns to retrieve
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        
    Returns:
        Combined DataFrame with all requested features
    """
    all_dfs = []
    
    feature_labels_params = parse_feature_label_params(feature_labels_params)
    for feature_label_param in feature_labels_params:
        feature_label, params = feature_label_param

        logger.info(f"Reading cached feature: {feature_label}")
        
        # Get the feature module
        feature_module = get_feature_by_label(feature_label)
        if feature_module is None:
            logger.warning(f"Feature module '{feature_label}' not found, skipping cache read.")
            continue
        
        # Read from cache
        try:
            def load(d_from, d_to):
                params_dir=params.get_params_dir()
                folder_path = cache_context.get_feature_path(feature_label, params_dir)
                df = market_data.ingest.cache_read.read_daily_from_local_cache(
                        folder_path,
                        d_from = d_from,
                        d_to = d_to,
                        columns=columns,
                )        
                return d_from, df

            df = read_multithreaded(
                read_func=load,
                time_range=time_range,
                max_workers=max_workers
            )

            if df is not None and not df.empty:
                all_dfs.append(df)
                logger.info(f"Successfully read {len(df)} rows for feature '{feature_label}'")
            else:
                logger.warning(f"No data found in cache for feature '{feature_label}'")
        
        except Exception as e:
            logger.error(f"Error reading cache for feature '{feature_label}': {e}")
    
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