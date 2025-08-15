import logging
import typing

import pandas as pd

import market_data.feature.impl  # needed to register features
import market_data.util.cache.read
from market_data.feature.label import FeatureLabel, FeatureLabelCollection
from market_data.feature.param import SequentialFeatureParam
from market_data.feature.registry import get_feature_by_label
from market_data.ingest.common import CacheContext
from market_data.util.cache.parallel_processing import read_multithreaded
from market_data.util.time import TimeRange

logger = logging.getLogger(__name__)


def read_feature(
        feature_label_obj: FeatureLabel,
        time_range: TimeRange = None,
        columns: typing.List[str] = None,
        cache_context: CacheContext = None,
        max_workers: int = 10,
        seq_param: SequentialFeatureParam = None,
    ) -> pd.DataFrame:
    """
    Read cached features for multiple feature types and parameters.
    
    Args:
        feature_label_obj: A feature label and its parameters to read
        time_range: Time range to fetch data for
        columns: Specific columns to retrieve
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        max_workers: Number of workers for parallel reading (default: 10)
        seq_param: Sequential feature parameters
        
    Returns:
        Combined DataFrame with all requested features
    """
    feature_label = feature_label_obj.feature_label
    
    logger.info(f"Reading cached feature: {feature_label}")
    
    # Get the feature module
    feature_cls = get_feature_by_label(feature_label)
    if feature_cls is None:
        logger.warning(f"Feature class '{feature_label}' not found, skipping cache read.")
        return pd.DataFrame()
    
    # Read from cache
    try:
        def load(d_from, d_to):
            params_dir = feature_label_obj.params.get_params_dir(seq_param=seq_param)
            folder_path = cache_context.get_feature_path(feature_label, params_dir)
            df = market_data.util.cache.read.read_daily_from_local_cache(
                    folder_path,
                    d_from,
                    d_to,
                    columns=columns,
            )        
            return d_from, df

        df = read_multithreaded(
            read_func=load,
            time_range=time_range,
            max_workers=max_workers
        )

        if df is None or df.empty:
            logger.warning(f"No data found in cache for feature '{feature_label}'")
            return pd.DataFrame()
        return df
    
    except Exception as e:
        logger.error(f"Error reading cache for feature '{feature_label}': {e}")
        return pd.DataFrame()


def read_multi_features(
        feature_label_collection: FeatureLabelCollection,
        time_range: TimeRange = None,
        columns: typing.List[str] = None,
        cache_context: CacheContext = None,
        max_workers: int = 10,
    ) -> pd.DataFrame:
    """
    Read cached features for multiple feature types and parameters.
    
    Args:
        feature_label_collection: Collection of feature labels and their parameters to read
        time_range: Time range to fetch data for
        columns: Specific columns to retrieve
        cache_context: Cache context containing dataset_mode, export_mode, aggregation_mode
        max_workers: Number of workers for parallel reading (default: 10)
        
    Returns:
        Combined DataFrame with all requested features
    """
    all_dfs = []
    
    # Process each FeatureLabelCollection object
    for feature_label_obj in feature_label_collection:
        df = read_feature(feature_label_obj, time_range, columns, cache_context, max_workers)
        if df is not None and not df.empty:
            all_dfs.append(df)

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