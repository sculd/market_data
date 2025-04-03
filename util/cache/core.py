"""
Core caching functions for reading and writing data.

This module provides the fundamental functions for caching data,
including reading from cache, querying data, and writing to cache.
"""

import pandas as pd
import logging
import os
from typing import Optional, List, Union
from datetime import datetime

from ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE
from util.time import TimeRange
from .paths import get_cache_path, get_full_table_id

logger = logging.getLogger(__name__)

def read_from_cache_or_query_and_cache(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    overwirte_cache: bool = False
) -> Optional[pd.DataFrame]:
    """
    Read data from cache if available, otherwise query and cache it.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        overwirte_cache: Whether to overwrite existing cache files
        
    Returns:
        DataFrame with the data, or None if no data is available
    """
    # Try to read from cache first
    df = read_from_cache(dataset_mode, export_mode, aggregation_mode, time_range)
    if df is not None:
        return df
        
    # If not in cache or overwrite requested, query and cache
    return query_and_cache(dataset_mode, export_mode, aggregation_mode, time_range, overwirte_cache)

def read_from_cache(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange
) -> Optional[pd.DataFrame]:
    """
    Read data from cache if available.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        
    Returns:
        DataFrame with the data, or None if not in cache
    """
    cache_path = get_cache_path(dataset_mode, export_mode, aggregation_mode, time_range)
    if not os.path.exists(cache_path):
        return None
        
    try:
        return pd.read_parquet(cache_path)
    except Exception as e:
        logger.error(f"Error reading from cache: {e}")
        return None

def query_and_cache(
    dataset_mode: DATASET_MODE,
    export_mode: EXPORT_MODE,
    aggregation_mode: AGGREGATION_MODE,
    time_range: TimeRange,
    overwrite_cache: bool = False
) -> Optional[pd.DataFrame]:
    """
    Query data from BigQuery and cache it.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        overwrite_cache: Whether to overwrite existing cache files
        
    Returns:
        DataFrame with the data, or None if query fails
    """
    # TODO: Implement BigQuery query logic
    # This is a placeholder - the actual implementation would depend on
    # your BigQuery setup and query requirements
    pass 