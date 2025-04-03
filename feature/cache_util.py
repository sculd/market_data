"""
Backward compatibility layer for cache utilities.

This module re-exports functions from the new unified cache module
to maintain backward compatibility with existing code.
"""

from util.cache import (
    # Path utilities
    params_to_dir_name,
    to_filename,
    
    # Time utilities
    anchor_to_begin_of_day,
    split_t_range,
    is_exact_cache_interval,
    
    # DataFrame utilities
    cache_daily_df,
    fetch_from_daily_cache,
    read_from_cache_generic,
    cache_data_by_day,
    
    # Constants
    CACHE_INTERVAL,
    CACHE_TIMEZONE,
    TIMESTAMP_INDEX_NAME
)

__all__ = [
    # Path utilities
    'params_to_dir_name',
    'to_filename',
    
    # Time utilities
    'anchor_to_begin_of_day',
    'split_t_range',
    'is_exact_cache_interval',
    
    # DataFrame utilities
    'cache_daily_df',
    'fetch_from_daily_cache',
    'read_from_cache_generic',
    'cache_data_by_day',
    
    # Constants
    'CACHE_INTERVAL',
    'CACHE_TIMEZONE',
    'TIMESTAMP_INDEX_NAME'
]
