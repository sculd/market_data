"""
Path utilities for caching.

This module provides functions for handling file paths and directory names
in the caching system.
"""

import datetime
import logging
import os

logger = logging.getLogger(__name__)


def get_cache_base_path():
    """
    Get the base cache path from environment variable ALGO_CACHE_BASE 
    or default to ~/algo_cache.
    
    Returns:
        str: Expanded cache base path
    """
    base_path = os.environ.get('ALGO_CACHE_BASE', '~/algo_cache')
    return os.path.expanduser(base_path) 


def to_local_filename(folder_path: str, t_from: datetime.date, t_to: datetime.datetime) -> str:
    t_str_from = t_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_str_to = t_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    fn = os.path.join(folder_path, f"{t_str_from}_{t_str_to}.parquet")
    dir = os.path.dirname(fn)
    try:
        os.makedirs(dir, exist_ok=True)
    except FileExistsError:
        pass

    return fn
