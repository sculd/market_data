"""
Path utilities for caching.

This module provides functions for handling file paths and directory names
in the caching system.
"""

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