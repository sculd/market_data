"""
Path and directory utilities for caching.

This module provides functions for handling paths, directories, and file names
for the caching system.
"""

import os
import json
import hashlib
import re
from pathlib import Path
from typing import Optional, Dict, Any

def _sanitize_for_path(value_str: str) -> str:
    """Convert a string to a filesystem-safe string for directory names"""
    # Replace any non-alphanumeric characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', str(value_str))
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized

def params_to_dir_name(params: Dict[str, Any]) -> str:
    """
    Convert a dictionary of parameters to a directory name string.
    
    Args:
        params: Dictionary of parameter names and values
        
    Returns:
        String suitable for use as a directory name
    """
    if not params:
        return ''
        
    # Sort items to ensure consistent ordering
    items = sorted(params.items())
    
    # Create a list of "key=value" strings
    param_strings = []
    for key, value in items:
        if isinstance(value, (list, tuple)):
            # For lists/tuples, join with underscores
            value_str = '_'.join(str(x) for x in value)
        else:
            value_str = str(value)
            
        # Sanitize both key and value
        safe_key = _sanitize_for_path(key)
        safe_value = _sanitize_for_path(value_str)
        
        param_strings.append(f"{safe_key}={safe_value}")
    
    # Join all parameter strings with underscores
    return '_'.join(param_strings)

def to_filename(basedir: str, label: str, t_from: str, t_to: str, params_dir: Optional[str] = None, dataset_id: Optional[str] = None) -> str:
    """
    Generate a filename for caching data.
    
    Args:
        basedir: Base directory for cache
        label: Label for the data type (e.g., 'features', 'targets')
        t_from: Start time
        t_to: End time
        params_dir: Optional directory name for parameters
        dataset_id: Optional dataset identifier
        
    Returns:
        Full path to the cache file
    """
    # Ensure the base directory exists
    Path(basedir).mkdir(parents=True, exist_ok=True)
    
    # Create the directory structure
    if dataset_id:
        base_dir = os.path.join(basedir, label, dataset_id)
    else:
        base_dir = os.path.join(basedir, label)
    
    # Add parameters directory if provided
    if params_dir:
        base_dir = os.path.join(base_dir, params_dir)
    
    # Ensure the directory exists
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    # Create the filename
    filename = f"{t_from}_{t_to}.parquet"
    
    # Return the full path
    return os.path.join(base_dir, filename)

def get_cache_path(dataset_mode: str, export_mode: str, aggregation_mode: str, time_range: Any) -> str:
    """
    Get the cache path for a specific dataset and time range.
    
    Args:
        dataset_mode: Dataset mode (e.g., 'OKX', 'BITHUMB')
        export_mode: Export mode (e.g., 'BY_MINUTE', 'ORDERBOOK_LEVEL1')
        aggregation_mode: Aggregation mode (e.g., 'TAKE_LASTEST')
        time_range: Time range object
        
    Returns:
        Cache path string
    """
    t_from, t_to = time_range.to_datetime()
    return f"{dataset_mode}_{export_mode}_{str(aggregation_mode)}_{t_from}_{t_to}"

def get_full_table_id(dataset_mode: str, export_mode: str) -> str:
    """
    Get the full table ID for a dataset and export mode.
    
    Args:
        dataset_mode: Dataset mode (e.g., 'OKX', 'BITHUMB')
        export_mode: Export mode (e.g., 'BY_MINUTE', 'ORDERBOOK_LEVEL1')
        
    Returns:
        Full table ID string
    """
    return f"{dataset_mode}_{export_mode}" 