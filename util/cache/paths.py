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

# The base directory for cache
CACHE_BASE_PATH = os.path.expanduser('~/algo_cache')
Path(CACHE_BASE_PATH).mkdir(parents=True, exist_ok=True)

def _sanitize_for_path(value_str: str) -> str:
    """Convert a string to a filesystem-safe string for directory names"""
    # Remove or replace characters that could cause issues in filenames
    value_str = re.sub(r'[\\/*?:"<>|]', '_', value_str)
    # Replace spaces, commas and dots with underscores
    value_str = re.sub(r'[\s,\.]+', '_', value_str)
    # Remove any trailing or leading underscores
    value_str = value_str.strip('_')
    return value_str

def params_to_dir_name(params: Dict[str, Any]) -> str:
    """
    Convert a parameters dictionary to a directory name string.
    
    For list parameters, joins the elements with underscores to make them
    human-readable. For other complex parameters like dicts, still uses a hash.
    
    Args:
        params: Dictionary of parameters to convert to a directory name
        
    Returns:
        String suitable for use as a directory name
    """
    if not params:
        return "default"
        
    parts = []
    for key, value in sorted(params.items()):
        # Skip None values
        if value is None:
            continue
            
        # For list values, create a human-readable string with actual values
        if isinstance(value, list):
            # Handle different types of elements
            if all(isinstance(x, (int, float)) for x in value):
                # For numeric lists, join with underscores
                value_str = f"{key}-" + "_".join(str(x) for x in value)
                
                # Truncate if too long (over 50 chars)
                if len(value_str) > 50:
                    value_str = f"{key}-len{len(value)}-" + "_".join(str(x) for x in value[:3]) + "..."
            else:
                # For non-numeric lists or mixed lists, still use length and hash
                value_str = json.dumps(value, sort_keys=True)
                value_hash = hashlib.md5(value_str.encode()).hexdigest()[:6]
                value_str = f"{key}-len{len(value)}-{value_hash}"
                
            parts.append(_sanitize_for_path(value_str))
            
        # For dictionaries or sets, just use a hash
        elif isinstance(value, (dict, set)):
            value_str = json.dumps(sorted(value) if isinstance(value, set) else value, sort_keys=True)
            value_hash = hashlib.md5(value_str.encode()).hexdigest()[:8]
            parts.append(f"{key}-{value_hash}")
            
        # For simple values, use them directly
        else:
            parts.append(f"{key}-{value}")
    
    if not parts:
        return "default"
        
    return "_".join(parts)

def to_filename(basedir: str, label: str, t_from: str, t_to: str, params_dir: Optional[str] = None, dataset_id: Optional[str] = None) -> str:
    """
    Generate a filename for the cached data based on time range and parameters.
    
    Args:
        basedir: Base directory for caching
        label: Type of data (e.g., "features", "targets")
        t_from: Start time of the data
        t_to: End time of the data
        params_dir: Parameters directory name, generated from params_to_dir_name
        dataset_id: Dataset identifier (e.g., "trading-290017.market_data_okx.by_minute_AGGREGATION_MODE.TAKE_LASTEST")
        
    Returns:
        Full path to the cache file
    """
    # Base directory structure with dataset_id if provided
    if dataset_id:
        base_dir = os.path.join(basedir, label, dataset_id)
    else:
        base_dir = os.path.join(basedir, label)
    
    # Include params in the directory structure if provided
    if params_dir:
        dir_path = os.path.join(base_dir, params_dir)
    else:
        dir_path = base_dir
        
    fn = os.path.join(dir_path, f"{t_from}_{t_to}.parquet")
    
    # Ensure directory exists
    Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)
    return fn

def get_cache_path(dataset_mode: str, export_mode: str, aggregation_mode: str, time_range: Any) -> str:
    """
    Get the cache path for a specific dataset and time range.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        aggregation_mode: Aggregation mode (MIN_1, MIN_5, etc.)
        time_range: TimeRange object specifying the time range
        
    Returns:
        Full path to the cache file
    """
    t_from, t_to = time_range.to_datetime()
    dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{aggregation_mode}"
    return to_filename(CACHE_BASE_PATH, "raw_data", t_from, t_to, dataset_id=dataset_id)

def get_full_table_id(dataset_mode: str, export_mode: str) -> str:
    """
    Get the full table ID for BigQuery.
    
    Args:
        dataset_mode: Dataset mode (LIVE, REPLAY, etc.)
        export_mode: Export mode (OHLC, TICKS, etc.)
        
    Returns:
        Full table ID string
    """
    return f"trading-290017.market_data_okx.{dataset_mode.lower()}.{export_mode.lower()}" 