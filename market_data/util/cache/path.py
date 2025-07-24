"""
Path utilities for caching.

This module provides functions for handling file paths and directory names
in the caching system.
"""

import os
import json
import hashlib
import re
from pathlib import Path
import logging
import datetime

from market_data.ingest.bq.common import get_full_table_id, DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE

logger = logging.getLogger(__name__)

def _sanitize_for_path(value_str: str) -> str:
    """Convert a string to a filesystem-safe string for directory names"""
    # Remove or replace characters that could cause issues in filenames
    value_str = re.sub(r'[\\/*?:"<>|]', '_', value_str)
    # Replace spaces, commas and dots with underscores
    value_str = re.sub(r'[\s,\.]+', '_', value_str)
    # Remove any trailing or leading underscores
    value_str = value_str.strip('_')
    return value_str

def params_to_dir_name(params: dict) -> str:
    """
    Convert a parameters dictionary to a directory name string.
    
    For list parameters, joins the elements with underscores to make them
    human-readable. For other complex parameters like dicts, still uses a hash.
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
            if all(isinstance(x, (int, float, str)) for x in value):
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
            
        # For simple values, sanitize them for filesystem safety
        else:
            safe_key = _sanitize_for_path(key)
            safe_value = _sanitize_for_path(str(value))
            parts.append(f"{safe_key}-{safe_value}")
    
    if not parts:
        return "default"
        
    return "_".join(parts)

def to_filename(basedir: str, label: str, t_from: datetime.datetime, t_to: datetime.datetime, 
               params_dir: str = None, dataset_id: str = None, 
               dataset_mode: DATASET_MODE = None, export_mode: EXPORT_MODE = None, 
               aggregation_mode: AGGREGATION_MODE = None) -> str:
    """
    Generate a filename for the cached data based on time range and parameters.
    
    Parameters:
    -----------
    basedir : str
        Base directory for caching
    label : str
        Type of data (e.g., "features", "targets")
    t_from : datetime.datetime
        Start time of the data
    t_to : datetime.datetime
        End time of the data
    params_dir : str, optional
        Parameters directory name, generated from params_to_dir_name
    dataset_id : str, optional
        Dataset identifier. If provided, this will be included in the path.
        If not provided but dataset_mode, export_mode, and aggregation_mode are provided,
        the dataset_id will be generated using get_full_table_id.
    dataset_mode : DATASET_MODE, optional
        Dataset mode (LIVE, REPLAY, etc.)
    export_mode : EXPORT_MODE, optional
        Export mode (OHLC, TICKS, etc.)
    aggregation_mode : AGGREGATION_MODE, optional
        Aggregation mode (MIN_1, MIN_5, etc.)
    """
    t_str_from = t_from.strftime("%Y-%m-%dT%H:%M:%S%z")
    t_str_to = t_to.strftime("%Y-%m-%dT%H:%M:%S%z")
    
    # Generate dataset_id if not provided but we have the necessary parameters
    if dataset_id is None and all(x is not None for x in [dataset_mode, export_mode]):
        dataset_id = f"{get_full_table_id(dataset_mode, export_mode)}_{str(aggregation_mode)}"

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
        
    fn = os.path.join(dir_path, f"{t_str_from}_{t_str_to}.parquet")
    
    # Ensure directory exists
    Path(os.path.dirname(fn)).mkdir(parents=True, exist_ok=True)
    return fn 

def get_cache_base_path():
    """
    Get the base cache path from environment variable ALGO_CACHE_BASE 
    or default to ~/algo_cache.
    
    Returns:
        str: Expanded cache base path
    """
    base_path = os.environ.get('ALGO_CACHE_BASE', '~/algo_cache')
    return os.path.expanduser(base_path) 