"""
Path utilities for caching.

This module provides functions for handling file paths and directory names
in the caching system.
"""

import hashlib
import json
import logging
import os
import re

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
            
        # For simple values
        else:
            parts.append(f"{key}-{value}")
    
    if not parts:
        return "default"
        
    return "_".join(parts)

def get_cache_base_path():
    """
    Get the base cache path from environment variable ALGO_CACHE_BASE 
    or default to ~/algo_cache.
    
    Returns:
        str: Expanded cache base path
    """
    base_path = os.environ.get('ALGO_CACHE_BASE', '~/algo_cache')
    return os.path.expanduser(base_path) 