import os

from market_data.ingest.bq.cache import to_filename, _cache_base_path
from market_data.ingest.bq.common import DATASET_MODE, EXPORT_MODE, AGGREGATION_MODE, get_full_table_id
from market_data.util.time import TimeRange
from market_data.util.cache.time import split_t_range
from market_data.machine_learning.resample import ResampleParams

def group_consecutive_dates(date_ranges):
    """
    Group consecutive date ranges into larger ranges.
    
    Args:
        date_ranges: List of (start_date, end_date) tuples
        
    Returns:
        List of (start_date, end_date) tuples with consecutive dates grouped
    """
    if not date_ranges:
        return []
    
    # Sort by start date
    sorted_ranges = sorted(date_ranges, key=lambda x: x[0])
    
    grouped_ranges = []
    current_start, current_end = sorted_ranges[0]
    
    for i in range(1, len(sorted_ranges)):
        next_start, next_end = sorted_ranges[i]
        
        # If next range starts on the same day as current range ends,
        # they are consecutive
        if next_start.date() == current_end.date():
            current_end = next_end
        else:
            grouped_ranges.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    
    # Don't forget to add the last range
    grouped_ranges.append((current_start, current_end))
    
    return grouped_ranges

def parse_resample_params(param_str):
    """
    Parse a string in the format 'price_col,threshold' into ResampleParams.
    Example: 'close,0.07' -> ResampleParams(price_col='close', threshold=0.07)
    
    Args:
        param_str: String in format 'price_col,threshold'
        
    Returns:
        ResampleParams instance
    """
    if not param_str:
        return ResampleParams()
        
    try:
        parts = param_str.split(',')
        if len(parts) != 2:
            raise ValueError("Format should be 'price_col,threshold'")
            
        price_col = parts[0].strip()
        threshold = float(parts[1].strip())
        
        return ResampleParams(price_col=price_col, threshold=threshold)
    except Exception as e:
        raise ValueError(f"Invalid resample_params format: {e}. Format should be 'price_col,threshold' (e.g. 'close,0.07')")
