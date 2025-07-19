"""
Time utilities for caching.

This module provides functions for handling time ranges and intervals
in the caching system.
"""

import datetime
import pytz
import logging
from typing import Callable, List, Tuple
from market_data.util.time import TimeRange

logger = logging.getLogger(__name__)

# Cache by day
CACHE_INTERVAL = datetime.timedelta(days=1)
# Timezone for day boundaries 
CACHE_TIMEZONE = pytz.timezone('America/New_York')

def anchor_to_begin_of_day(t: datetime.datetime) -> datetime.datetime:
    """Anchor a timestamp to the beginning of its day"""
    return CACHE_TIMEZONE.localize(datetime.datetime(year=t.year, month=t.month, day=t.day, hour=0, minute=0, second=0))

def split_t_range(t_from: datetime.datetime, t_to: datetime.datetime, 
                 interval: datetime.timedelta = CACHE_INTERVAL,
                 warm_up: datetime.timedelta = None) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Split a time range into intervals, with an optional warm-up period for each interval.
    
    Parameters:
    -----------
    t_from : datetime.datetime
        Start time of the entire range
    t_to : datetime.datetime
        End time of the entire range
    interval : datetime.timedelta
        Step size for advancing through the range (default is CACHE_INTERVAL = 1 day)
    warm_up : datetime.timedelta, optional
        Optional warm-up period to include at the beginning of each range
        
    Returns:
    --------
    List of (start_time, end_time) tuples, where:
        - Each range moves forward by 'interval'
        - If warm_up is provided, each start_time is adjusted to include the warm-up period
        - The warm-up period overlaps with the previous interval
    
    Examples:
    ---------
    With interval=1 day, no warm-up:
        [(day1, day2), (day2, day3), (day3, day4)]
    
    With interval=1 day, warm_up=2 days:
        [(day1, day2), (day1-2days, day3), (day2-2days, day4)]
    """
    ret = []
    # Anchor times to beginning of day
    if warm_up is not None:
        t1 = anchor_to_begin_of_day(t_from - warm_up)
    else:
        t1 = anchor_to_begin_of_day(t_from)
    t2 = anchor_to_begin_of_day(t_from + interval)
    ret.append((t1, t2))
    
    # Process subsequent ranges with warm-up if specified
    while t2 < t_to:
        # Move forward by interval for the end time
        t2 = anchor_to_begin_of_day(t2 + interval)
        
        # Calculate start time, adjusted for warm-up if provided
        if warm_up is not None:
            # Start earlier by warm_up period
            t1_with_warmup = anchor_to_begin_of_day(t2 - interval - warm_up)
        else:
            # Without warm-up, start at previous end
            t1_with_warmup = anchor_to_begin_of_day(t2 - interval)
        
        # Add the range to our list
        ret.append((t1_with_warmup, t2))
    
    # Adjust the last range if it exceeds t_to
    last = (ret[-1][0], min(ret[-1][1], t_to))
    ret[-1] = last
    
    return ret


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


def chop_missing_time_range(
        missing_range_finder_func: Callable, time_range: TimeRange, 
        overwrite_cache: bool, calculation_batch_days: int = 1) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    '''
    Chop a time range into calculation batches.
    If overwrite_cache is True, process all ranges.
    If overwrite_cache is False, only process missing ranges.

    Parameters:
    -----------
    missing_range_finder_func: Callable
        A function that returns a list of missing time ranges. Should take a TimeRange as input and return a list of [from, to] tuples.
    time_range: TimeRange
        The time range to chop.
    overwrite_cache: bool
        If True, process all ranges.
        If False, only process missing ranges.
    calculation_batch_days: int
        The number of days to process in each batch.

    Returns:
    --------
    A list of [from, to] tuples.
    '''
    calculation_interval = datetime.timedelta(days=calculation_batch_days)
    # Determine which ranges need to be calculated
    if overwrite_cache:
        # If overwriting cache, process all ranges
        t_from, t_to = time_range.to_datetime()
        calculation_ranges = split_t_range(t_from, t_to, interval=calculation_interval)
        print(f"  Overwrite cache enabled - processing all {len(calculation_ranges)} ranges")
    else:
        # If not overwriting cache, only process missing ranges
        missing_ranges = missing_range_finder_func(time_range=time_range)
        
        if not missing_ranges:
            print("  All resampled data already cached - skipping calculation")
            return
        
        # Group consecutive missing ranges and split into calculation batches
        grouped_ranges = group_consecutive_dates(missing_ranges)
        calculation_ranges = []
        
        for grouped_start, grouped_end in grouped_ranges:
            # Split each grouped range into calculation batches
            batch_ranges = split_t_range(grouped_start, grouped_end, interval=calculation_interval)
            calculation_ranges.extend(batch_ranges)
        
        print(f"  Found {len(missing_ranges)} missing days, grouped into {len(grouped_ranges)} ranges, "
                f"split into {len(calculation_ranges)} calculation batches")

    return calculation_ranges

def is_exact_cache_interval(t_from: datetime.datetime, t_to: datetime.datetime) -> bool:
    """Check if time range is exactly one cache interval (day) starting at zero hour"""
    t_from_plus_interval = anchor_to_begin_of_day(t_from + CACHE_INTERVAL)
    #t_from_plus_interval = t_from + CACHE_INTERVAL
    if t_to != t_from_plus_interval:
        return False

    def at_begin_of_day(t: datetime.datetime) -> bool:
        return t.hour == 0 and t.minute == 0 and t.second == 0

    return at_begin_of_day(t_from) and at_begin_of_day(t_to) 