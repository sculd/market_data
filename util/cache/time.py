"""
Time utilities for caching.

This module provides functions for handling time ranges and intervals
in the caching system.
"""

import datetime
import pytz
import logging
import typing
from util.time import TimeRange

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
                 warm_up: datetime.timedelta = None) -> typing.List[typing.Tuple[datetime.datetime, datetime.datetime]]:
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

def is_exact_cache_interval(t_from: datetime.datetime, t_to: datetime.datetime) -> bool:
    """Check if time range is exactly one cache interval (day) starting at zero hour"""
    t_from_plus_interval = anchor_to_begin_of_day(t_from + CACHE_INTERVAL)
    if t_to != t_from_plus_interval:
        return False

    def at_begin_of_day(t: datetime.datetime) -> bool:
        return t.hour == 0 and t.minute == 0 and t.second == 0

    return at_begin_of_day(t_from) and at_begin_of_day(t_to) 