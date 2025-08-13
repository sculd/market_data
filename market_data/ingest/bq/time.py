import datetime
import typing

from market_data.util.time import TimeRange

def to_t(
        t_from: datetime.datetime = None,
        t_to: datetime.datetime = None,
        epoch_seconds_from: int = None,
        epoch_seconds_to: int = None,
        date_str_from: str = None,
        date_str_to: str = None,
        ) -> typing.Tuple[datetime.datetime, datetime.datetime]:
    """
    Convert various time range parameters to a tuple of datetime objects.
    
    Args:
        t_from: Start time as datetime
        t_to: End time as datetime
        epoch_seconds_from: Start time as Unix timestamp
        epoch_seconds_to: End time as Unix timestamp
        date_str_from: Start date as string (YYYY-MM-DD)
        date_str_to: End date as string (YYYY-MM-DD)
        
    Returns:
        Tuple of (start_time, end_time) as datetime objects
    """
    time_range = TimeRange(
        t_from=t_from,
        t_to=t_to,
        epoch_seconds_from=epoch_seconds_from,
        epoch_seconds_to=epoch_seconds_to,
        date_str_from=date_str_from,
        date_str_to=date_str_to
    )
    return time_range.to_datetime()

def t_to_bq_t_str(t: datetime.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S%z")
