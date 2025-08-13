import datetime
import typing
from dataclasses import dataclass
from typing import Optional, Tuple

import pytz


@dataclass
class TimeRange:
    """
    Encapsulates time range parameters for data queries.
    
    Only one pair of parameters should be provided:
    - (t_from, t_to): datetime objects
    - (epoch_seconds_from, epoch_seconds_to): Unix timestamps
    - (date_str_from, date_str_to): Date strings in YYYY-MM-DD format
    
    Attributes:
        t_from: Start time as datetime
        t_to: End time as datetime
        epoch_seconds_from: Start time as Unix timestamp
        epoch_seconds_to: End time as Unix timestamp
        date_str_from: Start date as string (YYYY-MM-DD)
        date_str_to: End date as string (YYYY-MM-DD)
    """
    t_from: Optional[datetime.datetime] = None
    t_to: Optional[datetime.datetime] = None
    epoch_seconds_from: Optional[int] = None
    epoch_seconds_to: Optional[int] = None
    date_str_from: Optional[str] = None
    date_str_to: Optional[str] = None
    
    def __post_init__(self):
        """Validate that only one pair of parameters is provided."""
        # Count how many pairs are provided
        pairs_provided = 0
        if self.t_from is not None and self.t_to is not None:
            pairs_provided += 1
        if self.epoch_seconds_from is not None and self.epoch_seconds_to is not None:
            pairs_provided += 1
        if self.date_str_from is not None and self.date_str_to is not None:
            pairs_provided += 1
            
        if pairs_provided != 1:
            raise ValueError(
                "Exactly one pair of time parameters must be provided: "
                "(t_from, t_to), (epoch_seconds_from, epoch_seconds_to), "
                "or (date_str_from, date_str_to)"
            )
    
    def to_datetime(self) -> Tuple[datetime.datetime, datetime.datetime]:
        """
        Convert the time range to a tuple of datetime objects.
        
        Returns:
            Tuple of (start_time, end_time) as datetime objects
        """
        if self.t_from is not None and self.t_to is not None:
            return self.t_from, self.t_to
        elif self.epoch_seconds_from is not None and self.epoch_seconds_to is not None:
            return (
                epoch_seconds_to_datetime(self.epoch_seconds_from),
                epoch_seconds_to_datetime(self.epoch_seconds_to)
            )
        elif self.date_str_from is not None and self.date_str_to is not None:
            return (
                date_str_to_et_t(self.date_str_from),
                date_str_to_et_t(self.date_str_to)
            )
        else:
            raise ValueError("No valid time range parameters provided")

def truncate_timestamp_seconds_to_minute(timestamp_seconds):
    t_tz = epoch_seconds_to_datetime(timestamp_seconds)
    t_tz_minute = t_tz.replace(second=0, microsecond=0)
    return t_tz_minute

def epoch_seconds_to_datetime(timestamp_seconds):
    t = datetime.datetime.utcfromtimestamp(timestamp_seconds)
    return pytz.utc.localize(t)

def epoch_seconds_to_str(timestamp_seconds):
    return str(epoch_seconds_to_datetime(timestamp_seconds))

def date_str_to_et_t(date_str):
    '''
    Convert a date string to a datetime object in Eastern timezone.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        datetime object localized to Eastern timezone
    '''
    t = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return pytz.timezone('America/New_York').localize(t)

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
