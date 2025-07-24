"""
Time of Day Features Module

This module provides functions for calculating time-of-day features
such as hour of day and trading session bins based on US market hours.
"""

import pandas as pd
import numpy as np
import logging
import datetime
import pytz
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from market_data.feature.registry import register_feature

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "time_of_day"

@dataclass
class TimeOfDayParams:
    """Parameters for time of day feature calculations."""
    timezone: str = "US/Eastern"  # US stock market timezone
    
    def get_params_dir(self) -> str:
        """
        Generate a directory name string from parameters.
        
        Returns:
            Directory name string for caching
        """
        from market_data.util.cache.path import params_to_dir_name
        
        params_dict = {
            'tz': self.timezone.replace('/', '_')
        }
        return params_to_dir_name(params_dict)
        
    def get_warm_up_period(self) -> datetime.timedelta:
        # Time features don't need warm-up
        return datetime.timedelta(minutes=0)

    def get_warm_up_days(self) -> int:
        """
        Time features don't require warm-up data. Returns 0.x
        """
        return 0

def _get_time_bin_vectorized(dt_series: pd.Series) -> pd.Series:
    """
    Vectorized version of time bin calculation.
    
    Args:
        dt_series: Series of datetime objects in US Eastern Time
        
    Returns:
        Series of time bins (0-6)
    """
    hours = dt_series.hour
    minutes = dt_series.minute
    
    # Create conditions for each bin
    conditions = [
        (hours < 9) | ((hours == 9) & (minutes < 30)),  # Bin 0: Before 9:30 AM
        ((hours == 9) & (minutes >= 30)) | ((hours == 10) & (minutes < 30)),  # Bin 1: 9:30-10:29 AM
        ((hours == 10) & (minutes >= 30)) | (hours == 11),  # Bin 2: 10:30-11:59 AM
        (hours >= 12) & (hours < 14),  # Bin 3: 12:00-1:59 PM  
        (hours >= 14) & (((hours == 15) & (minutes < 30)) | (hours < 15)),  # Bin 4: 2:00-3:29 PM
        ((hours == 15) & (minutes >= 30)) | ((hours == 16) & (minutes == 0)),  # Bin 5: 3:30-4:00 PM
    ]
    
    choices = [0, 1, 2, 3, 4, 5]
    
    # Default case (after 4 PM) is bin 6
    return pd.Series(np.select(conditions, choices, default=6), index=dt_series)

@register_feature(FEATURE_LABEL)
class TimeOfDayFeature:
    """Time of day feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[TimeOfDayParams] = None) -> pd.DataFrame:
        """
        Calculate time-of-day features.
        
        Args:
            df: Input DataFrame with OHLCV data and timestamp index
            params: Parameters for time of day calculation
            
        Returns:
            DataFrame with calculated time of day features
        """
        if params is None:
            params = TimeOfDayParams()
            
        logger.info(f"Calculating time of day features with timezone {params.timezone}")
        
        # Ensure we have timestamp index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                # Use timestamp column
                timestamps = df['timestamp']
            else:
                raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")
        else:
            timestamps = df.index
            
        # Ensure we have symbol column
        if 'symbol' not in df.columns:
            raise ValueError("Symbol column not found in DataFrame")
        
        # Convert timestamps to US Eastern Time
        et_tz = pytz.timezone(params.timezone)
        
        # Handle timezone conversion
        if timestamps.tz is None:
            # Assume UTC if no timezone info
            timestamps_utc = timestamps.tz_localize('UTC')
            timestamps_et = timestamps_utc.tz_convert(et_tz)
        else:
            # Convert directly to Eastern Time (no need for UTC intermediate step)
            timestamps_et = timestamps.tz_convert(et_tz)
        
        # Create result DataFrame with same index as input
        result = pd.DataFrame(index=df.index)
        
        # Calculate time features for unique timestamps (market-wide features)
        unique_timestamps = timestamps_et.drop_duplicates()
        
        # Calculate hour of day (0-23) for unique timestamps
        hour_of_day = pd.Series(unique_timestamps.hour, index=unique_timestamps)
        hour_of_day = hour_of_day.groupby(hour_of_day.index).mean().reindex(df.index)
        
        # Calculate time of day bin (0-6) for unique timestamps  
        time_bin = _get_time_bin_vectorized(unique_timestamps)
        time_bin = time_bin.groupby(time_bin.index).mean().reindex(df.index)
        
        # Assign to result DataFrame
        result['hour_of_day'] = hour_of_day
        result['time_of_day_bin'] = time_bin
        
        # Add symbol column (this is a market-wide feature, so same for all symbols)
        result['symbol'] = df['symbol']
        
        logger.info(f"Generated time features with {result['time_of_day_bin'].nunique()} unique time bins")
        
        # Convert to proper MultiIndex format
        result = result.reset_index().set_index(['timestamp', 'symbol'])
        
        return result 