"""
Feature Parameter Module

This module provides the base class for feature calculation parameters.
All feature parameter classes should inherit from FeatureParam.
"""

import abc
import datetime
import math

from market_data.util.param import Param


class FeatureParam(Param):
    """
    Base class for feature calculation parameters.
    
    This class inherits from the generic Param class and adds feature-specific
    requirements like warm-up period calculation.
    """
    
    @abc.abstractmethod
    def get_warm_up_period(self) -> datetime.timedelta:
        """
        Get the warm-up period required for feature calculation.
        
        Returns:
            datetime.timedelta: The minimum warm-up period needed for this feature
        """
        pass

    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period based on the Bollinger Bands period.

        It assumes that the period is in minutes and the data is 24/7.
        
        Returns:
            int: Recommended number of warm-up days
        """
        warm_up_period = self.get_warm_up_period()
        days_needed = math.ceil(warm_up_period.total_seconds() / (24 * 60 * 60))
        
        return days_needed
        