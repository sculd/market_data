"""
Feature Parameter Module

This module provides the base class for feature calculation parameters.
All feature parameter classes should inherit from FeatureParam.
"""

import abc
import datetime
import math
import os

from market_data.util.param import Param


class SequentialFeatureParam(Param):
    """
    Parameters for sequential feature calculation.
    
    Attributes:
        sequence_window: Number of consecutive values to include in the sequence (default: 60)
    """
    
    def __init__(self, sequence_window: int = 60):
        """
        Initialize sequential feature parameters.
        
        Args:
            sequence_window: Number of consecutive values to include in the sequence
        """
        self.sequence_window = sequence_window
    
    @classmethod
    def from_str(cls, param_str: str) -> 'SequentialFeatureParam':
        """Parse parameters from string format: sequence_window:60"""
        if not param_str:
            return cls()
        
        if ':' in param_str:
            key, value = param_str.split(':', 1)
            if key == 'sequence_window':
                return cls(sequence_window=int(value))
        
        # If format doesn't match, try to parse as just the number
        try:
            sequence_window = int(param_str)
            return cls(sequence_window=sequence_window)
        except ValueError:
            return cls()
    
    def to_str(self) -> str:
        """Convert parameters to string format: sequence_window:60"""
        return f"sequence_window:{self.sequence_window}"


class FeatureParam(Param):
    """
    Base class for feature calculation parameters.
    
    This class inherits from the generic Param class and adds feature-specific
    requirements like warm-up period calculation.
    """
    
    def get_params_dir(self, seq_param: SequentialFeatureParam = None) -> str:
        if seq_param is not None:
            seq_dir = f"seq_{seq_param.get_params_dir()}"
            return os.path.join(self.to_str(), seq_dir)
        
        return self.to_str()
    
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
