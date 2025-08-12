"""
Resample Parameter Module

This module provides the base class for resampling parameters.
All resample parameter classes should inherit from ResampleParam.
"""

import abc

from market_data.util.param import Param


class ResampleParam(Param):
    """
    Base class for resampling parameters.
    
    This class inherits from the generic Param class and adds resample-specific
    requirements like frequency specification.
    """
    
    @abc.abstractmethod
    def get_target_frequency(self) -> str:
        """
        Get the target frequency for resampling.
        
        Returns:
            str: 'adaptive', ... etc.
        """
        pass
    
