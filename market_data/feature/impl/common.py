"""
Common feature parameters and utilities.
"""

from typing import Any, Dict
from market_data.util.cache.path import params_to_dir_name

class SequentialFeatureParam:
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
        
    def get_params_dir(self) -> str:
        """
        Get the directory name for caching sequential features.
        
        Returns:
            str: Directory name based on sequence window
        """
        params_dict = {
            'sequence_window': self.sequence_window
        }
        return params_to_dir_name(params_dict) 