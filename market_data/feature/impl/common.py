"""
Common feature parameters and utilities.
"""

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
        
