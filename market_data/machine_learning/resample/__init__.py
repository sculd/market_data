"""
Resample Module

This module provides various resampling methods for financial time series data.
All resample methods are automatically registered when the module is imported.
"""

# Import all resample implementations to register them
from market_data.machine_learning.resample.calc import CumSumResampleParams
from market_data.machine_learning.resample.calc_reversal import ResampleReversalParams
# Import the registry functions
from market_data.machine_learning.resample.resample_registry import (
    get_resample_function,
    get_resample_method,
    get_resample_method_info,
    get_resample_params_class,
    is_resample_method_registered,
    list_registered_functions,
    list_registered_params,
    list_registered_resample_methods,
)

# Also import the adaptive method for completeness
try:
    from market_data.machine_learning.resample.calc_adaptive import AdaptiveResampleParams
except ImportError:
    pass  # Skip if adaptive module has issues

__all__ = [
    # Registry functions
    'get_resample_params_class',
    'get_resample_function', 
    'get_resample_method',
    'list_registered_resample_methods',
    'list_registered_params',
    'list_registered_functions',
    'is_resample_method_registered',
    'get_resample_method_info',
    
    # Parameter classes
    'CumSumResampleParams',
    'ResampleReversalParams',
] 