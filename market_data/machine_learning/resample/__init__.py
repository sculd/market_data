"""
Resample Module

This module provides various resampling methods for financial time series data.
All resample methods are automatically registered when the module is imported.
"""

# Import the registry functions
from .resample_registry import (
    get_resample_params_class,
    get_resample_function,
    get_resample_method,
    list_registered_resample_methods,
    list_registered_params,
    list_registered_functions,
    is_resample_method_registered,
    get_resample_method_info
)

# Import all resample implementations to register them
from .resample import ResampleParams, resample_at_events
from .resample_reversal import ResampleReversalParams

# Also import the adaptive method for completeness
try:
    from .resample_adaptive import AdaptiveResampleParams
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
    'ResampleParams',
    'ResampleReversalParams',
    
    # Functions (for backward compatibility)
    'resample_at_events',
] 