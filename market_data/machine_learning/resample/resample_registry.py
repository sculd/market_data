"""
Resample Registry

This module provides a registry for resample methods and utility functions
to register and retrieve resample implementations by their labels.
"""

from typing import Dict, Any, Callable, TypeVar, Type, Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Type variable for resample parameters
P = TypeVar('P')

# Type definition for resample function
ResampleFunction = Callable[[pd.DataFrame, P], pd.DataFrame]

# Separate registries for parameters and functions
_RESAMPLE_PARAMS_REGISTRY: Dict[str, Type] = {}
_RESAMPLE_FUNCTIONS_REGISTRY: Dict[str, ResampleFunction] = {}

def register_resample_param(label: str):
    """
    Decorator to register a resample parameter class with a specific label.
    """
    def decorator(params_class):
        if label in _RESAMPLE_PARAMS_REGISTRY:
            logger.warning(f"Resample label '{label}' is already registered. Overwriting previous registration.")
        
        _RESAMPLE_PARAMS_REGISTRY[label] = params_class
        
        # Add the label as a class attribute for convenience
        params_class._registry_label = label
        return params_class
    return decorator

def register_resample_function(label: str):
    """
    Decorator to register a resample function for a specific label.
    """
    def decorator(resample_function: ResampleFunction):
        if label not in _RESAMPLE_PARAMS_REGISTRY:
            logger.warning(f"Resample label '{label}' not found in params registry. Registering function anyway.")
        
        _RESAMPLE_FUNCTIONS_REGISTRY[label] = resample_function
        return resample_function
    return decorator

def get_resample_params_class(label: str) -> Optional[Type]:
    """
    Get the parameter class for a resample method by its label.
    """
    return _RESAMPLE_PARAMS_REGISTRY.get(label)

def get_resample_function(label: str) -> Optional[ResampleFunction]:
    """
    Get the resample function for a resample method by its label.
    """
    return _RESAMPLE_FUNCTIONS_REGISTRY.get(label)

def get_resample_method(label: str) -> Optional[Dict[str, Any]]:
    """
    Get the complete resample method (params class + function) by its label.
    """
    params_class = _RESAMPLE_PARAMS_REGISTRY.get(label)
    resample_function = _RESAMPLE_FUNCTIONS_REGISTRY.get(label)
    
    if params_class is None and resample_function is None:
        return None
    
    return {
        'params_class': params_class,
        'resample_function': resample_function
    }

def list_registered_resample_methods() -> List[str]:
    """
    Get a list of all registered resample method labels.
    """
    all_labels = set(_RESAMPLE_PARAMS_REGISTRY.keys()) | set(_RESAMPLE_FUNCTIONS_REGISTRY.keys())
    return sorted(list(all_labels))

def is_resample_method_registered(label: str) -> bool:
    """
    Check if a resample method is registered (either params or function).
    """
    return label in _RESAMPLE_PARAMS_REGISTRY or label in _RESAMPLE_FUNCTIONS_REGISTRY

def get_resample_method_info(label: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a registered resample method.
    """
    params_class = _RESAMPLE_PARAMS_REGISTRY.get(label)
    resample_function = _RESAMPLE_FUNCTIONS_REGISTRY.get(label)
    
    if params_class is None and resample_function is None:
        return None
    
    return {
        'label': label,
        'params_class': params_class,
        'params_class_name': params_class.__name__ if params_class else None,
        'resample_function': resample_function,
        'has_function': resample_function is not None,
        'has_params': params_class is not None,
        'docstring': params_class.__doc__ if params_class else None
    }

def list_registered_params() -> List[str]:
    """
    Get a list of all registered parameter class labels.
    """
    return sorted(list(_RESAMPLE_PARAMS_REGISTRY.keys()))

def list_registered_functions() -> List[str]:
    """
    Get a list of all registered function labels.
    """
    return sorted(list(_RESAMPLE_FUNCTIONS_REGISTRY.keys()))
