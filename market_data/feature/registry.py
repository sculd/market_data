"""
Feature Registry

This module provides a registry for feature modules and utility functions
to register and retrieve feature implementations by their labels.
"""

from typing import Dict, Any, Callable, TypeVar, Type, Optional, List, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Type variable for feature parameters
P = TypeVar('P')

# Type definition for feature calculation function
FeatureCalculator = Callable[[pd.DataFrame, P], pd.DataFrame]

# The registry that stores mappings from labels to feature modules
_FEATURE_REGISTRY: Dict[str, Any] = {}

def register_feature(label: str):
    """
    Decorator to register a feature module with a specific label.
    
    Args:
        label: Unique identifier for the feature module
        
    Returns:
        Decorator function that registers the module
    """
    def decorator(module):
        if label in _FEATURE_REGISTRY:
            logger.warning(f"Feature label '{label}' is already registered. Overwriting previous registration.")
        _FEATURE_REGISTRY[label] = module
        return module
    return decorator

def get_feature_by_label(label: str) -> Optional[Any]:
    """
    Get a feature module by its label.
    
    Args:
        label: The unique identifier for the feature module
        
    Returns:
        The feature module or None if not found
    """
    module = _FEATURE_REGISTRY.get(label)
    if module is None:
        logger.warning(f"Feature module with label '{label}' not found in registry.")
    return module

def list_registered_features() -> List[str]:
    """
    Get a list of all registered feature labels.
    
    Returns:
        List of registered feature labels
    """
    return list(_FEATURE_REGISTRY.keys()) 