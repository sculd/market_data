import logging
import inspect
import importlib
from typing import Optional, Any, Union, Tuple, List
from market_data.feature.registry import get_feature_by_label, list_registered_features


import logging
logger = logging.getLogger(__name__)

def _create_default_params(feature_module, feature_label: str) -> Optional[Any]:
    """
    Create a default instance of the parameter class for a feature module.
    
    Args:
        feature_module: The feature module instance
        feature_label: The feature label
        
    Returns:
        An instance of the parameter class, or None if not found
    """
    try:
        # Try different naming patterns to find the params class
        # Common patterns include: ReturnParams, VolatilityParams, etc.
        
        # Pattern 1: Try to find a class attribute with 'Params' in the name
        for name, obj in inspect.getmembers(feature_module):
            if inspect.isclass(obj) and 'Params' in name:
                return obj()
        
        # Pattern 2: Try to import from the impl module directly
        try:
            # Assuming the module structure is like market_data.feature.impl.returns
            module_path = f"market_data.feature.impl.{feature_label}"
            module = importlib.import_module(module_path)
            
            # Look for a class ending with "Params"
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.endswith('Params'):
                    return obj()
        except (ImportError, AttributeError):
            pass
        
        # Pattern 3: Check if the module has a calculate method that accepts None as params
        if hasattr(feature_module, 'calculate'):
            # If calculate method handles None params by creating default ones,
            # we can use None directly
            return None
        
        return None
    except Exception as e:
        logger.warning(f"Error creating default params for {feature_label}: {e}")
        return None


def parse_feature_label_param(feature_label_param: Union[str, Tuple[str, Any]]) -> Tuple[str, Any]:
    """
    Parse a feature label and parameters.
    
    Args:
        feature_label_param: Either a feature label string or a tuple of (feature_label, parameters).
                             If only a feature label is provided, a default parameter instance will be created.
    Returns:
        A tuple of (feature_label, parameters)
    """
    # Parse feature_label and params
    if isinstance(feature_label_param, str):
        feature_label = feature_label_param
        params = None
    else:
        feature_label, params = feature_label_param
    
    # Create default params if needed
    if params is None:
        # Get feature module
        feature_module = get_feature_by_label(feature_label)
        if feature_module is None:
            raise ValueError(f"Feature module '{feature_label}' not found in registry")
        
        params = _create_default_params(feature_module, feature_label)
        if params is None:
            raise ValueError(f"Failed to create default parameters for feature '{feature_label}'")
    
    # Verify params has required method
    if not hasattr(params, 'get_params_dir'):
        raise ValueError(f"Parameters object for feature '{feature_label}' must have get_params_dir method")

    return feature_label, params


def parse_feature_label_params(
    feature_label_params: Optional[List[Union[str, Tuple[str, Any]]]] = None,
    ) -> List[Tuple[str, Any]]:
    """
    Parse a list of feature labels and parameters.
    
    Args:
        feature_label_params: Optional list of feature labels and parameters.
                             If None, all registered features with default parameters will be used.
    Returns:
        A list of tuples of (feature_label, parameters)
    """
    ret = []

    # If feature_labels_params is None, use all available feature labels with default parameters
    if feature_label_params is None:
        logger.info("No specific features specified, using all available features with default parameters")
        feature_labels = list_registered_features()
        for feature_label in feature_labels:
            feature_label, params = parse_feature_label_param(feature_label)
            ret.append((feature_label, params))
    else:
        for feature_label_param in feature_label_params:
            feature_label, params = parse_feature_label_param(feature_label_param)
            ret.append((feature_label, params))

    return ret
