import logging
import inspect
import importlib
from typing import Optional, Any

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
