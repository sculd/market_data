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

def list_registered_features(security_type: str = "all") -> List[str]:
    """
    Get a list of all registered feature labels.
    
    Returns:
        List of registered feature labels
    """
    features = list(_FEATURE_REGISTRY.keys())
    if security_type == "forex":
        features = [f for f in features if not f.startswith("btc_")]
    return features

# Column pattern to feature mapping
# Maps regex patterns or prefixes to their respective feature modules
_COLUMN_TO_FEATURE_MAP = {
    # Returns feature columns (return_X where X is a period)
    "return_\\d+": "returns",
    
    # Volatility feature columns (volatility_X where X is a window)
    "volatility_\\d+": "volatility",
    
    # Bollinger Bands feature columns
    "bb_upper": "bollinger",
    "bb_middle": "bollinger",
    "bb_lower": "bollinger",
    "bb_position": "bollinger",
    "bb_width": "bollinger",
    
    # EMA feature columns
    "ema_\\d+": "ema",
    "ema_rel_\\d+": "ema",
    
    # Technical Indicators feature columns
    "rsi": "indicators",
    "open_close_ratio": "indicators",
    "autocorr_lag1": "indicators",
    "hl_range_pct": "indicators",
    "close_zscore": "indicators",
    "close_minmax": "indicators",
    
    # Volume feature columns
    "obv_.+": "volume",
    "volume_ratio_.+": "volume",
    
    # Market Regime feature columns
    "return_mean": "market_regime",
    "return_variance": "market_regime",
    "return_volatility": "market_regime",
    "return_skewness": "market_regime",
    "return_excess_kurtosis": "market_regime",
    "volatility_regime_\\d+": "market_regime",
    "volatility_zscore_\\d+": "market_regime",
    
    # GARCH feature columns
    "garch_volatility": "garch",
    
    # BTC feature columns
    "btc_return_\\d+": "btc_features",

    "ffd_zscore_.+": "ffd_zscore",
    "ffd_volatility_zscore_.+": "ffd_volatility_zscore",
}

def find_features_for_columns(column_names: List[str]) -> Dict[str, List[str]]:
    """
    Returns:
        Dictionary mapping feature labels to lists of columns they can generate
        from the requested column list
    """
    import re
    from collections import defaultdict
    result = defaultdict(list)
    
    # For each column, find the matching feature
    for col in column_names:
        feature_label = None
        
        # First check exact matches
        if col in _COLUMN_TO_FEATURE_MAP:
            feature_label = _COLUMN_TO_FEATURE_MAP[col]
        else:
            # Then check regex patterns
            for pattern, label in _COLUMN_TO_FEATURE_MAP.items():
                # Skip exact match patterns we've already checked
                if pattern in column_names:
                    continue
                    
                # Try to match the pattern as a regex
                try:
                    if re.fullmatch(pattern, col):
                        feature_label = label
                        break
                except re.error:
                    # If it's not a valid regex, skip it
                    pass
        
        if feature_label:
            # Add column to the feature's list - no need to check if key exists
            result[feature_label].append(col)
        else:
            # Log a warning when a column doesn't match any feature
            logger.warning(f"No feature found for column '{col}'. This column will not be generated.")
    
    # Convert defaultdict to regular dict before returning
    return dict(result)

def find_feature_params_for_columns(column_names: List[str]) -> List[Tuple[str, Any]]:
    """
    Find feature labels and parameters needed for the specified column names.
    
    Args:
        column_names: List of column names to look for
        
    Returns:
        List of tuples (feature_label, params) where params is the parameter object
        for the feature. You can use these tuples directly with feature.calculate(df, params).
    """
    from market_data.feature.util import _create_default_params
    import re

    # First get the mapping from features to columns
    feature_map = find_features_for_columns(column_names)
    
    # Create the result list with feature labels and their params
    result = []
    for feature_label, columns in feature_map.items():
        # Create default parameters for the feature
        params = _create_default_params(feature_label)
        
        if params is None:
            logger.warning(f"Could not create parameters for feature '{feature_label}', skipping")
            continue
        
        # Check if we need to customize parameters or if defaults are sufficient
        need_customization = False
        
        # Determine if customization is needed based on feature type and columns
        if feature_label == "returns" and hasattr(params, 'periods'):
            # Check if all column periods are covered by default periods
            col_periods = set()
            for col in columns:
                match = re.match(r'return_(\d+)', col)
                if match:
                    col_periods.add(int(match.group(1)))
            
            # If there are column periods not in default params, customize
            default_periods = set(params.periods)
            if not col_periods.issubset(default_periods):
                need_customization = True
                params.periods = sorted(list(col_periods))
        
        elif feature_label == "volatility" and hasattr(params, 'windows'):
            # Check if all column windows are covered by default windows
            col_windows = set()
            for col in columns:
                match = re.match(r'volatility_(\d+)', col)
                if match:
                    col_windows.add(int(match.group(1)))
            
            # If there are column windows not in default params, customize
            default_windows = set(params.windows)
            if not col_windows.issubset(default_windows):
                need_customization = True
                params.windows = sorted(list(col_windows))
        
        elif feature_label == "ema" and hasattr(params, 'periods'):
            # Check if all column periods are covered by default periods
            # and if price relatives are needed
            col_periods = set()
            need_price_relatives = False
            for col in columns:
                if col.startswith('ema_rel_'):
                    match = re.match(r'ema_rel_(\d+)', col)
                    if match:
                        col_periods.add(int(match.group(1)))
                        need_price_relatives = True
                else:
                    match = re.match(r'ema_(\d+)', col)
                    if match:
                        col_periods.add(int(match.group(1)))
            
            # If there are column periods not in default params, customize
            default_periods = set(params.periods)
            default_price_relatives = getattr(params, 'include_price_relatives', False)
            
            if not col_periods.issubset(default_periods) or (need_price_relatives and not default_price_relatives):
                need_customization = True
                params.periods = sorted(list(col_periods))
                if hasattr(params, 'include_price_relatives'):
                    params.include_price_relatives = need_price_relatives
        
        elif feature_label == "volume" and hasattr(params, 'ratio_periods'):
            # Check if all column periods are covered by default ratio periods
            col_periods = set()
            for col in columns:
                match = re.match(r'volume_ratio_(\d+)', col)
                if match:
                    col_periods.add(int(match.group(1)))
            
            # If there are column periods not in default params, customize
            default_periods = set(params.ratio_periods)
            if not col_periods.issubset(default_periods):
                need_customization = True
                params.ratio_periods = sorted(list(col_periods))
        
        elif feature_label == "market_regime" and hasattr(params, 'volatility_windows'):
            # Check if all column windows are covered by default volatility windows
            col_windows = set()
            for col in columns:
                match = re.match(r'volatility_(?:regime|zscore)_(\d+)', col)
                if match:
                    col_windows.add(int(match.group(1)))
            
            # If there are column windows not in default params, customize
            default_windows = set(params.volatility_windows)
            if not col_windows.issubset(default_windows):
                need_customization = True
                params.volatility_windows = sorted(list(col_windows))
        
        elif feature_label == "btc_features" and hasattr(params, 'return_periods'):
            # Check if all column periods are covered by default return periods
            col_periods = set()
            for col in columns:
                match = re.match(r'btc_return_(\d+)', col)
                if match:
                    col_periods.add(int(match.group(1)))
            
            # If there are column periods not in default params, customize
            default_periods = set(params.return_periods)
            if not col_periods.issubset(default_periods):
                need_customization = True
                params.return_periods = sorted(list(col_periods))
                
        # Log whether we're using default or customized parameters
        if need_customization:
            logger.warning(f"Customizing parameters for feature '{feature_label}' based on column names")
        else:
            logger.debug(f"Using default parameters for feature '{feature_label}'")
        
        # Add the (feature_label, params) tuple to the result
        result.append((feature_label, params))
    
    return result 