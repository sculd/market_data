"""
Feature engineering module for market data analysis.
"""

# Import and expose key functions from feature module
from feature.feature import (
    FeatureParams,
    create_features,
    create_sequence_features
)

# Import and expose key functions and variables from target module
from feature.target import (
    TargetParams,
    create_targets,
    get_target_columns,
)

# Define what gets imported with "from feature import *"
__all__ = [
    # Feature engineering
    'FeatureParams',
    'create_features',
    'create_sequence_features',
    
    # Target engineering
    'TargetParams',
    'create_targets',
    'get_target_columns',
]
