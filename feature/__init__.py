"""
Feature engineering module for market data analysis.
"""

# Import and expose key functions from feature module
from feature.feature import (
    FeatureEngineer,
    create_features,
    create_sequence_features
)

# Import and expose key functions and variables from target module
from feature.target import (
    TargetEngineer,
    create_targets,
    get_target_columns,
    TARGET_COLUMNS_DEFAULT,
    TARGET_COLUMNS_FULL
)

# Import and expose key functions from data module
from feature.data import (
    create_features_with_targets,
    create_sequence_features_with_targets
)

# Define what gets imported with "from feature import *"
__all__ = [
    # Feature engineering
    'FeatureEngineer',
    'create_features',
    'create_sequence_features',
    
    # Target engineering
    'TargetEngineer',
    'create_targets',
    'get_target_columns',
    'TARGET_COLUMNS_DEFAULT',
    'TARGET_COLUMNS_FULL',
    
    # Data preparation
    'create_features_with_targets',
    'create_sequence_features_with_targets'
]
