"""
Machine learning module for market data analysis.
"""

# Import and expose key functions from data module
from machine_learning.data import (
    create_features_with_targets,
    create_sequence_features_with_targets
)

# Define what gets imported with "from feature import *"
__all__ = [
    # Data preparation
    'create_features_with_targets',
    'create_sequence_features_with_targets'
]
