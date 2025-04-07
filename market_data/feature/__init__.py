"""
Feature package for market data.

This package provides modules for calculating and managing market features.
"""

from market_data.feature.registry import (
    register_feature,
    get_feature_by_label,
    list_registered_features
)

from market_data.feature.cache_reader import (
    read_multi_feature_cache
)

from market_data.feature.cache_writer import (
    cache_feature_cache
)

# Import implementations package to ensure all features are registered
import market_data.feature.impl

__all__ = [
    'register_feature',
    'get_feature_by_label',
    'list_registered_features',
    'read_multi_feature_cache',
    'cache_feature_cache'
]
