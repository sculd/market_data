"""
Feature implementations package.

This package contains specific implementations of market features.
All modules are imported here to ensure they are registered with the feature registry.
"""

# Import all feature implementations to ensure they are registered
from market_data.feature.impl import returns
from market_data.feature.impl import volatility
from market_data.feature.impl import bollinger
from market_data.feature.impl import indicators
from market_data.feature.impl import ema
from market_data.feature.impl import volume
from market_data.feature.impl import market_regime
from market_data.feature.impl import btc_features
from market_data.feature.impl import garch

__all__ = [
    'returns',
    'volatility',
    'bollinger',
    'indicators',
    'ema',
    'volume',
    'market_regime',
    'btc_features',
    'garch'
] 