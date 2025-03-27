"""
Machine learning utilities for market data analysis.
"""

# Import and expose key functions from data module
from machine_learning.feature_target import (
    create_features_with_targets,
    create_sequence_features_with_targets
)
# Import and expose key functions from resample module
from machine_learning.resample import (
    get_events_t,
    get_events_t_multi,
    create_resampled_dataset,
    create_resampled_seq_dataset
)
# Import and expose key functions from data export module
from machine_learning.data import (
    export_resampled_datasets,
    export_resampled_sequence_datasets
)

# Define what gets imported with "from machine_learning import *"
__all__ = [
    # Data preparation
    'create_features_with_targets',
    'create_sequence_features_with_targets',
    'get_events_t',
    'get_events_t_multi',
    'create_resampled_dataset',
    'create_resampled_seq_dataset',
    # Data export (resampled only)
    'export_resampled_datasets',
    'export_resampled_sequence_datasets',
]
