"""
Sequential Feature

This module provides functions for sequentializing features,
allowing for sequential features to be created from non-sequential features.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from numba import jit

from market_data.feature.param import SequentialFeatureParam

logger = logging.getLogger(__name__)

@jit(cache=True, nopython=True)
def create_sequences_numba(values: np.ndarray, sequence_window: int) -> np.ndarray:
    """
    Create sequences using Numba-accelerated code for a single column.
    
    Args:
        values: 1D array of feature values
        sequence_window: Size of the sequence window
        
    Returns:
        2D array of sequences
    """
    n_sequences = len(values) - sequence_window + 1
    sequences = np.empty((n_sequences, sequence_window), dtype=np.float64)
    
    for i in range(n_sequences):
        sequences[i] = values[i:i + sequence_window]
        
    return sequences

@jit(cache=True, nopython=True)
def create_sequences_multi_numba(values: np.ndarray, sequence_window: int) -> np.ndarray:
    """
    Create sequences using Numba-accelerated code for multiple columns at once.
    
    Args:
        values: 2D array of feature values with shape (n_samples, n_features)
        sequence_window: Size of the sequence window
        
    Returns:
        3D array of sequences with shape (n_sequences, n_features, sequence_window)
    """
    n_samples, n_features = values.shape
    n_sequences = n_samples - sequence_window + 1
    
    # Initialize output array with shape (n_sequences, n_features, sequence_window)
    sequences = np.empty((n_sequences, n_features, sequence_window), dtype=np.float64)
    
    # For each sequence position
    for i in range(n_sequences):
        # For each feature
        for j in range(n_features):
            # Extract the sequence window for this feature
            sequences[i, j] = values[i:i + sequence_window, j]
        
    return sequences

def sequentialize_feature(
    df: pd.DataFrame,
    seq_params: SequentialFeatureParam = SequentialFeatureParam(),
) -> Optional[pd.DataFrame]:
    """
    Create sequential features from a regular feature DataFrame.
    
    Args:
        df: DataFrame containing regular features
        seq_params: Sequential feature parameters
        
    Returns:
        DataFrame with sequential features or None if error occurs
    """
    try:  
        # Sort by timestamp once at the beginning
        df = df.sort_index(level='timestamp')
        
        # Get feature columns (excluding index)
        feature_cols = [c for c in df.columns.tolist() if c != 'symbol']
        
        # Initialize arrays to store results
        all_sequences = []
        all_timestamps = []
        all_symbols = []
        
        # Group by symbol for efficient processing
        if 'symbol' in df.index.names:
            groupby = df.groupby(level='symbol')
        elif 'symbol' in df.columns:
            groupby = df.groupby('symbol')
        else:
            raise ValueError("No symbol column found in DataFrame")
        
        for symbol, symbol_data in groupby:
            # Get number of sequences for this symbol
            n_sequences = len(symbol_data) - seq_params.sequence_window + 1
            if n_sequences <= 0:
                continue
                
            # Get timestamp values once
            symbol_timestamps = symbol_data.index.get_level_values('timestamp')
            
            # Convert all features to a 2D numpy array (samples × features)
            feature_values = symbol_data[feature_cols].values
            
            # Create sequences for all features at once using the multi-column version
            # Result shape: (n_sequences, n_features, sequence_window)
            symbol_sequences = create_sequences_multi_numba(feature_values, seq_params.sequence_window)
            
            # Store results
            all_sequences.append(symbol_sequences)
            all_timestamps.extend(symbol_timestamps[-n_sequences:])
            all_symbols.extend([symbol] * n_sequences)
        
        if not all_sequences:
            logger.warning("No sequences could be created")
            return None
            
        # Combine all sequences
        all_sequences = np.vstack(all_sequences)
        
        # Create DataFrame
        data = {}
        for col_idx, col in enumerate(feature_cols):
            data[col] = all_sequences[:, col_idx].tolist()
            
        data['timestamp'] = all_timestamps
        data['symbol'] = all_symbols
        
        seq_df = pd.DataFrame(data)
        seq_df = seq_df.set_index(['timestamp', 'symbol'])
        
        return seq_df
        
    except Exception as e:
        logger.error(f"Error sequentializing feature: {e}")
        return None
        
