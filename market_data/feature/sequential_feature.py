"""
Sequential Feature

This module provides functions for sequentializing features,
allowing for sequential features to be created from non-sequential features.
"""

import pandas as pd
import logging
from typing import Optional, Dict, Any
import numpy as np
from numba import jit

from market_data.feature.impl.common import SequentialFeatureParam

logger = logging.getLogger(__name__)

@jit(cache=True, nopython=True)
def create_sequences_numba(values: np.ndarray, sequence_window: int) -> np.ndarray:
    """
    Create sequences using Numba-accelerated code.
    
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
        feature_cols = df.columns.tolist()
        
        # Initialize arrays to store results
        all_sequences = []
        all_timestamps = []
        all_symbols = []
        
        # Group by symbol for efficient processing
        for symbol, symbol_data in df.groupby(level='symbol'):
            # Get number of sequences for this symbol
            n_sequences = len(symbol_data) - seq_params.sequence_window + 1
            if n_sequences <= 0:
                continue
                
            # Get timestamp values once
            symbol_timestamps = symbol_data.index.get_level_values('timestamp')
                
            # Pre-allocate arrays for this symbol's sequences
            symbol_sequences = np.empty((n_sequences, len(feature_cols), seq_params.sequence_window), dtype=np.float64)
            
            # Create sequences for each feature using Numba
            for col_idx, col in enumerate(feature_cols):
                values = symbol_data[col].values
                sequences = create_sequences_numba(values, seq_params.sequence_window)
                symbol_sequences[:, col_idx] = sequences
            
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
        
