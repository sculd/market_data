import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from pathlib import Path

from feature.target import get_target_columns

logger = logging.getLogger(__name__)

def _export_datasets_by_label(
    combined_df: pd.DataFrame,
    export_dir: Union[str, Path],
    file_prefix: str = "dataset",
    file_format: str = "parquet",
    forward_return_only: bool = False,
    ts_labels: bool = False
) -> Dict[str, str]:
    """
    Export separate datasets for each target label in the combined DataFrame.
    
    This function:
    1. Identifies all target columns (with 'label_' prefix)
    2. For each label, creates a dataset with all features and that specific label
    3. Exports each dataset to a separate file
    
    Args:
        combined_df: DataFrame with features and target labels (from create_features_with_targets)
        export_dir: Directory where datasets will be saved
        file_prefix: Prefix for output filenames
        file_format: Format for export ('parquet' (default), 'csv', 'feather')
        forward_return_only: If True, only export datasets for forward return labels
        ts_labels: If True, add timestamp to filenames to avoid overwriting
    
    Returns:
        Dictionary mapping label names to exported file paths
    """
    # Ensure the export directory exists
    export_path = Path(export_dir)
    os.makedirs(export_path, exist_ok=True)
    
    # Identify feature columns (those without 'label_' prefix)
    feature_cols = [col for col in combined_df.columns if not col.startswith('label_') and col != 'symbol']
    
    # Identify all label columns
    label_cols = [col for col in combined_df.columns if col.startswith('label_')]
    
    if forward_return_only:
        label_cols = [col for col in label_cols if 'forward_return' in col]
    
    # Initialize dictionary to track exported files
    exported_files = {}
    
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") if ts_labels else ""
    
    # Create and export a dataset for each label
    for label in label_cols:
        # Create a dataset with all features and this specific label, excluding symbol
        columns_to_keep = feature_cols + [label]
        dataset = combined_df.reset_index()[['timestamp'] + columns_to_keep].copy()
        
        # Remove rows where the label is NaN
        dataset = dataset.dropna(subset=[label])
        
        # Create a sanitized filename
        sanitized_label = label.replace('label_', '').replace('/', '_')
        filename = f"{file_prefix}_{sanitized_label}"
        if timestamp_str:
            filename = f"{filename}_{timestamp_str}"
        
        # Export the dataset
        filepath = export_path / f"{filename}.{file_format}"
        
        if file_format == 'csv':
            dataset.to_csv(filepath, index=False)
        elif file_format == 'parquet':
            dataset.to_parquet(filepath, index=False)
        elif file_format == 'feather':
            dataset.to_feather(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        exported_files[label] = str(filepath)
        logger.info(f"Exported dataset for {label} to {filepath}")
    
    return exported_files

def _export_sequence_datasets_by_label(
    seq_df: pd.DataFrame,
    export_dir: Union[str, Path],
    file_prefix: str = "seq_dataset",
    file_format: str = "pickle",
    forward_return_only: bool = False,
    ts_labels: bool = False
) -> Dict[str, str]:
    """
    Export separate datasets for each target label in the sequence DataFrame.
    
    This function handles sequence feature arrays which can't be stored in flat file formats.
    
    Args:
        seq_df: DataFrame with sequence features and target labels
                (from create_sequence_features_with_targets)
        export_dir: Directory where datasets will be saved
        file_prefix: Prefix for output filenames
        file_format: Format for export ('pickle' recommended for sequence data)
        forward_return_only: If True, only export datasets for forward return labels
        ts_labels: If True, add timestamp to filenames to avoid overwriting
    
    Returns:
        Dictionary mapping label names to exported file paths
    """
    # Ensure the export directory exists
    export_path = Path(export_dir)
    os.makedirs(export_path, exist_ok=True)
    
    # Identify sequence feature columns (those containing numpy arrays)
    seq_feature_cols = []
    for col in seq_df.columns:
        # Check if first non-NaN value is a numpy array
        if col.startswith('label_') or col == 'symbol':
            continue
            
        values = seq_df[col].dropna()
        if len(values) > 0 and isinstance(values.iloc[0], (np.ndarray, list)):
            seq_feature_cols.append(col)
    
    # Identify metadata columns (not arrays and not labels, excluding symbol)
    metadata_cols = [col for col in seq_df.columns 
                    if (col not in seq_feature_cols) and 
                       not col.startswith('label_') and 
                       col != 'symbol']
    
    # Identify all label columns
    label_cols = [col for col in seq_df.columns if col.startswith('label_')]
    
    if forward_return_only:
        label_cols = [col for col in label_cols if 'forward_return' in col]
    
    # Initialize dictionary to track exported files
    exported_files = {}
    
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") if ts_labels else ""
    
    # Create and export a dataset for each label
    for label in label_cols:
        # Create a dataset with all sequence features and this specific label, excluding symbol
        columns_to_keep = seq_feature_cols + metadata_cols + [label]
        dataset = seq_df.reset_index()[['timestamp'] + columns_to_keep].copy()
        
        # Remove rows where the label is NaN
        dataset = dataset.dropna(subset=[label])
        
        # Create a sanitized filename
        sanitized_label = label.replace('label_', '').replace('/', '_')
        filename = f"{file_prefix}_{sanitized_label}"
        if timestamp_str:
            filename = f"{filename}_{timestamp_str}"
        
        # Export the dataset - sequence data usually requires pickle format
        filepath = export_path / f"{filename}.{file_format}"
        
        if file_format == 'pickle':
            dataset.to_pickle(filepath)
        else:
            logger.warning(f"Format {file_format} may not preserve numpy arrays; pickle recommended")
            if file_format == 'parquet':
                dataset.to_parquet(filepath, index=False)
            elif file_format == 'csv':
                # Warning: This will likely fail or corrupt sequence arrays
                dataset.to_csv(filepath, index=False)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        
        exported_files[label] = str(filepath)
        logger.info(f"Exported sequence dataset for {label} to {filepath}")
    
    return exported_files

def export_resampled_datasets(
    df: pd.DataFrame,
    export_dir: Union[str, Path],
    price_col: str = 'close',
    threshold: float = 0.05,
    forward_periods: List[int] = [2, 10],
    tp_values: List[float] = [0.03],
    sl_values: List[float] = [0.03],
    return_periods: List[int] = [1, 5, 15, 30, 60, 120],
    ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
    add_btc_features: bool = True,
    forward_return_only: bool = False,
    file_format: str = 'parquet',
    ts_labels: bool = True
) -> Dict[str, str]:
    """
    Create and export resampled datasets split by label.
    
    This function:
    1. Creates a resampled dataset using market events
    2. Splits it by label and exports each split
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        export_dir: Directory where datasets will be saved
        price_col: Column to use for detecting significant price movements
        threshold: Threshold for significant price movement (as decimal)
        forward_periods: Periods (in minutes) for calculating forward returns
        tp_values: Take-profit threshold values (as decimal)
        sl_values: Stop-loss threshold values (as decimal)
        return_periods: Periods (in minutes) for calculating returns
        ema_periods: Periods (in minutes) for calculating EMAs
        add_btc_features: Whether to add BTC-related features
        forward_return_only: If True, only export datasets for forward return labels
        file_format: Format for export ('parquet' (default), 'csv', 'feather')
        ts_labels: If True, add timestamp to filenames
        
    Returns:
        Dictionary mapping label names to exported file paths
    """
    from machine_learning.resample import create_resampled_dataset
    
    # Generate resampled dataset
    resampled_df = create_resampled_dataset(
        df,
        price_col=price_col,
        threshold=threshold,
        forward_periods=forward_periods,
        tp_values=tp_values,
        sl_values=sl_values,
        return_periods=return_periods,
        ema_periods=ema_periods,
        add_btc_features=add_btc_features
    )
    
    if resampled_df.empty:
        logger.warning("No events found matching the threshold criteria. No datasets exported.")
        return {}
    
    # Export the datasets by label
    return _export_datasets_by_label(
        resampled_df,
        export_dir,
        file_prefix=f"resampled_{threshold}",
        file_format=file_format,
        forward_return_only=forward_return_only,
        ts_labels=ts_labels
    )

def export_resampled_sequence_datasets(
    df: pd.DataFrame,
    export_dir: Union[str, Path],
    price_col: str = 'close',
    threshold: float = 0.05,
    sequence_length: int = 60,
    forward_periods: List[int] = [2, 10],
    tp_values: List[float] = [0.03],
    sl_values: List[float] = [0.03],
    return_periods: List[int] = [1, 5, 15, 30, 60, 120],
    ema_periods: List[int] = [5, 15, 30, 60, 120, 240],
    add_btc_features: bool = True,
    forward_return_only: bool = False,
    file_format: str = 'pickle',
    ts_labels: bool = True
) -> Dict[str, str]:
    """
    Create and export resampled sequence datasets split by label.
    
    This function:
    1. Creates a resampled sequence dataset using market events
    2. Splits it by label and exports each split
    
    Args:
        df: DataFrame with timestamp index and OHLCV columns
        export_dir: Directory where datasets will be saved
        price_col: Column to use for detecting significant price movements
        threshold: Threshold for significant price movement (as decimal)
        sequence_length: Length of the sequence arrays
        forward_periods: Periods (in minutes) for calculating forward returns
        tp_values: Take-profit threshold values (as decimal)
        sl_values: Stop-loss threshold values (as decimal)
        return_periods: Periods (in minutes) for calculating returns
        ema_periods: Periods (in minutes) for calculating EMAs
        add_btc_features: Whether to add BTC-related features
        forward_return_only: If True, only export datasets for forward return labels
        file_format: Format for export ('pickle' recommended for sequence data)
        ts_labels: If True, add timestamp to filenames
        
    Returns:
        Dictionary mapping label names to exported file paths
    """
    from machine_learning.resample import create_resampled_seq_dataset
    
    # Generate resampled sequence dataset
    resampled_seq_df = create_resampled_seq_dataset(
        df,
        price_col=price_col,
        threshold=threshold,
        sequence_length=sequence_length,
        forward_periods=forward_periods,
        tp_values=tp_values,
        sl_values=sl_values,
        return_periods=return_periods,
        ema_periods=ema_periods,
        add_btc_features=add_btc_features
    )
    
    if resampled_seq_df.empty:
        logger.warning("No events found matching the threshold criteria. No sequence datasets exported.")
        return {}
    
    # Export the sequence datasets by label
    return _export_sequence_datasets_by_label(
        resampled_seq_df,
        export_dir,
        file_prefix=f"resampled_seq_{threshold}",
        file_format=file_format,
        forward_return_only=forward_return_only,
        ts_labels=ts_labels
    )
