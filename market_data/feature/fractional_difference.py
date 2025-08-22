from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numba as nb
import numpy as np
import pandas as pd

_threshold_default = 1e-3
_default_zscore_window = 100

@dataclass
class FFDParams:
    # note the minimal window size for 0.4 d and 1e-3 thresh is 61+1
    d: int = 0.40
    threshold: float = _threshold_default
    
@dataclass
class ZscoredFFDParams:
    ffd_params: FFDParams = field(default_factory=FFDParams)
    zscore_window: int = _default_zscore_window
    

@nb.njit(cache=True)
def _get_weights_ffd_numba(d: float, threshold: float = _threshold_default):
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < threshold:
            break
        w.append(w_)
        k += 1
    
    # Convert to numpy array and reverse
    w_array = np.array(w)
    return w_array[::-1]

@nb.njit(cache=True)
def _get_ffd_series_numba(prices: np.ndarray, d: float, threshold: float = _threshold_default):
    n = len(prices)
    
    # Get weights
    w = _get_weights_ffd_numba(d, threshold)
    width = len(w)
    
    # Initialize output array
    output = np.full(n, np.nan)
    
    # Calculate FFD values
    for i in range(width, n):
        # Check if any values in the window are NaN
        window_has_nan = False
        for j in range(width):
            if np.isnan(prices[i - width + j]):
                window_has_nan = True
                break
        
        if not window_has_nan:
            # Calculate weighted sum
            ffd_value = 0.0
            for j in range(width):
                ffd_value += w[j] * prices[i - width + j]
            output[i] = ffd_value
    
    return output

@nb.njit(cache=True)
def _rolling_mean_numba(arr: np.ndarray, window: int):
    """
    Numba-compatible rolling mean calculation.
    """
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        # Check if we have enough non-NaN values
        valid_count = 0
        sum_val = 0.0
        
        for j in range(i - window + 1, i + 1):
            if not np.isnan(arr[j]):
                valid_count += 1
                sum_val += arr[j]
        
        if valid_count > 0:
            result[i] = sum_val / valid_count
    
    return result

@nb.njit(cache=True)
def _rolling_std_numba(arr: np.ndarray, window: int):
    """
    Numba-compatible rolling standard deviation calculation.
    """
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        # Calculate mean first
        valid_count = 0
        sum_val = 0.0
        
        for j in range(i - window + 1, i + 1):
            if not np.isnan(arr[j]):
                valid_count += 1
                sum_val += arr[j]
        
        if valid_count > 1:  # Need at least 2 points for std
            mean_val = sum_val / valid_count
            
            # Calculate variance
            sum_sq_diff = 0.0
            for j in range(i - window + 1, i + 1):
                if not np.isnan(arr[j]):
                    diff = arr[j] - mean_val
                    sum_sq_diff += diff * diff
            
            variance = sum_sq_diff / (valid_count - 1)
            result[i] = np.sqrt(variance)
    
    return result

@nb.njit(cache=True)
def _get_zscored_ffd_series_numba(prices: np.ndarray, d: float, threshold: float = _threshold_default, zscore_window: int = _default_zscore_window):
    # Get FFD series
    ffd = _get_ffd_series_numba(prices, d, threshold)
    
    # Calculate rolling mean and std
    ffd_mean = _rolling_mean_numba(ffd, zscore_window)
    ffd_std = _rolling_std_numba(ffd, zscore_window)
    
    # Calculate z-scores using vectorized operations
    # Create mask for valid calculations
    mask = (ffd_std > 0) & ~np.isnan(ffd) & ~np.isnan(ffd_mean) & ~np.isnan(ffd_std)
    
    # Vectorized z-score calculation
    zscored = np.where(mask, (ffd - ffd_mean) / ffd_std, np.nan)
    
    return zscored

def get_ffd_series(series: pd.Series, params: FFDParams = None, use_numba: bool = True):
    params = params or FFDParams()
    if use_numba:
        params = params or FFDParams()
        prices_array = series.values
        ffd_array = _get_ffd_series_numba(prices_array, params.d, params.threshold)
        return pd.Series(ffd_array, index=series.index)
    else:
        w = _get_weights_ffd_numba(params.d, params.threshold)
        width = len(w)
        output = pd.Series(dtype='float64', index=series.index)
        for i in range(width, len(series)):
            window = series.iloc[i - width:i]
            if window.isnull().any():
                continue
            output.iloc[i] = np.dot(w, window)
        return output.dropna()

def get_normalized_ffd_series(series: pd.Series, params: FFDParams = None):
    params = params or FFDParams()
    ffd = get_ffd_series(series, params=params)
    normalized_ffd = ffd / series
    return normalized_ffd.dropna()

def get_zscored_ffd_series(series: pd.Series, zscored_params: ZscoredFFDParams = None, use_numba: bool = True):
    zscored_params = zscored_params or ZscoredFFDParams()
    if use_numba:
        prices_array = series.values
        zscored_array = _get_zscored_ffd_series_numba(
            prices_array, 
            zscored_params.ffd_params.d, 
            zscored_params.ffd_params.threshold,
            zscored_params.zscore_window
        )
        return pd.Series(zscored_array, index=series.index)
    else:
        ffd = get_ffd_series(series, params=zscored_params.ffd_params)
        ffd_z = (ffd - ffd.rolling(zscored_params.zscore_window).mean()) / ffd.rolling(zscored_params.zscore_window).std()
        return ffd_z.dropna()
