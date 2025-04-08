import numpy as np
import numba as nb


@nb.njit(cache=True)
def _calculate_rolling_std_numba(values, window):
    """
    Calculate rolling standard deviation using Numba for performance.
    
    Args:
        values: Array of input values
        window: Window size for calculation
        
    Returns:
        Array of standard deviation values
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        window_values = values[i-window:i]
        valid_values = window_values[~np.isnan(window_values)]
        
        if len(valid_values) >= window // 2:  # Require at least half of the window to be valid
            result[i] = np.std(valid_values)
    
    return result

@nb.njit(cache=True)
def _calculate_rolling_mean_numba(values, window):
    """
    Calculate rolling mean using Numba for performance.
    
    Args:
        values: Array of input values
        window: Window size for calculation
        
    Returns:
        Array of mean values
    """
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        window_values = values[i-window:i]
        valid_values = window_values[~np.isnan(window_values)]
        
        if len(valid_values) >= window // 2:  # Require at least half of the window to be valid
            result[i] = np.mean(valid_values)
    
    return result
