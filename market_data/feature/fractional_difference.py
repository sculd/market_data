import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

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


def _get_weights_ffd(params: FFDParams = None):
    params = params or FFDParams()
    w = [1.]
    k = 1
    while True:
        w_ = -w[-1] * (params.d - k + 1) / k
        if abs(w_) < params.threshold:
            break
        w.append(w_)
        k += 1
    w = np.array(w[::-1])
    return w

def get_ffd_series(series: pd.Series, params: FFDParams = None):
    params = params or FFDParams()
    w = _get_weights_ffd(params)
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

def get_zscored_ffd_series(series: pd.Series, zscored_params: ZscoredFFDParams = None):
    zscored_params = zscored_params or ZscoredFFDParams()
    ffd = get_ffd_series(series, params=zscored_params.ffd_params)
    ffd_z = (ffd - ffd.rolling(zscored_params.zscore_window).mean()) / ffd.rolling(zscored_params.zscore_window).std()
    return ffd_z.dropna()


