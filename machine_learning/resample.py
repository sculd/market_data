import pandas as pd

def get_events_t(df: pd.DataFrame, col: str, threshold: float = 0.05) -> pd.DataFrame:
    """
    Get time index from a DataFrame where the target column cumulatively changes by more than threshold.
    
    Args:
        df: DataFrame with timestamp index and target column
        col: Name of the target column
        threshold: Threshold for the change in target column
    """
    t_events = []
    s_pos, s_neg = 0, 0
    diff =  df[col].diff()
    for i in diff.index[1:]:
        s_pos = max(0, s_pos + diff.loc[i])
        s_neg = min(0, s_neg + diff.loc[i])
        if s_pos > threshold:
            t_events.append(i)
            s_pos = 0
        elif s_neg < -threshold:
            t_events.append(i)
            s_neg = 0
            
    return pd.DatetimeIndex(t_events)
