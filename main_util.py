from market_data.machine_learning.resample import ResampleParams

def parse_resample_params(param_str):
    """
    Parse a string in the format 'price_col,threshold' into ResampleParams.
    Example: 'close,0.07' -> ResampleParams(price_col='close', threshold=0.07)
    
    Args:
        param_str: String in format 'price_col,threshold'
        
    Returns:
        ResampleParams instance
    """
    if not param_str:
        return ResampleParams()
        
    try:
        parts = param_str.split(',')
        if len(parts) != 2:
            raise ValueError("Format should be 'price_col,threshold'")
            
        price_col = parts[0].strip()
        threshold = float(parts[1].strip())
        
        return ResampleParams(price_col=price_col, threshold=threshold)
    except Exception as e:
        raise ValueError(f"Invalid resample_params format: {e}. Format should be 'price_col,threshold' (e.g. 'close,0.07')")
