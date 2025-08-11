"""
GARCH Volatility Feature Module

This module provides functions for calculating GARCH volatility.
"""

import pandas as pd
import numpy as np
import logging
import datetime
import numba as nb
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from market_data.feature.registry import register_feature
from market_data.feature.param import FeatureParam
from market_data.feature.impl.returns import _calculate_simple_returns_numba

logger = logging.getLogger(__name__)

# Feature label for registration
FEATURE_LABEL = "garch"

# Numba-accelerated GARCH volatility calculation
@nb.njit(cache=True)
def _initialize_returns_numba(returns):
    """
    Replace initial NaN values in returns with zeros.
    
    Args:
        returns: Array of return values
        
    Returns:
        Array of initialized returns
    """
    n = len(returns)
    result = returns.copy()
    
    i = 0
    while i < n and np.isnan(result[i]):
        result[i] = 0.0
        i += 1
    
    return result

@nb.njit(cache=True)
def _calculate_garch_volatility_numba(returns, omega, alpha, beta):
    """
    Calculate GARCH(1,1) volatility using Numba for performance.
    
    Args:
        returns: Array of return values (with NaNs replaced)
        omega: GARCH intercept parameter
        alpha: GARCH parameter for squared returns
        beta: GARCH parameter for lagged volatility
        
    Returns:
        Array of volatility values
    """
    n = len(returns)
    sigma2 = np.zeros(n)
    
    # Initialize with variance of non-NaN returns
    valid_returns = returns[~np.isnan(returns)]
    if len(valid_returns) > 0:
        sigma2[0] = np.var(valid_returns)
    else:
        sigma2[0] = omega  # Default value if all returns are NaN
    
    # Pre-calculate squared returns
    squared_returns = np.zeros(n)
    for i in range(n):
        if not np.isnan(returns[i]):
            squared_returns[i] = returns[i] * returns[i]
        else:
            squared_returns[i] = 0.0
    
    # Calculate GARCH volatility
    for i in range(1, n):
        if np.isnan(returns[i-1]):
            sigma2[i] = sigma2[i-1]  # Keep last value for NaN returns
        else:
            sigma2[i] = omega + alpha * squared_returns[i-1] + beta * sigma2[i-1]
    
    # Convert variance to volatility (standard deviation)
    volatility = np.sqrt(sigma2)
    
    return volatility

@nb.njit(cache=True)
def _annualize_volatility_numba(volatility, annualization_factor):
    """
    Annualize volatility using Numba.
    
    Args:
        volatility: Array of volatility values
        annualization_factor: Factor for annualization
        
    Returns:
        Array of annualized volatility values
    """
    n = len(volatility)
    result = np.zeros(n)
    
    for i in range(n):
        result[i] = volatility[i] * annualization_factor
    
    return result

@dataclass
class GARCHParams(FeatureParam):
    """Parameters for GARCH volatility calculations."""
    omega: float = 0.1
    alpha: float = 0.1
    beta: float = 0.8
    price_col: str = "close"
    annualize: bool = True
    trading_periods_per_year: int = 252 * 24 * 60  # Minutes in a trading year for crypto (24/7)
    
    def get_warm_up_period(self) -> datetime.timedelta:
        # GARCH models need warm-up data for parameter convergence
        return datetime.timedelta(days=3)
        
    def get_warm_up_days(self) -> int:
        """
        Calculate the recommended warm-up period for GARCH volatility calculation.
        
        GARCH models need a warm-up period for parameter convergence.
        
        Returns:
            int: Recommended number of warm-up days
        """
        # For GARCH models, we typically need more data for parameter convergence
        # Recommend at least 3 days of data for warm-up
        return 3
    
    def to_str(self) -> str:
        """Convert parameters to string format: omega:0.1,alpha:0.1,beta:0.8,price_col:close,annualize:true"""
        return f"omega:{self.omega},alpha:{self.alpha},beta:{self.beta},price_col:{self.price_col},annualize:{str(self.annualize).lower()},trading_periods_per_year:{self.trading_periods_per_year}"
    
    @classmethod
    def from_str(cls, feature_label_str: str) -> 'GARCHParams':
        """Parse GARCH parameters from JSON-like format: omega:0.1,alpha:0.1,beta:0.8,price_col:close,annualize:true"""
        params = {}
        for pair in feature_label_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'omega':
                    params['omega'] = float(value)
                elif key == 'alpha':
                    params['alpha'] = float(value)
                elif key == 'beta':
                    params['beta'] = float(value)
                elif key == 'price_col':
                    params['price_col'] = value
                elif key == 'annualize':
                    params['annualize'] = value.lower() == 'true'
                elif key == 'trading_periods_per_year':
                    params['trading_periods_per_year'] = int(value)
        return cls(**params)

@register_feature(FEATURE_LABEL)
class GARCHFeature:
    """GARCH volatility feature implementation."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[GARCHParams] = None) -> pd.DataFrame:
        """
        Calculate GARCH volatility.
        
        Args:
            df: Input DataFrame with OHLCV data
            params: Parameters for GARCH calculation
            
        Returns:
            DataFrame with calculated GARCH volatility
        """
        if params is None:
            params = GARCHParams()
            
        logger.info(f"Calculating GARCH volatility with omega={params.omega}, alpha={params.alpha}, beta={params.beta}")
        
        # Ensure we have the price column
        if params.price_col not in df.columns:
            raise ValueError(f"Price column '{params.price_col}' not found in DataFrame")
        
        # Check if 'symbol' column exists
        has_symbol = 'symbol' in df.columns
        if not has_symbol:
            logger.warning("DataFrame does not have a 'symbol' column, will use a single default symbol")
            df = df.copy()
            df['symbol'] = 'default'
        
        # Create list to store DataFrames for each symbol
        results = []
        
        # Process each symbol separately
        for symbol, group_df in df.groupby('symbol'):
            # Create result DataFrame for this symbol
            symbol_result = pd.DataFrame(index=group_df.index)
            
            # Add symbol column
            symbol_result['symbol'] = symbol
            
            # Extract prices as numpy array for Numba functions
            prices = group_df[params.price_col].values
            
            # Calculate returns using function from returns module
            returns = _calculate_simple_returns_numba(prices, 1)
            
            # Initialize returns (replace NaN values at beginning)
            returns_init = _initialize_returns_numba(returns)
            
            # Calculate GARCH volatility
            volatility = _calculate_garch_volatility_numba(
                returns_init, 
                params.omega, 
                params.alpha, 
                params.beta
            )
            
            # Annualize if requested
            if params.annualize:
                # Using sqrt(T) rule for annualization
                annualization_factor = np.sqrt(params.trading_periods_per_year)
                volatility = _annualize_volatility_numba(volatility, annualization_factor)
                
            symbol_result['garch_volatility'] = volatility
            
            # Add to results list
            results.append(symbol_result)
        
        # Combine all symbol results
        result = pd.concat(results).reset_index().set_index(['timestamp', 'symbol'])
        
        return result 