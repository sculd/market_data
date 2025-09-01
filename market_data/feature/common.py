import abc
from typing import Optional
from market_data.feature.param import FeatureParam

import pandas as pd


class Feature(metaclass=abc.ABCMeta):
    """Base feature class."""
    
    @abc.abstractmethod
    @staticmethod
    def calculate(df: pd.DataFrame, params: Optional[FeatureParam] = None) -> pd.DataFrame:
        """
        Calculate feature.
            
        Returns:
            DataFrame with calculated feature.
        """
        pass
