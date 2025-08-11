import abc


class Param(abc.ABC):
    """
    Generic base class for all parameter objects.
    
    This class provides the fundamental contract for parameter serialization
    and caching directory generation.
    """
    
    @abc.abstractmethod
    def get_params_dir(self) -> str:
        """
        Generate a directory name string from parameters for caching purposes.
        
        Returns:
            str: Directory name string that uniquely identifies these parameters
        """
        pass
    
    @classmethod
    @abc.abstractmethod
    def from_str(cls, param_str: str):
        """Parse parameters from string representation"""
        pass

    @abc.abstractmethod
    def to_str(self) -> str:
        """Convert parameters to string representation"""
        pass

