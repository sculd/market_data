from dataclasses import dataclass, field
from typing import List

from market_data.util.param import Param

# Default values for target parameters
DEFAULT_FORWARD_PERIODS = [5, 10, 30]
DEFAULT_TP_VALUES = [0.015, 0.03, 0.05]

@dataclass
class TargetParams(Param):
    forward_period: int = 30
    tp_value: float = 0.03
    sl_value: float = 0.03
    
    @classmethod
    def from_str(cls, param_str: str) -> 'TargetParams':
        """Parse parameters from string format: forward_period:30,tp_value:0.03,sl_value:0.03"""
        params = {}
        for pair in param_str.split(','):
            if ':' in pair:
                key, value = pair.split(':', 1)
                if key == 'forward_period':
                    params['forward_period'] = int(value)
                elif key == 'tp_value':
                    params['tp_value'] = float(value)
                elif key == 'sl_value':
                    params['sl_value'] = float(value)
        return cls(**params)
    
    def to_str(self) -> str:
        """Convert parameters to string format: forward_period:30,tp_value:0.03,sl_value:0.03"""
        return f"forward_period:{self.forward_period},tp_value:{self.tp_value},sl_value:{self.sl_value}"
    
    def __repr__(self) -> str:
        """String representation of the parameters."""
        return (
            f"TargetParams(forward_period={self.forward_period}, "
            f"tp_value={self.tp_value}, "
            f"sl_value={self.sl_value})"
        )

def _get_default_target_params_list() -> List[TargetParams]:
    return [TargetParams(forward_period=period, tp_value=tp, sl_value=tp) 
            for period in DEFAULT_FORWARD_PERIODS 
            for tp in DEFAULT_TP_VALUES]

@dataclass
class TargetParamsBatch(Param):
    """
    Encapsulates parameters for target engineering.
    
    This class holds all the parameters needed for target calculation,
    providing a single source of truth for target configuration.
    """
    target_params_list: List[TargetParams] = field(default_factory=lambda: _get_default_target_params_list())

    @classmethod
    def from_str(cls, param_str: str) -> 'TargetParamsBatch':
        """Parse parameters from string format: fp:5,10,30|tp:0.015,0.03,0.05|sl:0.015,0.03,0.05"""
        if not param_str:
            return cls()
        
        params_dict = {}
        for part in param_str.split('|'):
            if ':' in part:
                key, values = part.split(':', 1)
                if key == 'fp':
                    params_dict['forward_periods'] = [int(v) for v in values.split(',')]
                elif key == 'tp':
                    params_dict['tp_values'] = [float(v) for v in values.split(',')]
                elif key == 'sl':
                    params_dict['sl_values'] = [float(v) for v in values.split(',')]
        
        # Create target params list from the parsed values
        forward_periods = params_dict.get('forward_periods', DEFAULT_FORWARD_PERIODS)
        tp_values = params_dict.get('tp_values', DEFAULT_TP_VALUES)
        sl_values = params_dict.get('sl_values', DEFAULT_TP_VALUES)
        
        target_params_list = [
            TargetParams(forward_period=fp, tp_value=tp, sl_value=sl)
            for fp in forward_periods
            for tp in tp_values
            for sl in sl_values
        ]
        
        return cls(target_params_list=target_params_list)
    
    def to_str(self) -> str:
        """Convert parameters to string format: fp:5,10,30|tp:0.015,0.03,0.05|sl:0.015,0.03,0.05"""
        # Extract unique values from the list
        forward_periods = sorted(set(p.forward_period for p in self.target_params_list))
        tp_values = sorted(set(p.tp_value for p in self.target_params_list))
        sl_values = sorted(set(p.sl_value for p in self.target_params_list))
        
        parts = []
        if forward_periods:
            parts.append(f"fp:{','.join(map(str, forward_periods))}")
        if tp_values:
            parts.append(f"tp:{','.join(map(str, tp_values))}")
        if sl_values:
            parts.append(f"sl:{','.join(map(str, sl_values))}")
        
        return '|'.join(parts)

    def __repr__(self) -> str:
        """String representation of the parameters."""
        return '\n'.join([f'{p.__repr__}' for p in self.target_params_list])