"""
Configuration management for the CLI
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class CLIConfig:
    """CLI configuration settings"""
    
    # Default values for common parameters
    default_dataset_mode: str = 'OKX'
    default_export_mode: str = 'BY_MINUTE'
    default_aggregation_mode: str = 'TAKE_LATEST'
    default_workers: Optional[int] = None
    
    # Cache settings
    cache_base_path: Optional[str] = None
    default_overwrite_cache: bool = False
    
    # Feature settings
    default_calculation_batch_days: int = 1
    default_warmup_days: Optional[int] = None
    
    # Resample settings  
    default_resample_type: str = 'cumsum'
    default_resample_params: Optional[str] = None
    
    # ML settings
    default_features: str = 'all'
    default_sequence_window: int = 30
    
    @classmethod
    def from_file(cls, path: Path) -> 'CLIConfig':
        """Load configuration from a YAML file"""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> 'CLIConfig':
        """Load configuration from environment variables"""
        config_data = {}
        
        # Map environment variables to config fields
        env_mapping = {
            'MARKET_DATA_DATASET_MODE': 'default_dataset_mode',
            'MARKET_DATA_EXPORT_MODE': 'default_export_mode', 
            'MARKET_DATA_AGGREGATION_MODE': 'default_aggregation_mode',
            'MARKET_DATA_WORKERS': 'default_workers',
            'MARKET_DATA_CACHE_BASE_PATH': 'cache_base_path',
            'MARKET_DATA_OVERWRITE_CACHE': 'default_overwrite_cache',
            'MARKET_DATA_BATCH_DAYS': 'default_calculation_batch_days',
            'MARKET_DATA_WARMUP_DAYS': 'default_warmup_days',
            'MARKET_DATA_RESAMPLE_TYPE': 'default_resample_type',
            'MARKET_DATA_RESAMPLE_PARAMS': 'default_resample_params',
            'MARKET_DATA_FEATURES': 'default_features',
            'MARKET_DATA_SEQUENCE_WINDOW': 'default_sequence_window',
        }
        
        for env_var, config_field in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion
                if config_field in ['default_workers', 'default_calculation_batch_days', 
                                   'default_warmup_days', 'default_sequence_window']:
                    try:
                        config_data[config_field] = int(value)
                    except ValueError:
                        pass
                elif config_field == 'default_overwrite_cache':
                    config_data[config_field] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    config_data[config_field] = value
        
        return cls(**config_data)
    
    @classmethod
    def default(cls) -> 'CLIConfig':
        """Create default configuration"""
        return cls()
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'CLIConfig':
        """
        Load configuration with the following priority:
        1. Specified config file (if provided)
        2. Environment variables
        3. Default values
        """
        # Start with default configuration
        config = cls.default()
        
        # Override with environment variables
        try:
            env_config = cls.from_env()
            config = cls.merge_configs(config, env_config)
        except Exception:
            pass  # Ignore env loading errors
        
        # Override with config file if specified
        if config_path:
            try:
                file_config = cls.from_file(Path(config_path))
                config = cls.merge_configs(config, file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_path}: {e}")
        
        return config
    
    @staticmethod
    def merge_configs(base: 'CLIConfig', override: 'CLIConfig') -> 'CLIConfig':
        """Merge two configurations, with override taking precedence"""
        base_dict = base.__dict__.copy()
        override_dict = override.__dict__.copy()
        
        # Only override non-None values
        for key, value in override_dict.items():
            if value is not None:
                base_dict[key] = value
        
        return CLIConfig(**base_dict)
    
    def to_yaml(self, path: Path):
        """Save configuration to a YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, sort_keys=True)
    
    def create_example_config(self, path: Path):
        """Create an example configuration file with comments"""
        example_content = """# Market Data CLI Configuration
# This file contains default settings for the CLI tool

# Default data source settings
default_dataset_mode: OKX              # Options: OKX, FOREX_IBKR, etc.
default_export_mode: BY_MINUTE         # Options: BY_MINUTE, RAW, etc.
default_aggregation_mode: TAKE_LATEST # Options: TAKE_LATEST, COLLECT_ALL_UPDATES, etc.

# Processing settings
default_workers: null                  # Number of parallel workers (null = auto-detect)
default_calculation_batch_days: 1      # Batch size for processing
default_warmup_days: null             # Warmup period (null = auto-detect)

# Cache settings
cache_base_path: null                  # Base path for cache (null = use default)
default_overwrite_cache: false        # Whether to overwrite existing cache by default

# Resampling settings
default_resample_type: cumsum          # Default resampling method
default_resample_params: null         # Default resample parameters

# ML data settings
default_features: all                  # Default features to process
default_sequence_window: 30           # Default sequence window for sequential features
"""
        with open(path, 'w') as f:
            f.write(example_content)


def load_config(config_path: Optional[str] = None) -> CLIConfig:
    """Convenience function to load configuration"""
    return CLIConfig.load(config_path)