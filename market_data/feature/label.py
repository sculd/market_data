import dataclasses
import datetime
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from market_data.util.cache.path import get_cache_base_path

logger = logging.getLogger(__name__)


def _find_param_class(feature_label: str) -> Optional[Any]:
    try:
        module_path = f"market_data.feature.impl.{feature_label}"
        module = importlib.import_module(module_path)
        
        for name, obj in inspect.getmembers(module):
            if not inspect.isclass(obj):
                continue
            if not name.endswith('Params'):
                continue
            if obj.__module__ != module_path:
                continue
            return obj
        
        return None
    except Exception as e:
        logger.warning(f"Error finding param class for {feature_label}: {e}")
        return None


@dataclasses.dataclass
class FeatureLabel:
    param_delim = '@'

    def __init__(self, feature_label: str, params: Any = None):
        self.feature_label = feature_label
        if params is None:
            param_cls = _find_param_class(self.feature_label)
            params = param_cls()
            if params is None:
                raise ValueError(f"Failed to create default parameters for feature '{self.feature_label}'")
        self.params = params

        if not hasattr(params, 'get_params_dir'):
            raise ValueError(f"Parameters object for feature '{self.feature_label}' must have get_params_dir method")

    def __str__(self):
        return f"{self.feature_label} with {self.params}"

    @classmethod
    def from_str(cls, feature_label_str: str) -> 'FeatureLabel':
        feature_label, params_str = feature_label_str.split(cls.param_delim)
        param_cls = _find_param_class(feature_label)
        if not params_str:
            params = param_cls()
        else:
            params = param_cls.from_str(params_str)
        return cls(feature_label, params)

    def get_warmup_period(
            self,
        ) -> datetime.timedelta:
        """
        Get the maximum warm-up period required for a list of feature labels and parameters.
        """
        return max(datetime.timedelta(minutes=1), self.params.get_warm_up_period())

@dataclasses.dataclass
class FeatureLabelCollection:
    def __init__(self):
        self.feature_labels: List[FeatureLabel] = []

    def __str__(self):
        return '\n'.join([str(label) for label in self.feature_labels])

    def __iter__(self):
        return iter(self.feature_labels)

    def with_feature_label(self, feature_label: FeatureLabel):
        self.feature_labels.append(feature_label)
        return self

    @classmethod
    def from_str(cls, feature_label_collection_str: str) -> 'FeatureLabelCollection':
        ret = cls()
        for feature_label_str in feature_label_collection_str.split():
            feature_label = FeatureLabel.from_str(feature_label_str)
            ret = ret.with_feature_label(feature_label)
        return ret

    def get_warmup_period(
            self,
        ) -> datetime.timedelta:
        """
        Get the maximum warm-up period required for a list of feature labels and parameters.
        """
        return max([fl.get_warmup_period() for fl in self.feature_labels])

    def to_dict(self) -> dict:
        """Convert FeatureLabelCollection to dictionary for serialization."""
        return {
            'feature_labels': [
                {
                    'feature_label': fl.feature_label,
                    'params': self._serialize_params(fl.params)
                }
                for fl in self.feature_labels
            ]
        }
    
    def _serialize_params(self, params) -> dict:
        """Helper method to properly serialize parameters."""
        # First check if it has a custom to_dict method
        if hasattr(params, 'to_dict') and callable(getattr(params, 'to_dict')):
            return params.to_dict()
        
        # For dataclass objects, use asdict which handles nested dataclasses properly
        if hasattr(params, '__dataclass_fields__'):
            return dataclasses.asdict(params)
        
        # Fallback to __dict__ for regular objects
        if hasattr(params, '__dict__'):
            return params.__dict__
        
        # Last resort: convert to string representation
        return {'_str_repr': str(params)}

    def _reconstruct_dataclass(self, cls, data_dict):
        """Recursively reconstruct a dataclass from a dictionary."""
        if not hasattr(cls, '__dataclass_fields__'):
            return cls(**data_dict) if data_dict else cls()
        
        reconstructed_params = {}
        fields = dataclasses.fields(cls)
        
        for field in fields:
            field_name = field.name
            if field_name in data_dict:
                field_value = data_dict[field_name]
                field_type = field.type
                
                # Handle nested dataclass objects
                if hasattr(field_type, '__dataclass_fields__') and isinstance(field_value, dict):
                    # Recursively reconstruct nested dataclass
                    reconstructed_params[field_name] = self._reconstruct_dataclass(field_type, field_value)
                else:
                    reconstructed_params[field_name] = field_value
            elif field.default != dataclasses.MISSING:
                reconstructed_params[field_name] = field.default
            elif field.default_factory != dataclasses.MISSING:
                reconstructed_params[field_name] = field.default_factory()
        
        return cls(**reconstructed_params)

    @classmethod
    def from_dict(cls, data: dict) -> 'FeatureLabelCollection':
        """Create FeatureLabelCollection from dictionary."""
        collection = cls()
        for fl_data in data['feature_labels']:
            param_cls = _find_param_class(fl_data['feature_label'])
            if param_cls and isinstance(fl_data['params'], dict):
                params = collection._reconstruct_dataclass(param_cls, fl_data['params'])
            else:
                params = param_cls() if param_cls else None
            
            feature_label = FeatureLabel(fl_data['feature_label'], params)
            collection.with_feature_label(feature_label)
        
        return collection


class FeatureLabelCollectionsManager:
    """Manager for saving and loading FeatureLabelCollection objects with tag-based organization."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the manager.
        
        Args:
            base_path: Base path for storage. If None, uses ALGO_CACHE_BASE from environment.
        """
        if base_path is None:
            base_path = get_cache_base_path()
        
        self.base_path = Path(base_path)
        self.feature_labels_dir = self.base_path / "feature_labels"
        self.feature_labels_dir.mkdir(parents=True, exist_ok=True)
    
    def get_collection_path(self, tag: str) -> Path:
        """Get the file path for a tagged collection."""
        return self.feature_labels_dir / tag / "collection.json"
    
    def save(self, collection: FeatureLabelCollection, tag: str) -> None:
        """
        Save a FeatureLabelCollection with the given tag.
        
        Args:
            collection: The FeatureLabelCollection to save
            tag: Tag to identify this collection
        """
        collection_path = self.get_collection_path(tag)
        collection_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = collection.to_dict()
        
        with open(collection_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved FeatureLabelCollection with tag '{tag}' to {collection_path}")
    
    def load(self, tag: str) -> FeatureLabelCollection:
        """
        Load a FeatureLabelCollection by tag.
        
        Args:
            tag: Tag identifying the collection to load
            
        Returns:
            The loaded FeatureLabelCollection
            
        Raises:
            FileNotFoundError: If the collection doesn't exist
        """
        collection_path = self.get_collection_path(tag)
        
        if not collection_path.exists():
            raise FileNotFoundError(f"No FeatureLabelCollection found with tag '{tag}' at {collection_path}")
        
        with open(collection_path, 'r') as f:
            data = json.load(f)
        
        collection = FeatureLabelCollection.from_dict(data)
        logger.info(f"Loaded FeatureLabelCollection with tag '{tag}' from {collection_path}")
        
        return collection
    
    def exists(self, tag: str) -> bool:
        """Check if a collection with the given tag exists."""
        return self.get_collection_path(tag).exists()
    
    def list_tags(self) -> List[str]:
        """List all available tags."""
        if not self.feature_labels_dir.exists():
            return []
        
        tags = []
        for item in self.feature_labels_dir.iterdir():
            if item.is_dir() and (item / "collection.json").exists():
                tags.append(item.name)
        
        return sorted(tags)
    
    def delete(self, tag: str) -> None:
        """
        Delete a collection by tag.
        
        Args:
            tag: Tag identifying the collection to delete
            
        Raises:
            FileNotFoundError: If the collection doesn't exist
        """
        collection_path = self.get_collection_path(tag)
        
        if not collection_path.exists():
            raise FileNotFoundError(f"No FeatureLabelCollection found with tag '{tag}' at {collection_path}")
        
        collection_path.unlink()
        
        # Remove directory if empty
        try:
            collection_path.parent.rmdir()
        except OSError:
            pass  # Directory not empty
        
        logger.info(f"Deleted FeatureLabelCollection with tag '{tag}'")


