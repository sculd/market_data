import logging
import inspect
import importlib
import datetime
from typing import Optional, Any, List

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


class FeatureLabelCollection:
    def __init__(self):
        self.feature_labels: List[FeatureLabel] = []

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
        return max([fl.get_warm_up_days() for fl in self.feature_labels])

