import dataclasses
import datetime
import logging
import os
from typing import Optional

from market_data.feature.label import FeatureLabel, FeatureLabelCollection
from market_data.feature.sequential_feature import SequentialFeatureParam
from market_data.ingest.common import CacheContext
from market_data.machine_learning.resample.calc import CumSumResampleParams
from market_data.machine_learning.resample.param import ResampleParam
from market_data.target.param import TargetParamsBatch
from market_data.util.param import Param

logger = logging.getLogger(__name__)


def verify_parsed_params_match(
    cached_resample: ResampleParam,
    cached_target: TargetParamsBatch,
    cached_seq: Optional[SequentialFeatureParam],
    requested_resample: ResampleParam,
    requested_target: TargetParamsBatch,
    requested_seq: Optional[SequentialFeatureParam],
) -> bool:
    """
    Verify that the cached parameters match the requested parameters.
    Uses to_str() methods for comparison.
    
    Returns:
        True if parameters match, False otherwise
    """
    # Check resample parameters
    if requested_resample and cached_resample:
        if requested_resample.to_str() != cached_resample.to_str():
            logger.debug(f"Resample parameter mismatch: requested={requested_resample.to_str()}, cached={cached_resample.to_str()}")
            return False
        # Also check class type
        if type(requested_resample) != type(cached_resample):
            logger.debug(f"Resample class mismatch: requested={type(requested_resample).__name__}, cached={type(cached_resample).__name__}")
            return False
    elif bool(requested_resample) != bool(cached_resample):
        logger.debug("Resample parameter existence mismatch")
        return False
    
    # Check target parameters
    if requested_target and cached_target:
        # Compare using get_params_dir()
        if requested_target.get_params_dir() != cached_target.get_params_dir():
            logger.debug(f"Target parameter mismatch: requested={requested_target.get_params_dir()}, cached={cached_target.get_params_dir()}")
            return False
    elif bool(requested_target) != bool(cached_target):
        logger.debug("Target parameter existence mismatch")
        return False
    
    # Check sequential parameters
    if (requested_seq is not None) != (cached_seq is not None):
        logger.debug(f"Sequential parameter existence mismatch: requested={requested_seq is not None}, cached={cached_seq is not None}")
        return False
    
    if requested_seq is not None and cached_seq is not None:
        # Compare using get_params_dir()
        if requested_seq.get_params_dir() != cached_seq.get_params_dir():
            logger.debug(f"Sequential parameter mismatch: requested={requested_seq.get_params_dir()}, cached={cached_seq.get_params_dir()}")
            return False
    
    return True


@dataclasses.dataclass
class MlDataParam(Param):
    feature_collection: FeatureLabelCollection
    target_params_batch: TargetParamsBatch
    resample_params: ResampleParam
    seq_param: Optional[SequentialFeatureParam] = None
    resample_columns: list[str] = dataclasses.field(default_factory=list)
    
    def to_str(self) -> str:
        lines = []
        lines.append("ML Data Cache Parameters")
        lines.append("=" * 30)
        lines.append("")
        
        # Resample parameters - store class name and to_str() output
        lines.append("Resample Parameters:")
        lines.append(f"class: {self.resample_params.__class__.__module__}.{self.resample_params.__class__.__name__}")
        lines.append(f"params: {self.resample_params.to_str()}")
        lines.append("")
        
        # Target parameters
        lines.append("Target Parameters:")
        lines.append(f"params: {self.target_params_batch.get_params_dir()}")
        lines.append("")
        
        # Sequential parameters
        if self.seq_param is not None:
            lines.append("Sequential Parameters:")
            lines.append(f"params: {self.seq_param.get_params_dir()}")
            lines.append("")
        
        # Resample columns
        if self.resample_columns:
            lines.append("Resample Columns:")
            lines.append(f"columns: {','.join(self.resample_columns)}")
            lines.append("")
        
        # Feature parameters
        lines.append("Feature Parameters:")
        for feature_label_obj in sorted(self.feature_collection.feature_labels, key=lambda x: x.feature_label):
            lines.append(f"{feature_label_obj.feature_label}: {feature_label_obj.params.to_str()}")
        lines.append("")
        
        # Add timestamp
        lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
        
        description = "\n".join(lines)
        return description
    
    @classmethod
    def from_str(cls, param_str: str) -> "MlDataParam":
        # Parse sections
        resample_params = None
        feature_collection = FeatureLabelCollection()
        target_params_batch = None
        seq_param = None
        resample_columns = []
        
        lines = param_str.split('\n')
        current_section = None
        resample_class_str = None
        
        for line in lines:
            line_stripped = line.strip()
            
            # Identify section headers
            if line_stripped == "Resample Parameters:":
                current_section = "resample"
                continue
            elif line_stripped == "Target Parameters:":
                current_section = "target"
                continue
            elif line_stripped == "Sequential Parameters:":
                current_section = "sequential"
                continue
            elif line_stripped == "Resample Columns:":
                current_section = "resample_columns"
                continue
            elif line_stripped == "Feature Parameters:":
                current_section = "features"
                continue
            elif line_stripped.startswith("Generated:"):
                # End of meaningful content
                break
            elif line_stripped == "" or line_stripped.startswith("="):
                # Skip empty lines and separators
                continue
            
            # Parse content based on current section
            if current_section == "resample":
                if line_stripped.startswith("class:"):
                    resample_class_str = line_stripped.split("class:", 1)[1].strip()
                elif line_stripped.startswith("params:"):
                    params_str = line_stripped.split("params:", 1)[1].strip()
                    
                    # Import and instantiate the correct resample class
                    if resample_class_str:
                        try:
                            module_name, class_name = resample_class_str.rsplit('.', 1)
                            module = __import__(module_name, fromlist=[class_name])
                            resample_class = getattr(module, class_name)
                            resample_params = resample_class.from_str(params_str)
                        except Exception as e:
                            logger.warning(f"Failed to parse resample params: {e}")
                            # Fallback to default
                            resample_params = CumSumResampleParams.from_str(params_str)
                    else:
                        # Default to CumSumResampleParams
                        resample_params = CumSumResampleParams.from_str(params_str)
                        
            elif current_section == "target" and line_stripped.startswith("params:"):
                # For now, create a basic TargetParamsBatch
                # TODO: Implement proper from_str for TargetParamsBatch
                target_params_batch = TargetParamsBatch()
                
            elif current_section == "sequential" and line_stripped.startswith("params:"):
                # For now, create a basic SequentialFeatureParam
                # TODO: Implement proper from_str for SequentialFeatureParam
                seq_param = SequentialFeatureParam()
                
            elif current_section == "resample_columns" and line_stripped.startswith("columns:"):
                columns_str = line_stripped.split("columns:", 1)[1].strip()
                if columns_str:
                    resample_columns = [col.strip() for col in columns_str.split(',')]
                
            elif current_section == "features" and ':' in line_stripped:
                # Parse feature label and params
                feature_label, params_str = line_stripped.split(':', 1)
                feature_label = feature_label.strip()
                params_str = params_str.strip()
                
                # Create FeatureLabel with parsed params
                feature_label_obj = FeatureLabel(feature_label)
                
                # Use from_str to parse the params
                if hasattr(feature_label_obj.params, 'from_str'):
                    try:
                        feature_label_obj.params = feature_label_obj.params.__class__.from_str(params_str)
                    except Exception as e:
                        logger.debug(f"Error parsing params for {feature_label}: {e}, using defaults")
                
                feature_collection.with_feature_label(feature_label_obj)

        return cls(
            feature_collection=feature_collection,
            target_params_batch=target_params_batch,
            resample_params=resample_params,
            seq_param=seq_param,
            resample_columns=resample_columns
        )


    def write_description_file(
            self,
            params_dir: str,
    ) -> None:
        """
        Write parameter description to description.txt file in the cache directory.
        Uses to_str() methods for serialization.
        """
        description = self.to_str()
        description_path = os.path.join(params_dir, "description.txt")
        
        # Create directory if it doesn't exist
        os.makedirs(params_dir, exist_ok=True)
        
        # Write description file
        with open(description_path, 'w') as f:
            f.write(description)


    @classmethod
    def read_description_file(
            cls,
            description_path: str,
    ) -> "MlDataParam":
        """
        Read and parse parameter description from description.txt file.
        Uses from_str() methods for deserialization.
        
        Args:
            description_path: Path to the description.txt file
            
        Returns:
            Tuple of (resample_params, feature_collection, target_params_batch, seq_param)
            or None if file cannot be parsed
        """
        if not os.path.exists(description_path):
            logger.warning(f"Description file not found: {description_path}")
            return None
        
        try:
            with open(description_path, 'r') as f:
                content = f.read()
            
            return cls.from_str(content)            
        except Exception as e:
            logger.error(f"Error parsing description file {description_path}: {e}")
            return None


    def find_cached_ml_data_with_features(
            self,
            cache_context: CacheContext,
    ) -> Optional[str]:
        """
        Find a cached ML data UUID folder that contains all requested feature labels.
        
        Returns:
            UUID string of the cache folder containing all requested features, or None if not found
        """
        # Get the base ML data path
        base_path = cache_context.get_ml_data_path("")
        
        if not os.path.exists(base_path):
            logger.warning(f"ML data cache directory does not exist: {base_path}")
            return None
        
        # Get requested feature labels
        requested_features = set()
        for feature_label_obj in self.feature_collection.feature_labels:
            requested_features.add(f"{feature_label_obj.feature_label}@{feature_label_obj.params}")
        
        logger.info(f"Looking for cache with features: {requested_features}")
        
        # Scan UUID folders
        for uuid_folder in os.listdir(base_path):
            uuid_path = os.path.join(base_path, uuid_folder)
            
            # Skip if not a directory
            if not os.path.isdir(uuid_path):
                continue
                
            # Check for description.txt file
            description_path = os.path.join(uuid_path, "description.txt")
            if not os.path.exists(description_path):
                logger.error(f"No description.txt in {uuid_folder}, skipping")
                continue
            
            # Parse description file
            cached_params = MlDataParam.read_description_file(description_path)
            if cached_params is None:
                logger.error(f"Could not parse description file in {uuid_folder}")
                continue
            
            cached_resample = cached_params.resample_params
            cached_features_collection = cached_params.feature_collection
            cached_target = cached_params.target_params_batch
            cached_seq = cached_params.seq_param
            
            # Get cached feature labels
            cached_features = set()
            for feature_label_obj in cached_features_collection.feature_labels:
                cached_features.add(f"{feature_label_obj.feature_label}@{feature_label_obj.params}")
            
            # Check if this cache contains all requested features
            if requested_features.issubset(cached_features):
                logger.info(f"Found matching cache UUID: {uuid_folder}")
                logger.info(f"  Cached features: {cached_features}")
                logger.info(f"  Requested features: {requested_features}")
                
                # Verify other parameters match
                if verify_parsed_params_match(
                    cached_resample,
                    cached_target,
                    cached_seq,
                    self.resample_params,
                    self.target_params_batch,
                    self.seq_param
                ):
                    return uuid_folder
                else:
                    logger.debug(f"Cache {uuid_folder} has matching features but different parameters")
        
        logger.warning(f"No cached ML data found containing all requested features: {requested_features}")
        return None
