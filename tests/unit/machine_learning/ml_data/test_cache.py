"""
Unit tests for market_data.machine_learning.ml_data.cache module.

This module tests the caching functionality for ML data including:
- Description file writing and reading
- Parameter matching
- Cache discovery and loading
- Column selection
"""

import datetime
import os
import tempfile
import logging
from typing import Optional
from unittest.mock import MagicMock, patch

import setup_env  # needed for env variables

import pandas as pd
import pytest

from market_data.feature.label import FeatureLabel, FeatureLabelCollection
from market_data.feature.impl.ema import EMAParams
from market_data.feature.param import SequentialFeatureParam
from market_data.ingest.common import CacheContext
from market_data.machine_learning.ml_data.cache_description import (
    find_cached_ml_data_with_features,
    read_description_file,
    verify_parsed_params_match,
    write_description_file,
)
from market_data.machine_learning.resample.calc import CumSumResampleParams
from market_data.machine_learning.resample.param import ResampleParam
from market_data.target.calc import TargetParamsBatch
from market_data.util.time import TimeRange

logger = logging.getLogger(__name__)

class TestWriteDescriptionFile:
    """Test the _write_description_file function."""
    
    def test_write_description_file_basic(self):
        """Test basic description file writing functionality."""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample parameters
            resample_params = CumSumResampleParams(price_col="close", threshold=0.05)
            
            # Create a simple feature collection
            feature_collection = FeatureLabelCollection()
            feature_collection.with_feature_label(FeatureLabel("ema"))
            feature_collection.with_feature_label(FeatureLabel("bollinger"))
            
            target_params_batch = TargetParamsBatch()
            
            # Act
            write_description_file(
                params_dir=temp_dir,
                resample_params=resample_params,
                feature_collection=feature_collection,
                target_params_batch=target_params_batch,
                seq_param=None
            )
            
            # Assert
            description_path = os.path.join(temp_dir, "description.txt")
            assert os.path.exists(description_path), "Description file should be created"
            
            # Read and verify content
            with open(description_path, 'r') as f:
                content = f.read()
            
            # Check that key sections are present
            assert "ML Data Cache Parameters" in content
            assert "Resample Parameters:" in content
            assert "Target Parameters:" in content
            assert "Feature Parameters:" in content
            assert "Generated:" in content
            
            # Check that resample params are written correctly
            assert "class: market_data.machine_learning.resample.calc.CumSumResampleParams" in content
            assert "params: price_col:close,threshold:0.05" in content
            
            # Check that feature params are written
            assert "ema:" in content
            assert "bollinger:" in content
            
            # Ensure no sequential params section exists
            assert "Sequential Parameters:" not in content


class TestReadDescriptionFile:
    """Test the _read_description_file function."""
    
    def test_read_description_file_basic(self):
        """Test basic description file reading functionality."""
        # Arrange - Use static test asset from working directory
        description_path = os.path.join(os.getcwd(), "tests", "assets", "sample_description.txt")
        
        # Ensure the test asset exists
        assert os.path.exists(description_path), f"Test asset not found: {description_path}"
        
        # Act - Read and parse the description file
        result = read_description_file(description_path)
        
        # Assert - Verify parsing was successful
        assert result is not None, "Should successfully parse the description file"
        
        resample_params, feature_collection, target_params, seq_param = result
        
        # Verify resample parameters
        assert resample_params is not None, "Should parse resample parameters"
        assert isinstance(resample_params, CumSumResampleParams), "Should create correct resample class"
        assert resample_params.price_col == "close", "Should parse price_col correctly"
        assert resample_params.threshold == 0.05, "Should parse threshold correctly"
        
        # Verify feature collection
        assert feature_collection is not None, "Should parse feature collection"
        assert len(feature_collection.feature_labels) == 2, "Should parse 2 features"
        
        # Check feature labels
        feature_labels = [f.feature_label for f in feature_collection.feature_labels]
        assert "bollinger" in feature_labels, "Should include bollinger feature"
        assert "ema" in feature_labels, "Should include ema feature"
        
        # Verify feature parameters are parsed correctly
        ema_feature = next((f for f in feature_collection.feature_labels if f.feature_label == "ema"), None)
        assert ema_feature is not None, "Should find EMA feature"
        
        # Verify EMA parameters: periods:[5,15,30,60,120],price_col:close,include_price_relatives:true
        assert hasattr(ema_feature.params, 'periods'), "EMA params should have periods"
        assert ema_feature.params.periods == [5, 15, 30, 60, 120], "EMA periods should match expected values"
        assert hasattr(ema_feature.params, 'price_col'), "EMA params should have price_col"
        assert ema_feature.params.price_col == "close", "EMA price_col should be 'close'"
        assert hasattr(ema_feature.params, 'include_price_relatives'), "EMA params should have include_price_relatives"
        assert ema_feature.params.include_price_relatives == True, "EMA include_price_relatives should be True"
        
        bollinger_feature = next((f for f in feature_collection.feature_labels if f.feature_label == "bollinger"), None)
        assert bollinger_feature is not None, "Should find Bollinger feature"
        
        # Verify Bollinger parameters: period:20,std_dev:2.0,price_col:close
        assert hasattr(bollinger_feature.params, 'period'), "Bollinger params should have period"
        assert bollinger_feature.params.period == 20, "Bollinger period should be 20"
        assert hasattr(bollinger_feature.params, 'std_dev'), "Bollinger params should have std_dev"
        assert bollinger_feature.params.std_dev == 2.0, "Bollinger std_dev should be 2.0"
        assert hasattr(bollinger_feature.params, 'price_col'), "Bollinger params should have price_col"
        assert bollinger_feature.params.price_col == "close", "Bollinger price_col should be 'close'"
        
        # Verify target parameters
        assert target_params is not None, "Should parse target parameters"
        
        # Verify sequential parameters (should be None for this test file)
        assert seq_param is None, "Should not have sequential parameters in this test file"
        
        logger.info(f"Successfully parsed description file:")
        logger.info(f"  Resample: {type(resample_params).__name__} - {resample_params.to_str()}")
        logger.info(f"  Features: {feature_labels}")
        logger.info(f"  Target: {type(target_params).__name__}")
        logger.info(f"  Sequential: {seq_param}")


class TestFindCachedMlDataWithFeatures:
    """Test the _find_cached_ml_data_with_features function."""
    
    def test_find_cached_data_exact_match(self, cache_test_setup):
        """Test finding cached data with exact feature match using real static asset."""
        # Arrange - Get setup data from fixture
        setup = cache_test_setup
        mock_cache_context = setup['mock_cache_context']
        uuid_folder = setup['uuid_folder']
        
        # Create feature collection that matches the static asset (has bollinger and ema)
        feature_collection = FeatureLabelCollection()
        feature_collection.with_feature_label(FeatureLabel("bollinger"))
        feature_collection.with_feature_label(FeatureLabel("ema"))
        
        # Use same parameters as in the static asset
        resample_params = CumSumResampleParams(price_col="close", threshold=0.05)
        target_params = TargetParamsBatch()
        
        # Act
        result = find_cached_ml_data_with_features(
            cache_context=mock_cache_context,
            feature_collection=feature_collection,
            target_params_batch=target_params,
            resample_params=resample_params
        )
        
        # Assert
        assert result is not None, "Should find a matching cache"
        assert result == uuid_folder, "Should return the matching UUID"
        
        logger.info(f"Successfully found cache UUID: {result}")
        logger.info(f"Test verified real integration with static asset")
    
    def test_find_cached_data_superset_match(self, cache_test_setup):
        """Test finding cached data with superset of requested features."""
        # Arrange - Get setup data from fixture  
        setup = cache_test_setup
        mock_cache_context = setup['mock_cache_context']
        uuid_folder = setup['uuid_folder']
        
        # Request only EMA feature (cache has both bollinger and ema, so it's a superset)
        feature_collection = FeatureLabelCollection()
        feature_collection.with_feature_label(FeatureLabel("ema"))
        
        # Use same parameters as in the static asset
        resample_params = CumSumResampleParams(price_col="close", threshold=0.05)
        target_params = TargetParamsBatch()
        
        # Act
        result = find_cached_ml_data_with_features(
            cache_context=mock_cache_context,
            feature_collection=feature_collection,
            target_params_batch=target_params,
            resample_params=resample_params
        )
        
        # Assert
        assert result is not None, "Should find cache with superset of features"
        assert result == uuid_folder, "Should return the UUID with superset of features"
        
        logger.info(f"Successfully found superset cache UUID: {result}")
    
    def test_find_cached_data_no_match(self, cache_test_setup):
        """Test behavior when no matching cached data is found."""
        # Arrange - Get setup data from fixture  
        setup = cache_test_setup
        mock_cache_context = setup['mock_cache_context']
        uuid_folder = setup['uuid_folder']
        
        # Request only EMA feature (cache has both bollinger and ema, so it's a superset)
        feature_collection = FeatureLabelCollection()
        feature_collection.with_feature_label(FeatureLabel("bollinger"))
        feature_collection.with_feature_label(FeatureLabel("ema"))
        feature_collection.with_feature_label(FeatureLabel("volume"))
        
        # Use same parameters as in the static asset
        resample_params = CumSumResampleParams(price_col="close", threshold=0.05)
        target_params = TargetParamsBatch()
        
        # Act
        result = find_cached_ml_data_with_features(
            cache_context=mock_cache_context,
            feature_collection=feature_collection,
            target_params_batch=target_params,
            resample_params=resample_params
        )
        
        # Assert
        assert result is None, "Should NOT be able to find cache with superset of features"
    
    def test_find_cached_data_missing_features(self, cache_test_setup):
        """Test behavior when cached data is missing some requested features."""
        # Arrange - Get setup data from fixture  
        setup = cache_test_setup
        mock_cache_context = setup['mock_cache_context']
        uuid_folder = setup['uuid_folder']
        
        # Request only EMA feature (cache has both bollinger and ema, so it's a superset)
        feature_collection = FeatureLabelCollection()
        feature_collection.with_feature_label(FeatureLabel("bollinger"))
        # alter the ema param, which should be now missing.
        ema_param = EMAParams(periods = [10])
        feature_collection.with_feature_label(FeatureLabel("ema", ema_param))
        
        # Use same parameters as in the static asset
        resample_params = CumSumResampleParams(price_col="close", threshold=0.05)
        target_params = TargetParamsBatch()
        
        # Act
        result = find_cached_ml_data_with_features(
            cache_context=mock_cache_context,
            feature_collection=feature_collection,
            target_params_batch=target_params,
            resample_params=resample_params
        )
        
        # Assert
        assert result is None, "Should NOT be able to find cache with superset of features"


# Test fixtures
@pytest.fixture
def sample_feature_collection():
    """Create a sample FeatureLabelCollection for testing."""
    collection = FeatureLabelCollection()
    # TODO: Add sample features
    return collection


@pytest.fixture
def cache_test_setup():
    """Setup test cache directory with sample description file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test cache directory structure
        ml_data_path = os.path.join(temp_dir, "ml_data")
        os.makedirs(ml_data_path)
        
        # Create UUID folder with description file
        uuid_folder = "test-uuid-123"
        uuid_path = os.path.join(ml_data_path, uuid_folder)
        os.makedirs(uuid_path)
        
        # Copy our static test asset as the description file
        import shutil
        static_asset_path = os.path.join(os.getcwd(), "tests", "assets", "sample_description.txt")
        description_path = os.path.join(uuid_path, "description.txt")
        shutil.copy2(static_asset_path, description_path)
        
        # Create mock cache context that points to our test directory
        mock_cache_context = MagicMock(spec=CacheContext)
        mock_cache_context.get_ml_data_path.return_value = ml_data_path
        
        # Yield setup data to tests
        yield {
            'temp_dir': temp_dir,
            'ml_data_path': ml_data_path,
            'uuid_folder': uuid_folder,
            'uuid_path': uuid_path,
            'mock_cache_context': mock_cache_context,
            'description_path': description_path
        }


@pytest.fixture
def sample_resample_params():
    """Create sample resample parameters for testing."""
    return CumSumResampleParams(price_col="close", threshold=0.05)


@pytest.fixture
def sample_target_params():
    """Create sample target parameters for testing."""
    return TargetParamsBatch()


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_cache_context():
    """Create a mock CacheContext for testing."""
    context = MagicMock(spec=CacheContext)
    # TODO: Configure mock behavior
    return context


@pytest.fixture
def sample_time_range():
    """Create a sample TimeRange for testing."""
    start = datetime.datetime(2024, 1, 1)
    end = datetime.datetime(2024, 1, 2)
    return TimeRange(start, end)

