"""
Tests for configuration module.
"""
import pytest
import os
import tempfile
from src.utils.config import (
    load_config, 
    get_config, 
    reset_config,
    AppConfig,
    PipelineConfig,
    ControlConfig
)


class TestConfig:
    """Test cases for configuration loading."""
    
    def test_load_default_config(self):
        """Test loading default config when file doesn't exist."""
        # Make sure we're using default
        config = load_config("nonexistent.yaml")
        
        assert isinstance(config, AppConfig)
        assert config.pipeline.target_minutes == 5
        assert config.control.alpha == 0.05
    
    def test_load_from_file(self):
        """Test loading config from YAML file."""
        config = load_config("config.yaml")
        
        assert config.pipeline.target_minutes == 5
        assert config.pipeline.ttl_minutes == 15
        assert config.control.kp == 0.5
        assert config.control.ki == 0.1
        assert config.control.kd == 0.0
    
    def test_get_config_singleton(self):
        """Test that get_config returns singleton."""
        reset_config()
        
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_reset_config(self):
        """Test config reset."""
        reset_config()
        
        config1 = get_config()
        reset_config()
        config2 = get_config()
        
        # Should be different instances after reset
        assert config1 is not config2
    
    def test_pipeline_config_defaults(self):
        """Test PipelineConfig defaults."""
        config = PipelineConfig()
        
        assert config.target_minutes == 5
        assert config.ttl_minutes == 15
    
    def test_control_config_defaults(self):
        """Test ControlConfig defaults."""
        config = ControlConfig()
        
        assert config.kp == 0.5
        assert config.ki == 0.1
        assert config.kd == 0.0
        assert config.alpha == 0.05
        assert config.min_weight == 0.1
        assert config.max_weight == 2.0
    
    def test_custom_config_values(self):
        """Test custom config values from file."""
        config = load_config("config.yaml")
        
        # Verify custom values from our config.yaml
        assert config.features.buffer_capacity == 5
        assert config.model.n_estimators == 500
        assert config.labeler.threshold == 0.0005
        assert config.feedback.high_vol_threshold == 0.30