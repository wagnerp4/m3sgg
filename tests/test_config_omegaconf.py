"""Tests for OmegaConf-based configuration system.

This module contains comprehensive tests for the new configuration management
system using OmegaConf, including structured configurations, YAML loading,
interpolation, validation, and backward compatibility.

:author: VidSgg Team
:version: 0.1.0
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.config_structured import (
    BaseConfig, STTRANConfig, STKETConfig, TempuraConfig,
    EASGConfig, SceneLLMConfig, OEDConfig, get_config_class
)
from lib.config_omegaconf import ConfigManager, Config, create_config, merge_configs
from lib.config_utils import (
    validate_config, create_experiment_config, save_experiment_config,
    load_config_with_interpolation, create_config_from_dict,
    merge_config_files, create_config_template, get_config_summary
)


class TestStructuredConfigs:
    """Test structured configuration classes."""
    
    def test_base_config_creation(self):
        """Test BaseConfig creation with default values."""
        config = BaseConfig()
        assert config.mode == "predcls"
        assert config.save_path == "output"
        assert config.dataset == "action_genome"
        assert config.lr == 1e-5
        assert config.nepoch == 10
    
    def test_sttran_config_creation(self):
        """Test STTRANConfig creation."""
        config = STTRANConfig()
        assert config.model_type == "sttran"
        assert config.mode == "predcls"
    
    def test_stket_config_creation(self):
        """Test STKETConfig creation."""
        config = STKETConfig()
        assert config.model_type == "stket"
        assert config.enc_layer_num == 1
        assert config.dec_layer_num == 3
        assert config.pred_contact_threshold == 0.5
        assert config.window_size == 3
    
    def test_tempura_config_creation(self):
        """Test TempuraConfig creation."""
        config = TempuraConfig()
        assert config.model_type == "tempura"
        assert config.obj_head == "gmm"
        assert config.rel_head == "gmm"
        assert config.K == 4
        assert config.mem_fusion == "early"
    
    def test_scenellm_config_creation(self):
        """Test SceneLLMConfig creation."""
        config = SceneLLMConfig()
        assert config.model_type == "scenellm"
        assert config.embed_dim == 1024
        assert config.codebook_size == 8192
        assert config.llm_name == "google/gemma-2-2b"
        assert config.lora_r == 16
    
    def test_oed_config_creation(self):
        """Test OEDConfig creation."""
        config = OEDConfig()
        assert config.model_type == "oed"
        assert config.num_queries == 100
        assert config.dec_layers_hopd == 6
        assert config.num_attn_classes == 3
        assert config.alpha == 0.5
    
    def test_get_config_class(self):
        """Test get_config_class function."""
        assert get_config_class("sttran") == STTRANConfig
        assert get_config_class("stket") == STKETConfig
        assert get_config_class("tempura") == TempuraConfig
        assert get_config_class("EASG") == EASGConfig
        assert get_config_class("scenellm") == SceneLLMConfig
        assert get_config_class("oed") == OEDConfig
        
        with pytest.raises(ValueError):
            get_config_class("unknown_model")


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_config_manager_creation_with_model_type(self):
        """Test ConfigManager creation with model type."""
        with patch("sys.argv", ["test.py", "-model", "sttran"]):
            config = ConfigManager(model_type="sttran")
            assert config.model_type == "sttran"
            assert config._structured_config_class == STTRANConfig
    
    def test_config_manager_creation_with_config_file(self):
        """Test ConfigManager creation with config file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
model_type: "sttran"
mode: "sgcls"
lr: 2e-5
nepoch: 20
""")
            config_path = f.name
        
        try:
            config = ConfigManager(config_path=config_path)
            assert config.get("model_type") == "sttran"
            assert config.get("mode") == "sgcls"
            assert config.get("lr") == 2e-5
            assert config.get("nepoch") == 20
        finally:
            os.unlink(config_path)
    
    def test_config_manager_get_set(self):
        """Test ConfigManager get and set methods."""
        config = ConfigManager(model_type="sttran")
        
        # Test get with default
        assert config.get("nonexistent", "default") == "default"
        
        # Test set and get
        config.set("custom_param", "custom_value")
        assert config.get("custom_param") == "custom_value"
    
    def test_config_manager_attribute_access(self):
        """Test ConfigManager attribute-style access."""
        config = ConfigManager(model_type="sttran")
        
        # Test existing attribute
        assert config.model_type == "sttran"
        assert config.mode == "predcls"
        
        # Test non-existent attribute
        with pytest.raises(AttributeError):
            _ = config.nonexistent_attribute
    
    def test_config_manager_dict_access(self):
        """Test ConfigManager dictionary-style access."""
        config = ConfigManager(model_type="sttran")
        
        # Test existing key
        assert config["model_type"] == "sttran"
        assert config["mode"] == "predcls"
        
        # Test key existence
        assert "model_type" in config
        assert "nonexistent" not in config
        
        # Test key setting
        config["custom_key"] = "custom_value"
        assert config["custom_key"] == "custom_value"
    
    def test_config_manager_save_load(self):
        """Test ConfigManager save and load functionality."""
        config = ConfigManager(model_type="sttran")
        config.set("custom_param", "custom_value")
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = f.name
        
        try:
            # Save configuration
            config.save(config_path)
            
            # Load configuration
            loaded_config = ConfigManager(config_path=config_path)
            assert loaded_config.get("model_type") == "sttran"
            assert loaded_config.get("custom_param") == "custom_value"
        finally:
            os.unlink(config_path)
    
    def test_config_manager_to_dict(self):
        """Test ConfigManager to_dict method."""
        config = ConfigManager(model_type="sttran")
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["model_type"] == "sttran"
        assert config_dict["mode"] == "predcls"


class TestConfig:
    """Test backward-compatible Config class."""
    
    def test_config_creation(self):
        """Test Config creation with backward compatibility."""
        with patch("sys.argv", ["test.py", "-model", "sttran", "-lr", "2e-5"]):
            config = Config()
            
            # Test backward compatibility attributes
            assert hasattr(config, "args")
            assert config.model_type == "sttran"
            assert config.lr == 2e-5
            assert config.mode == "predcls"
    
    def test_config_legacy_interface(self):
        """Test Config legacy interface compatibility."""
        with patch("sys.argv", ["test.py", "-model", "sttran"]):
            config = Config()
            
            # Test that legacy attributes exist
            assert hasattr(config, "parser")
            assert config.parser is None  # Legacy attribute set to None
    
    def test_config_model_type_extraction(self):
        """Test model type extraction from command line."""
        test_cases = [
            (["test.py", "-model", "sttran"], "sttran"),
            (["test.py", "--model", "scenellm"], "scenellm"),
            (["test.py", "-model", "oed", "-lr", "1e-4"], "oed"),
            (["test.py"], None),
        ]
        
        for args, expected in test_cases:
            with patch("sys.argv", args):
                config = Config()
                if expected:
                    assert config.model_type == expected
                else:
                    # Should fall back to default or None
                    assert config.model_type is None or config.model_type == "sttran"


class TestConfigUtils:
    """Test configuration utility functions."""
    
    def test_validate_config(self):
        """Test configuration validation."""
        from omegaconf import OmegaConf
        
        # Test valid configuration
        valid_config = OmegaConf.create({
            "model_type": "sttran",
            "mode": "predcls",
            "lr": 1e-5,
            "nepoch": 10,
            "data_path": "data/action_genome"
        })
        
        errors = validate_config(valid_config, "sttran")
        assert len(errors) == 0
        
        # Test invalid configuration
        invalid_config = OmegaConf.create({
            "model_type": "sttran",
            "lr": 10.0,  # Invalid learning rate
            "nepoch": -1,  # Invalid epoch count
        })
        
        errors = validate_config(invalid_config, "sttran")
        assert len(errors) > 0
    
    def test_create_experiment_config(self):
        """Test experiment configuration creation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
model_type: "sttran"
mode: "predcls"
lr: 1e-5
nepoch: 10
""")
            base_config_path = f.name
        
        try:
            overrides = {"lr": 2e-5, "nepoch": 20}
            config = create_experiment_config(
                base_config_path, "test_experiment", overrides
            )
            
            assert config.experiment_name == "test_experiment"
            assert config.lr == 2e-5
            assert config.nepoch == 20
            assert "test_experiment" in config.save_path
        finally:
            os.unlink(base_config_path)
    
    def test_create_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "model_type": "sttran",
            "mode": "predcls",
            "lr": 1e-5,
            "nepoch": 10
        }
        
        config = create_config_from_dict(config_dict, "sttran")
        assert config.model_type == "sttran"
        assert config.mode == "predcls"
        assert config.lr == 1e-5
        assert config.nepoch == 10
    
    def test_merge_config_files(self):
        """Test merging multiple configuration files."""
        # Create temporary config files
        config1_path = None
        config2_path = None
        
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write("""
model_type: "sttran"
mode: "predcls"
lr: 1e-5
""")
                config1_path = f.name
            
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write("""
nepoch: 20
save_path: "output/test"
""")
                config2_path = f.name
            
            merged_config = merge_config_files(config1_path, config2_path)
            
            assert merged_config.model_type == "sttran"
            assert merged_config.mode == "predcls"
            assert merged_config.lr == 1e-5
            assert merged_config.nepoch == 20
            assert merged_config.save_path == "output/test"
        
        finally:
            if config1_path:
                os.unlink(config1_path)
            if config2_path:
                os.unlink(config2_path)
    
    def test_create_config_template(self):
        """Test configuration template creation."""
        template_content = create_config_template("sttran")
        
        assert isinstance(template_content, str)
        assert "STTRAN Model Configuration Template" in template_content
        assert "model_type: sttran" in template_content
        assert "mode: predcls" in template_content
    
    def test_get_config_summary(self):
        """Test configuration summary generation."""
        from omegaconf import OmegaConf
        
        config = OmegaConf.create({
            "model_type": "sttran",
            "dataset": "action_genome",
            "mode": "predcls",
            "lr": 1e-5,
            "nepoch": 10,
            "data_path": "data/action_genome",
            "save_path": "output"
        })
        
        summary = get_config_summary(config)
        
        assert summary["model_type"] == "sttran"
        assert summary["dataset"] == "action_genome"
        assert summary["mode"] == "predcls"
        assert summary["learning_rate"] == 1e-5
        assert summary["epochs"] == 10


class TestIntegration:
    """Integration tests for the configuration system."""
    
    def test_end_to_end_config_workflow(self):
        """Test complete configuration workflow."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
model_type: "sttran"
mode: "predcls"
dataset: "action_genome"
data_path: "data/action_genome"
lr: 1e-5
nepoch: 10
save_path: "output/test"
""")
            config_path = f.name
        
        try:
            # Load configuration
            config = ConfigManager(config_path=config_path)
            
            # Validate configuration
            errors = validate_config(config._config, "sttran")
            assert len(errors) == 0
            
            # Get summary
            summary = get_config_summary(config._config)
            assert summary["model_type"] == "sttran"
            
            # Save configuration
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
                save_path = f.name
            
            try:
                config.save(save_path)
                
                # Reload and verify
                reloaded_config = ConfigManager(config_path=save_path)
                assert reloaded_config.get("model_type") == "sttran"
                assert reloaded_config.get("lr") == 1e-5
            finally:
                os.unlink(save_path)
        
        finally:
            os.unlink(config_path)
    
    def test_backward_compatibility_with_existing_code(self):
        """Test backward compatibility with existing code patterns."""
        with patch("sys.argv", ["test.py", "-model", "sttran", "-lr", "2e-5", "-nepoch", "20"]):
            config = Config()
            
            # Test that existing code patterns still work
            assert config.model_type == "sttran"
            assert config.lr == 2e-5
            assert config.nepoch == 20
            
            # Test that args attribute exists (legacy compatibility)
            assert hasattr(config, "args")
            assert isinstance(config.args, dict)
            
            # Test that configuration can be used in existing patterns
            timestamp = "20240101_120000"
            data_path_suffix = os.path.basename(config.data_path)
            new_save_path = os.path.join(
                "output", data_path_suffix, config.model_type, config.mode, timestamp
            )
            
            assert "action_genome" in new_save_path
            assert "sttran" in new_save_path
            assert "predcls" in new_save_path


if __name__ == "__main__":
    pytest.main([__file__])
