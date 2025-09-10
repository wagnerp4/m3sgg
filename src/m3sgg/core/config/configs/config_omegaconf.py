"""OmegaConf-based configuration management for M3SGG.

This module provides configuration management using OmegaConf for structured
configuration handling, YAML loading, interpolation, and validation.

:author: M3SGG Team
:version: 0.1.0
"""

import os
import sys
from typing import Any, Dict, Optional, Union
from omegaconf import OmegaConf, DictConfig

from .config_structured import get_config_class


class ConfigManager:
    """Configuration manager using OmegaConf for structured configuration handling.

    This class provides a unified interface for managing configurations with
    support for YAML files, command-line arguments, and structured validation.

    :param model_type: Model type identifier
    :type model_type: Optional[str]
    :param config_path: Path to configuration file
    :type config_path: Optional[str]
    :param overrides: Configuration overrides
    :type overrides: Optional[Dict[str, Any]]
    """

    def __init__(
        self,
        model_type: Optional[str] = None,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the configuration manager.

        :param model_type: Model type identifier
        :type model_type: Optional[str]
        :param config_path: Path to configuration file
        :type config_path: Optional[str]
        :param overrides: Configuration overrides
        :type overrides: Optional[Dict[str, Any]]
        """
        self._config = None
        self._structured_config_class = None

        if config_path:
            self._load_from_file(config_path)
        elif model_type:
            self._load_from_model_type(model_type)
        else:
            # Try to extract from command line arguments
            self._load_from_args()

        if overrides:
            self._apply_overrides(overrides)

    def _load_from_file(self, config_path: str):
        """Load configuration from YAML file.

        :param config_path: Path to configuration file
        :type config_path: str
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self._config = OmegaConf.load(config_path)

        # Determine model type from config
        model_type = self._config.get("model_type")
        if model_type:
            self._structured_config_class = get_config_class(model_type)

    def _load_from_model_type(self, model_type: str):
        """Load configuration from model type.

        :param model_type: Model type identifier
        :type model_type: str
        """
        self._structured_config_class = get_config_class(model_type)

        # Create default configuration as a regular dict, not structured
        default_config = self._structured_config_class()
        config_dict = {key: value for key, value in default_config.__dict__.items()}
        self._config = OmegaConf.create(config_dict)

    def _load_from_args(self):
        """Load configuration from command line arguments.

        This method attempts to extract model type from sys.argv and create
        a default configuration.
        """
        model_type = None

        # Simple argument parsing for model type
        for i, arg in enumerate(sys.argv):
            if arg in ["-model", "--model"] and i + 1 < len(sys.argv):
                model_type = sys.argv[i + 1]
                break

        if model_type:
            self._load_from_model_type(model_type)
        else:
            # Default to STTRAN if no model type specified
            self._load_from_model_type("sttran")

    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides.

        :param overrides: Configuration overrides
        :type overrides: Dict[str, Any]
        """
        if self._config is None:
            raise RuntimeError("No configuration loaded")

        for key, value in overrides.items():
            self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        :param key: Configuration key
        :type key: str
        :param default: Default value if key not found
        :type default: Any
        :return: Configuration value
        :rtype: Any
        """
        if self._config is None:
            return default

        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value.

        :param key: Configuration key
        :type key: str
        :param value: Configuration value
        :type value: Any
        """
        if self._config is None:
            raise RuntimeError("No configuration loaded")

        self._config[key] = value

    def save(self, path: str):
        """Save configuration to file.

        :param path: Path to save configuration
        :type path: str
        """
        if self._config is None:
            raise RuntimeError("No configuration loaded")

        OmegaConf.save(self._config, path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        :return: Configuration as dictionary
        :rtype: Dict[str, Any]
        """
        if self._config is None:
            return {}

        return OmegaConf.to_container(self._config, resolve=True)

    def __getattr__(self, name: str) -> Any:
        """Get configuration attribute.

        :param name: Attribute name
        :type name: str
        :return: Configuration value
        :rtype: Any
        :raises AttributeError: If attribute not found
        """
        if self._config is None:
            raise AttributeError("Configuration not loaded")

        if name in self._config:
            return self._config[name]

        raise AttributeError(f"Configuration has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Get configuration item.

        :param key: Configuration key
        :type key: str
        :return: Configuration value
        :rtype: Any
        """
        if self._config is None:
            raise KeyError("Configuration not loaded")

        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        """Set configuration item.

        :param key: Configuration key
        :type key: str
        :param value: Configuration value
        :type value: Any
        """
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if configuration contains key.

        :param key: Configuration key
        :type key: str
        :return: True if key exists
        :rtype: bool
        """
        if self._config is None:
            return False

        return key in self._config


class Config:
    """Backward-compatible configuration class.

    This class provides backward compatibility with the existing configuration
    system while using the new OmegaConf-based infrastructure.
    """

    def __init__(self):
        """Initialize the configuration."""
        self._config_manager = ConfigManager()
        self._config = self._config_manager._config

        # Legacy attributes for backward compatibility
        self.args = self._config_manager.to_dict()
        self.parser = None  # Legacy attribute

    def __getattr__(self, name: str) -> Any:
        """Get configuration attribute.

        :param name: Attribute name
        :type name: str
        :return: Configuration value
        :rtype: Any
        """
        return getattr(self._config_manager, name)

    def __getitem__(self, key: str) -> Any:
        """Get configuration item.

        :param key: Configuration key
        :type key: str
        :return: Configuration value
        :rtype: Any
        """
        return self._config_manager[key]

    def __setitem__(self, key: str, value: Any):
        """Set configuration item.

        :param key: Configuration key
        :type key: str
        :param value: Configuration value
        :type value: Any
        """
        self._config_manager[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if configuration contains key.

        :param key: Configuration key
        :type key: str
        :return: True if key exists
        :rtype: bool
        """
        return key in self._config_manager


def create_config(model_type: str, **kwargs) -> ConfigManager:
    """Create a configuration manager for a specific model type.

    :param model_type: Model type identifier
    :type model_type: str
    :param kwargs: Additional configuration parameters
    :type kwargs: Any
    :return: Configuration manager
    :rtype: ConfigManager
    """
    return ConfigManager(model_type=model_type, overrides=kwargs)


def merge_configs(
    *configs: Union[ConfigManager, DictConfig, Dict[str, Any]],
) -> ConfigManager:
    """Merge multiple configurations.

    :param configs: Configurations to merge
    :type configs: Union[ConfigManager, DictConfig, Dict[str, Any]]
    :return: Merged configuration manager
    :rtype: ConfigManager
    """
    merged_config = None

    for config in configs:
        if isinstance(config, ConfigManager):
            config_dict = config.to_dict()
        elif isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config

        if merged_config is None:
            merged_config = OmegaConf.create(config_dict)
        else:
            merged_config = OmegaConf.merge(merged_config, config_dict)

    # Create new ConfigManager with merged config
    manager = ConfigManager()
    manager._config = merged_config

    # Determine model type from merged config
    model_type = merged_config.get("model_type")
    if model_type:
        manager._structured_config_class = get_config_class(model_type)

    return manager
