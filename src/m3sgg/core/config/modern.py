"""Modern configuration system using OmegaConf.

This module provides a modern configuration management system using OmegaConf
with support for YAML files, command-line overrides, interpolation, and
structured validation.

:author: M3SGG Team
:version: 0.1.0
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue

from .structured.base import BaseConfig, get_config_class


class ConfigManager:
    """Modern configuration manager using OmegaConf.

    This class provides a unified interface for configuration management that
    supports YAML files, command-line arguments, interpolation, and structured
    validation.

    :param config_path: Path to configuration file
    :type config_path: Optional[str]
    :param model_type: Model type for structured configuration
    :type model_type: Optional[str]
    :param cli_args: Command-line arguments to override config
    :type cli_args: Optional[List[str]]
    :param overrides: Additional configuration overrides
    :type overrides: Optional[Dict[str, Any]]
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_type: Optional[str] = None,
        cli_args: Optional[List[str]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the configuration manager.

        :param config_path: Path to configuration file
        :type config_path: Optional[str]
        :param model_type: Model type for structured configuration
        :type model_type: Optional[str]
        :param cli_args: Command-line arguments to override config
        :type cli_args: Optional[List[str]]
        :param overrides: Additional configuration overrides
        :type overrides: Optional[Dict[str, Any]]
        """
        self.config_path = config_path
        self.model_type = model_type
        self.cli_args = cli_args or sys.argv[1:]
        self.overrides = overrides or {}
        self._config: Optional[DictConfig] = None
        self._structured_config_class: Optional[Type] = None

        self._load_configuration()

    def _load_configuration(self):
        """Load configuration from various sources with proper precedence."""
        configs_to_merge = []

        # 1. Load base configuration
        base_config_path = Path("src/m3sgg/core/config/presets/base.yaml")
        if base_config_path.exists():
            base_config = OmegaConf.load(base_config_path)
            configs_to_merge.append(base_config)

        # 2. Load model-specific configuration if provided
        if self.config_path:
            if Path(self.config_path).exists():
                model_config = OmegaConf.load(self.config_path)
                configs_to_merge.append(model_config)
            else:
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_path}"
                )

        # 3. Load structured configuration if model_type is provided
        if self.model_type:
            self._structured_config_class = get_config_class(self.model_type)
            # Create a flexible config from the structured config instance
            structured_instance = self._structured_config_class()
            structured_config = OmegaConf.create(structured_instance.__dict__)
            configs_to_merge.append(structured_config)

        # 4. Parse command-line arguments
        if self.cli_args:
            cli_config = OmegaConf.from_cli(self.cli_args)
            configs_to_merge.append(cli_config)

        # 5. Apply overrides
        if self.overrides:
            overrides_config = OmegaConf.create(self.overrides)
            configs_to_merge.append(overrides_config)

        # 6. Merge all configurations
        if configs_to_merge:
            # Start with a flexible DictConfig to allow arbitrary keys
            self._config = OmegaConf.create({})
            for config in configs_to_merge:
                # Use merge with struct=False to allow additional keys
                self._config = OmegaConf.merge(self._config, config)
        else:
            # Fallback to default structured config
            if self.model_type:
                self._structured_config_class = get_config_class(self.model_type)
                self._config = OmegaConf.structured(self._structured_config_class)
            else:
                self._config = OmegaConf.structured(BaseConfig)

        # 7. Resolve interpolations
        OmegaConf.resolve(self._config)

        # 8. Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate the loaded configuration."""
        if not self._config:
            raise ValueError("No configuration loaded")

        # Check for missing mandatory values
        missing_keys = OmegaConf.missing_keys(self._config)
        if missing_keys:
            raise MissingMandatoryValue(
                f"Missing mandatory configuration keys: {missing_keys}"
            )

        # Validate model type if specified
        if hasattr(self._config, "model_type") and self.model_type:
            if self._config.model_type != self.model_type:
                print(
                    f"Warning: Model type mismatch. Expected {self.model_type}, got {self._config.model_type}"
                )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.

        :param key: Configuration key (supports dot notation)
        :type key: str
        :param default: Default value if key not found
        :type default: Any
        :return: Configuration value
        :rtype: Any
        """
        try:
            result = OmegaConf.select(self._config, key)
            if result is None:
                return default
            return result
        except (ConfigKeyError, KeyError):
            return default

    def set(self, key: str, value: Any):
        """Set a configuration value by key.

        :param key: Configuration key (supports dot notation)
        :type key: str
        :param value: Value to set
        :type value: Any
        """
        OmegaConf.update(self._config, key, value)

    def update(self, key: str, value: Any, merge: bool = True):
        """Update a configuration value by key.

        :param key: Configuration key (supports dot notation)
        :type key: str
        :param value: Value to update
        :type value: Any
        :param merge: Whether to merge dictionaries/lists
        :type merge: bool
        """
        OmegaConf.update(self._config, key, value, merge=merge)

    def save(self, path: str):
        """Save configuration to a YAML file.

        :param path: Path to save the configuration
        :type path: str
        """
        OmegaConf.save(self._config, path)

    def to_dict(self, resolve: bool = True) -> Dict[str, Any]:
        """Convert configuration to a dictionary.

        :param resolve: Whether to resolve interpolations
        :type resolve: bool
        :return: Configuration as dictionary
        :rtype: Dict[str, Any]
        """
        return OmegaConf.to_container(self._config, resolve=resolve)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to configuration values.

        :param name: Attribute name
        :type name: str
        :return: Configuration value
        :rtype: Any
        """
        try:
            return getattr(self._config, name)
        except AttributeError:
            raise AttributeError(f"Configuration has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration values.

        :param key: Configuration key
        :type key: str
        :return: Configuration value
        :rtype: Any
        """
        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting of configuration values.

        :param key: Configuration key
        :type key: str
        :param value: Value to set
        :type value: Any
        """
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if configuration contains a key.

        :param key: Configuration key
        :type key: str
        :return: True if key exists
        :rtype: bool
        """
        return key in self._config

    def __repr__(self) -> str:
        """String representation of the configuration.

        :return: String representation
        :rtype: str
        """
        return f"ConfigManager(model_type={self.model_type}, config_path={self.config_path})"

    def __str__(self) -> str:
        """String representation of the configuration.

        :return: YAML representation of configuration
        :rtype: str
        """
        return OmegaConf.to_yaml(self._config)


def create_config(
    model_type: str, config_path: Optional[str] = None, **overrides
) -> ConfigManager:
    """Create a configuration for a specific model type.

    :param model_type: Model type identifier
    :type model_type: str
    :param config_path: Path to configuration file
    :type config_path: Optional[str]
    :param overrides: Configuration overrides
    :type overrides: dict
    :return: Configuration manager instance
    :rtype: ConfigManager
    """
    config = ConfigManager(config_path=config_path, model_type=model_type)

    # Apply overrides
    for key, value in overrides.items():
        config.set(key, value)

    return config


def load_config_from_file(config_path: str) -> ConfigManager:
    """Load configuration from a YAML file.

    :param config_path: Path to configuration file
    :type config_path: str
    :return: Configuration manager instance
    :rtype: ConfigManager
    """
    return ConfigManager(config_path=config_path)


def merge_configs(*configs: ConfigManager) -> ConfigManager:
    """Merge multiple configuration managers.

    :param configs: Configuration managers to merge
    :type configs: ConfigManager
    :return: Merged configuration manager
    :rtype: ConfigManager
    """
    if not configs:
        raise ValueError("At least one configuration must be provided")

    # Start with the first configuration
    merged_config = configs[0]._config

    # Merge with remaining configurations
    for config in configs[1:]:
        merged_config = OmegaConf.merge(merged_config, config._config)

    # Create new ConfigManager with merged configuration
    result = ConfigManager()
    result._config = merged_config
    return result
