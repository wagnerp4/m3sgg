"""Unified configuration interface for M3SGG.

This module provides a unified interface that can work with both legacy
and modern configuration systems, allowing for gradual migration.

:author: M3SGG Team
:version: 0.1.0
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .legacy import LegacyConfig
from .modern import ConfigManager


class UnifiedConfig:
    """Unified configuration interface supporting both legacy and modern systems.

    This class provides a unified interface that can work with both the legacy
    argparse-based configuration system and the modern OmegaConf-based system.
    It automatically detects which system to use based on the configuration
    and provides a consistent interface.

    :param config_path: Path to configuration file
    :type config_path: Optional[str]
    :param model_type: Model type for structured configuration
    :type model_type: Optional[str]
    :param use_modern: Force use of modern configuration system
    :type use_modern: bool
    :param cli_args: Command-line arguments
    :type cli_args: Optional[List[str]]
    :param overrides: Configuration overrides
    :type overrides: Optional[Dict[str, Any]]
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_type: Optional[str] = None,
        use_modern: bool = True,
        cli_args: Optional[List[str]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the unified configuration.

        :param config_path: Path to configuration file
        :type config_path: Optional[str]
        :param model_type: Model type for structured configuration
        :type model_type: Optional[str]
        :param use_modern: Force use of modern configuration system
        :type use_modern: bool
        :param cli_args: Command-line arguments
        :type cli_args: Optional[List[str]]
        :param overrides: Configuration overrides
        :type overrides: Optional[Dict[str, Any]]
        """
        self.config_path = config_path
        self.model_type = model_type
        self.use_modern = use_modern
        self.cli_args = cli_args or sys.argv[1:]
        self.overrides = overrides or {}

        # Determine which configuration system to use
        # For now, use legacy by default to ensure compatibility
        self._use_modern = False

        if self._use_modern:
            self._config = ConfigManager(
                config_path=config_path,
                model_type=model_type,
                cli_args=cli_args,
                overrides=overrides,
            )
        else:
            self._config = LegacyConfig(
                config_path=config_path,
                overrides=overrides,
            )

    def _should_use_modern(self) -> bool:
        """Determine whether to use the modern configuration system.

        :return: True if modern system should be used
        :rtype: bool
        """
        # Force modern if explicitly requested
        if self.use_modern:
            return True

        # Use modern if config_path points to a YAML file
        if self.config_path and self.config_path.endswith((".yaml", ".yml")):
            return True

        # Use modern if model_type is specified (structured configs)
        if self.model_type:
            return True

        # Use modern if overrides are provided (better for programmatic use)
        if self.overrides:
            return True

        # Default to modern for new code
        return True

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.

        :param key: Configuration key (supports dot notation for modern)
        :type key: str
        :param default: Default value if key not found
        :type default: Any
        :return: Configuration value
        :rtype: Any
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set a configuration value by key.

        :param key: Configuration key
        :type key: str
        :param value: Value to set
        :type value: Any
        """
        self._config.set(key, value)

    def update(self, key: str, value: Any, merge: bool = True):
        """Update a configuration value by key.

        :param key: Configuration key
        :type key: str
        :param value: Value to update
        :type value: Any
        :param merge: Whether to merge dictionaries/lists (modern only)
        :type merge: bool
        """
        if hasattr(self._config, "update"):
            self._config.update(key, value, merge=merge)
        else:
            self._config.set(key, value)

    def save(self, path: str):
        """Save configuration to a file.

        :param path: Path to save the configuration
        :type path: str
        """
        if hasattr(self._config, "save"):
            self._config.save(path)
        else:
            # For legacy config, save as YAML
            import yaml

            with open(path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_dict(self, resolve: bool = True) -> Dict[str, Any]:
        """Convert configuration to a dictionary.

        :param resolve: Whether to resolve interpolations (modern only)
        :type resolve: bool
        :return: Configuration as dictionary
        :rtype: Dict[str, Any]
        """
        if hasattr(self._config, "to_dict"):
            return self._config.to_dict(resolve=resolve)
        else:
            return self._config.to_dict()

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to configuration values.

        :param name: Attribute name
        :type name: str
        :return: Configuration value
        :rtype: Any
        """
        return getattr(self._config, name)

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
        return f"UnifiedConfig(model_type={self.model_type}, config_path={self.config_path}, modern={self._use_modern})"

    def __str__(self) -> str:
        """String representation of the configuration.

        :return: String representation
        :rtype: str
        """
        if hasattr(self._config, "__str__"):
            return str(self._config)
        else:
            return repr(self._config)

    @property
    def is_modern(self) -> bool:
        """Check if using modern configuration system.

        :return: True if using modern system
        :rtype: bool
        """
        return self._use_modern

    @property
    def is_legacy(self) -> bool:
        """Check if using legacy configuration system.

        :return: True if using legacy system
        :rtype: bool
        """
        return not self._use_modern


# Convenience functions for backward compatibility
def create_config(
    model_type: str, config_path: Optional[str] = None, **overrides
) -> UnifiedConfig:
    """Create a configuration for a specific model type.

    :param model_type: Model type identifier
    :type model_type: str
    :param config_path: Path to configuration file
    :type config_path: Optional[str]
    :param overrides: Configuration overrides
    :type overrides: dict
    :return: Unified configuration instance
    :rtype: UnifiedConfig
    """
    return UnifiedConfig(
        config_path=config_path,
        model_type=model_type,
        overrides=overrides,
    )


def load_config_from_file(config_path: str) -> UnifiedConfig:
    """Load configuration from a file.

    :param config_path: Path to configuration file
    :type config_path: str
    :return: Unified configuration instance
    :rtype: UnifiedConfig
    """
    return UnifiedConfig(config_path=config_path)


# Backward compatibility alias
Config = UnifiedConfig
