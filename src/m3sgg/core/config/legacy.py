"""Legacy configuration system for backward compatibility.

This module provides the original argparse-based configuration system
for backward compatibility with existing code that depends on the old
Config class interface.

:author: M3SGG Team
:version: 0.1.0
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add the parent directory to the path to import the original config
sys.path.insert(0, str(Path(__file__).parent))

from .config import Config as OriginalConfig


class LegacyConfig(OriginalConfig):
    """Legacy configuration class for backward compatibility.

    This class wraps the original Config class to provide backward
    compatibility while allowing for future migration to the modern
    configuration system.

    :param config_path: Path to configuration file
    :type config_path: Optional[str]
    :param overrides: Configuration overrides
    :type overrides: Optional[Dict[str, Any]]
    """

    def __init__(
        self, 
        config_path: Optional[str] = None, 
        overrides: Optional[Dict[str, Any]] = None
    ):
        """Initialize the legacy configuration.

        :param config_path: Path to configuration file
        :type config_path: Optional[str]
        :param overrides: Configuration overrides
        :type overrides: Optional[Dict[str, Any]]
        """
        super().__init__()
        
        # Apply overrides if provided
        if overrides:
            for key, value in overrides.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    # Add new attributes dynamically
                    setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        :return: Configuration as dictionary
        :rtype: Dict[str, Any]
        """
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_') and not callable(value)}

    def update(self, **kwargs):
        """Update configuration values.

        :param kwargs: Configuration updates
        :type kwargs: Any
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default.

        :param key: Configuration key
        :type key: str
        :param default: Default value if key not found
        :type default: Any
        :return: Configuration value
        :rtype: Any
        """
        return getattr(self, key, default)

    def set(self, key: str, value: Any):
        """Set configuration value.

        :param key: Configuration key
        :type key: str
        :param value: Value to set
        :type value: Any
        """
        setattr(self, key, value)
