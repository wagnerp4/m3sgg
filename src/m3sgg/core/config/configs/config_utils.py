"""Configuration utility functions for M3SGG.

This module provides utility functions for configuration management,
including validation, creation, merging, and template generation.

:author: M3SGG Team
:version: 0.1.0
"""

import os
from typing import Any, Dict, List, Optional, Union
from omegaconf import OmegaConf, DictConfig

from .config_structured import get_config_class


def validate_config(config: Union[DictConfig, Dict[str, Any]], model_type: str) -> List[str]:
    """Validate configuration against model requirements.
    
    :param config: Configuration to validate
    :type config: Union[DictConfig, Dict[str, Any]]
    :param model_type: Model type for validation
    :type model_type: str
    :return: List of validation errors
    :rtype: List[str]
    """
    errors = []
    
    try:
        get_config_class(model_type)
    except ValueError as e:
        errors.append(str(e))
        return errors
    
    # Convert to OmegaConf if needed
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    
    # Basic validation rules
    required_fields = ["model_type", "mode", "lr", "nepoch"]
    
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate learning rate
    if "lr" in config:
        lr = config.lr
        if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
            errors.append(f"Invalid learning rate: {lr}. Must be a positive number <= 1")
    
    # Validate epochs
    if "nepoch" in config:
        nepoch = config.nepoch
        if not isinstance(nepoch, (int, float)) or nepoch <= 0:
            errors.append(f"Invalid number of epochs: {nepoch}. Must be a positive number")
    
    # Validate mode
    if "mode" in config:
        mode = config.mode
        valid_modes = ["predcls", "sgcls", "sgdet"]
        if mode not in valid_modes:
            errors.append(f"Invalid mode: {mode}. Must be one of {valid_modes}")
    
    return errors


def create_config_from_dict(config_dict: Dict[str, Any], model_type: str) -> Any:
    """Create a configuration object from a dictionary.
    
    :param config_dict: Configuration dictionary
    :type config_dict: Dict[str, Any]
    :param model_type: Model type identifier
    :type model_type: str
    :return: Configuration object
    :rtype: Any
    """
    config_class = get_config_class(model_type)
    
    # Create configuration object with provided values
    config = config_class()
    
    # Update with provided values
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def create_experiment_config(
    base_config_path: str,
    experiment_name: str,
    overrides: Optional[Dict[str, Any]] = None
) -> Any:
    """Create an experiment configuration from a base config.
    
    :param base_config_path: Path to base configuration file
    :type base_config_path: str
    :param experiment_name: Name of the experiment
    :type experiment_name: str
    :param overrides: Configuration overrides
    :type overrides: Optional[Dict[str, Any]]
    :return: Experiment configuration
    :rtype: Any
    """
    # Load base configuration
    base_config = OmegaConf.load(base_config_path)
    
    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            base_config[key] = value
    
    # Add experiment name
    base_config["experiment_name"] = experiment_name
    
    # Update save path to include experiment name
    save_path = base_config.get("save_path", "output")
    base_config["save_path"] = os.path.join(save_path, experiment_name)
    
    # Get model type and create structured config
    model_type = base_config.get("model_type", "sttran")
    config_class = get_config_class(model_type)
    
    # Create configuration object
    config = config_class()
    
    # Update with merged values
    for key, value in base_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Add custom attributes dynamically
            setattr(config, key, value)
    
    return config


def merge_config_files(*config_paths: str) -> Any:
    """Merge multiple configuration files.
    
    :param config_paths: Paths to configuration files
    :type config_paths: str
    :return: Merged configuration
    :rtype: Any
    """
    merged_config = None
    
    for config_path in config_paths:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config = OmegaConf.load(config_path)
        
        if merged_config is None:
            merged_config = config
        else:
            merged_config = OmegaConf.merge(merged_config, config)
    
    # Get model type and create structured config
    model_type = merged_config.get("model_type", "sttran")
    config_class = get_config_class(model_type)
    
    # Create configuration object
    config = config_class()
    
    # Update with merged values
    for key, value in merged_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def create_config_template(model_type: str) -> str:
    """Create a configuration template for a model type.
    
    :param model_type: Model type identifier
    :type model_type: str
    :return: Configuration template as string
    :rtype: str
    """
    try:
        config_class = get_config_class(model_type)
    except ValueError:
        return f"# Unsupported model type: {model_type}"
    
    # Create a default configuration
    config = config_class()
    
    # Generate template content
    template_lines = [
        f"# {model_type.upper()} Model Configuration Template",
        "# Generated automatically - modify as needed",
        "",
        f"model_type: {model_type}",
    ]
    
    # Add configuration fields
    for field_name, field_value in config.__dict__.items():
        if field_name == "model_type":
            continue
        
        if isinstance(field_value, str):
            # Don't quote simple values that don't need quotes
            if field_value in ["predcls", "sgcls", "sgdet", "large", "mini", "adamw", "adam", "sgd", "cuda:0", "cpu"]:
                template_lines.append(f"{field_name}: {field_value}")
            else:
                template_lines.append(f"{field_name}: \"{field_value}\"")
        elif field_value is None:
            template_lines.append(f"{field_name}: null")
        else:
            template_lines.append(f"{field_name}: {field_value}")
    
    return "\n".join(template_lines)


def get_config_summary(config: Union[DictConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """Get a summary of the configuration.
    
    :param config: Configuration to summarize
    :type config: Union[DictConfig, Dict[str, Any]]
    :return: Configuration summary
    :rtype: Dict[str, Any]
    """
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    
    summary = {
        "model_type": config.get("model_type", "unknown"),
        "dataset": config.get("dataset", "unknown"),
        "mode": config.get("mode", "unknown"),
        "learning_rate": config.get("lr", "unknown"),
        "epochs": config.get("nepoch", "unknown"),
        "device": config.get("device", "unknown"),
        "batch_size": config.get("batch_size", "unknown"),
    }
    
    return summary


def load_config_with_interpolation(config_path: str) -> DictConfig:
    """Load configuration with interpolation support.
    
    :param config_path: Path to configuration file
    :type config_path: str
    :return: Configuration with interpolated values
    :rtype: DictConfig
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Resolve interpolations
    OmegaConf.resolve(config)
    
    return config


def save_experiment_config(config: Any, save_path: str) -> None:
    """Save experiment configuration to file.
    
    :param config: Configuration to save
    :type config: Any
    :param save_path: Path to save configuration
    :type save_path: str
    """
    # Convert to dictionary if needed
    if hasattr(config, "__dict__"):
        config_dict = config.__dict__
    elif isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config
    
    # Create OmegaConf config
    omega_config = OmegaConf.create(config_dict)
    
    # Save to file
    OmegaConf.save(omega_config, save_path)
