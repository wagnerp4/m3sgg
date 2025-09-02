"""Configuration utilities and helpers for VidSgg.

This module provides utility functions for configuration management, validation,
interpolation, and common configuration operations.

:author: VidSgg Team
:version: 0.1.0
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from omegaconf import OmegaConf, DictConfig, ListConfig
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue

from .config_structured import MODEL_CONFIGS


def register_custom_resolvers():
    """Register custom OmegaConf resolvers for VidSgg-specific functionality.
    
    This function registers resolvers for common operations like path resolution,
    environment variable access, and model-specific computations.
    """
    
    # Path resolver for relative paths
    def resolve_path(path: str, base_path: str = None) -> str:
        """Resolve a path relative to a base path or current working directory.
        
        :param path: Path to resolve
        :type path: str
        :param base_path: Base path for resolution
        :type base_path: str
        :return: Resolved absolute path
        :rtype: str
        """
        if base_path is None:
            base_path = os.getcwd()
        
        if os.path.isabs(path):
            return path
        
        return os.path.normpath(os.path.join(base_path, path))
    
    # Environment variable resolver with default
    def resolve_env(key: str, default: str = None) -> str:
        """Resolve environment variable with optional default.
        
        :param key: Environment variable key
        :type key: str
        :param default: Default value if key not found
        :type default: str
        :return: Environment variable value or default
        :rtype: str
        """
        return os.environ.get(key, default or "")
    
    # Model-specific resolver for dynamic parameter computation
    def resolve_model_param(model_type: str, param_name: str, default: Any = None) -> Any:
        """Resolve model-specific parameter with fallback to default.
        
        :param model_type: Model type identifier
        :type model_type: str
        :param param_name: Parameter name
        :type param_name: str
        :param default: Default value
        :type default: Any
        :return: Parameter value
        :rtype: Any
        """
        if model_type in MODEL_CONFIGS:
            config_class = MODEL_CONFIGS[model_type]
            if hasattr(config_class, param_name):
                return getattr(config_class, param_name)
        return default
    
    # Dataset size resolver for dynamic path computation
    def resolve_dataset_path(dataset: str, datasize: str, base_path: str = "data") -> str:
        """Resolve dataset path based on dataset name and size.
        
        :param dataset: Dataset name
        :type dataset: str
        :param datasize: Dataset size (mini/large)
        :type datasize: str
        :param base_path: Base data path
        :type base_path: str
        :return: Resolved dataset path
        :rtype: str
        """
        if datasize == "mini":
            if dataset == "action_genome":
                return os.path.join(base_path, "action_genome200")
            elif dataset == "EASG":
                return os.path.join(base_path, "EASG")
        else:
            return os.path.join(base_path, dataset)
        
        return os.path.join(base_path, dataset)
    
    # Register resolvers
    OmegaConf.register_new_resolver("path", resolve_path)
    OmegaConf.register_new_resolver("env", resolve_env)
    OmegaConf.register_new_resolver("model_param", resolve_model_param)
    OmegaConf.register_new_resolver("dataset_path", resolve_dataset_path)


def validate_config(config: DictConfig, model_type: str = None) -> List[str]:
    """Validate configuration and return list of validation errors.
    
    :param config: Configuration to validate
    :type config: DictConfig
    :param model_type: Model type for validation
    :type model_type: str
    :return: List of validation errors
    :rtype: List[str]
    """
    errors = []
    
    # Check for missing mandatory values
    missing_keys = OmegaConf.missing_keys(config)
    if missing_keys:
        errors.append(f"Missing mandatory values: {missing_keys}")
    
    # Validate model-specific parameters
    if model_type and model_type in MODEL_CONFIGS:
        config_class = MODEL_CONFIGS[model_type]
        
        # Check required fields for the model type
        required_fields = getattr(config_class, "__annotations__", {})
        for field_name, field_type in required_fields.items():
            if not OmegaConf.is_missing(config, field_name):
                try:
                    value = OmegaConf.select(config, field_name)
                    # Basic type validation
                    if not isinstance(value, field_type):
                        errors.append(f"Field '{field_name}' should be of type {field_type.__name__}, got {type(value).__name__}")
                except ConfigKeyError:
                    errors.append(f"Required field '{field_name}' not found for model type '{model_type}'")
    
    # Validate paths exist
    path_fields = ["data_path", "model_path", "save_path"]
    for field in path_fields:
        if not OmegaConf.is_missing(config, field):
            path_value = OmegaConf.select(config, field)
            if isinstance(path_value, str) and not os.path.exists(path_value):
                # Only warn for non-critical paths
                if field != "save_path":
                    errors.append(f"Path '{field}' does not exist: {path_value}")
    
    # Validate numeric ranges
    numeric_validations = {
        "lr": (1e-8, 1.0),
        "nepoch": (1, 1000),
        "enc_layer": (1, 20),
        "dec_layer": (1, 20),
    }
    
    for field, (min_val, max_val) in numeric_validations.items():
        if not OmegaConf.is_missing(config, field):
            value = OmegaConf.select(config, field)
            if isinstance(value, (int, float)):
                if not (min_val <= value <= max_val):
                    errors.append(f"Field '{field}' value {value} is outside valid range [{min_val}, {max_val}]")
    
    return errors


def create_experiment_config(
    base_config_path: str,
    experiment_name: str,
    overrides: Dict[str, Any] = None,
    output_dir: str = "output"
) -> DictConfig:
    """Create an experiment configuration by merging base config with overrides.
    
    :param base_config_path: Path to base configuration file
    :type base_config_path: str
    :param experiment_name: Name of the experiment
    :type experiment_name: str
    :param overrides: Configuration overrides
    :type overrides: Dict[str, Any]
    :param output_dir: Output directory for experiment
    :type output_dir: str
    :return: Merged configuration
    :rtype: DictConfig
    """
    # Load base configuration
    base_config = OmegaConf.load(base_config_path)
    
    # Create experiment-specific overrides
    experiment_overrides = {
        "experiment_name": experiment_name,
        "save_path": os.path.join(output_dir, experiment_name),
        "timestamp": "${oc.env:EXPERIMENT_TIMESTAMP,20240101_120000}"
    }
    
    # Add user overrides
    if overrides:
        experiment_overrides.update(overrides)
    
    # Merge configurations
    experiment_config = OmegaConf.merge(base_config, experiment_overrides)
    
    # Resolve interpolations
    OmegaConf.resolve(experiment_config)
    
    return experiment_config


def save_experiment_config(config: DictConfig, output_path: str):
    """Save experiment configuration to file.
    
    :param config: Configuration to save
    :type config: DictConfig
    :param output_path: Path to save configuration
    :type output_path: str
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save configuration
    OmegaConf.save(config, output_path)
    
    # Also save as a Python script for reproducibility
    script_path = output_path.replace(".yaml", "_script.py")
    with open(script_path, "w") as f:
        f.write("# Experiment configuration script\n")
        f.write("# Generated automatically from YAML configuration\n\n")
        f.write("from lib.config_omegaconf import create_config\n\n")
        f.write("config = create_config(\n")
        f.write(f'    model_type="{config.model_type}",\n')
        
        # Add key parameters
        key_params = ["mode", "dataset", "data_path", "lr", "nepoch"]
        for param in key_params:
            if not OmegaConf.is_missing(config, param):
                value = OmegaConf.select(config, param)
                if isinstance(value, str):
                    f.write(f'    {param}="{value}",\n')
                else:
                    f.write(f"    {param}={value},\n")
        
        f.write(")\n")


def load_config_with_interpolation(
    config_path: str,
    interpolation_vars: Dict[str, Any] = None
) -> DictConfig:
    """Load configuration with custom interpolation variables.
    
    :param config_path: Path to configuration file
    :type config_path: str
    :param interpolation_vars: Variables for interpolation
    :type interpolation_vars: Dict[str, Any]
    :return: Loaded configuration with resolved interpolations
    :rtype: DictConfig
    """
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Add interpolation variables
    if interpolation_vars:
        for key, value in interpolation_vars.items():
            OmegaConf.update(config, f"interpolation.{key}", value)
    
    # Resolve interpolations
    OmegaConf.resolve(config)
    
    return config


def create_config_from_dict(config_dict: Dict[str, Any], model_type: str = None) -> DictConfig:
    """Create configuration from dictionary with optional model type validation.
    
    :param config_dict: Configuration dictionary
    :type config_dict: Dict[str, Any]
    :param model_type: Model type for validation
    :type model_type: str
    :return: OmegaConf configuration
    :rtype: DictConfig
    """
    config = OmegaConf.create(config_dict)
    
    # Add model type if provided
    if model_type:
        OmegaConf.update(config, "model_type", model_type)
    
    # Validate configuration
    errors = validate_config(config, model_type)
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return config


def merge_config_files(*config_paths: str) -> DictConfig:
    """Merge multiple configuration files.
    
    :param config_paths: Paths to configuration files
    :type config_paths: str
    :return: Merged configuration
    :rtype: DictConfig
    """
    if not config_paths:
        raise ValueError("At least one configuration file must be provided")
    
    # Load all configurations
    configs = [OmegaConf.load(path) for path in config_paths]
    
    # Merge configurations
    merged_config = OmegaConf.merge(*configs)
    
    # Resolve interpolations
    OmegaConf.resolve(merged_config)
    
    return merged_config


def create_config_template(model_type: str, output_path: str = None) -> str:
    """Create a configuration template for a specific model type.
    
    :param model_type: Model type identifier
    :type model_type: str
    :param output_path: Path to save template
    :type output_path: str
    :return: YAML template content
    :rtype: str
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create structured configuration
    config_class = MODEL_CONFIGS[model_type]
    config = OmegaConf.structured(config_class)
    
    # Convert to YAML
    yaml_content = OmegaConf.to_yaml(config)
    
    # Add header comment
    header = f"# {model_type.upper()} Model Configuration Template\n"
    header += "# Generated automatically from structured configuration\n\n"
    
    yaml_content = header + yaml_content
    
    # Save to file if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(yaml_content)
    
    return yaml_content


def get_config_summary(config: DictConfig) -> Dict[str, Any]:
    """Get a summary of configuration parameters.
    
    :param config: Configuration to summarize
    :type config: DictConfig
    :return: Configuration summary
    :rtype: Dict[str, Any]
    """
    summary = {
        "model_type": OmegaConf.select(config, "model_type", default="unknown"),
        "dataset": OmegaConf.select(config, "dataset", default="unknown"),
        "mode": OmegaConf.select(config, "mode", default="unknown"),
        "learning_rate": OmegaConf.select(config, "lr", default="unknown"),
        "epochs": OmegaConf.select(config, "nepoch", default="unknown"),
        "data_path": OmegaConf.select(config, "data_path", default="unknown"),
        "save_path": OmegaConf.select(config, "save_path", default="unknown"),
    }
    
    # Add model-specific parameters
    model_type = summary["model_type"]
    if model_type in MODEL_CONFIGS:
        config_class = MODEL_CONFIGS[model_type]
        annotations = getattr(config_class, "__annotations__", {})
        
        for field_name in annotations:
            if field_name not in summary:
                value = OmegaConf.select(config, field_name, default="unknown")
                summary[field_name] = value
    
    return summary


# Initialize custom resolvers when module is imported
register_custom_resolvers()
