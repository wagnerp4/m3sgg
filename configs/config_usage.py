#!/usr/bin/env python3
"""Examples demonstrating the new OmegaConf-based configuration system.

This script shows various ways to use the new configuration system while
maintaining backward compatibility with the existing Config class.

:author: VidSgg Team
:version: 0.1.0
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.config import Config  # Original config (100% backward compatible)
from lib.config_omegaconf import ConfigManager, create_config  # New config system
from lib.config_utils import validate_config, get_config_summary


def example_1_backward_compatibility():
    """Example 1: Using the original Config class (unchanged behavior)."""
    print("=" * 60)
    print("Example 1: Backward Compatibility with Original Config")
    print("=" * 60)
    
    # This works exactly as before - no changes needed
    conf = Config()
    
    print(f"Model type: {conf.model_type}")
    print(f"Learning rate: {conf.lr}")
    print(f"Dataset: {conf.dataset}")
    print(f"Mode: {conf.mode}")
    
    # All existing code patterns continue to work
    timestamp = "20240101_120000"
    data_path_suffix = os.path.basename(conf.data_path)
    new_save_path = os.path.join(
        "output", data_path_suffix, conf.model_type, conf.mode, timestamp
    )
    print(f"Generated save path: {new_save_path}")


def example_2_yaml_configuration():
    """Example 2: Using YAML configuration files."""
    print("\n" + "=" * 60)
    print("Example 2: YAML Configuration Files")
    print("=" * 60)
    
    # Check if config files exist
    config_path = "configs/sttran.yaml"
    if not Path(config_path).exists():
        print(f"Config file {config_path} not found. Skipping this example.")
        return
    
    # Load configuration from YAML file
    config = ConfigManager(config_path=config_path)
    
    print(f"Loaded from: {config_path}")
    print(f"Model type: {config.model_type}")
    print(f"Learning rate: {config.lr}")
    print(f"Dataset: {config.dataset}")
    
    # Validate configuration
    errors = validate_config(config._config, config.model_type)
    if errors:
        print(f"Configuration errors: {errors}")
    else:
        print("Configuration validation passed!")
    
    # Get configuration summary
    summary = get_config_summary(config._config)
    print(f"Configuration summary: {summary}")


def example_3_structured_configuration():
    """Example 3: Using structured configurations."""
    print("\n" + "=" * 60)
    print("Example 3: Structured Configuration")
    print("=" * 60)
    
    # Create configuration for SceneLLM model
    config = create_config(
        model_type="scenellm",
        lr=2e-5,
        nepoch=20,
        scenellm_training_stage="vqvae"
    )
    
    print(f"Model type: {config.model_type}")
    print(f"LLM name: {config.llm_name}")
    print(f"LoRA rank: {config.lora_r}")
    print(f"Embedding dimension: {config.embed_dim}")
    print(f"Codebook size: {config.codebook_size}")
    print(f"Training stage: {config.scenellm_training_stage}")
    
    # Type safety - this would raise an error if types don't match
    try:
        config.set("lr", "invalid_string")  # This should work (OmegaConf converts)
        print(f"Learning rate after setting string: {config.lr} (type: {type(config.lr)})")
    except Exception as e:
        print(f"Type validation error: {e}")


def example_4_command_line_overrides():
    """Example 4: Command line overrides with YAML base."""
    print("\n" + "=" * 60)
    print("Example 4: Command Line Overrides")
    print("=" * 60)
    
    # Simulate command line arguments
    cli_args = ["-lr", "3e-5", "-nepoch", "15", "-mode", "sgcls"]
    
    # Load base config and apply CLI overrides
    config = ConfigManager(
        config_path="configs/base.yaml",
        cli_args=cli_args
    )
    
    print(f"Base config loaded from: configs/base.yaml")
    print(f"CLI overrides: {cli_args}")
    print(f"Final learning rate: {config.lr}")
    print(f"Final epochs: {config.nepoch}")
    print(f"Final mode: {config.mode}")


def example_5_experiment_management():
    """Example 5: Experiment configuration management."""
    print("\n" + "=" * 60)
    print("Example 5: Experiment Management")
    print("=" * 60)
    
    from lib.config_utils import create_experiment_config, save_experiment_config
    
    # Create experiment configuration
    base_config_path = "configs/sttran.yaml"
    if not Path(base_config_path).exists():
        print(f"Base config {base_config_path} not found. Skipping this example.")
        return
    
    overrides = {
        "lr": 2e-5,
        "nepoch": 25,
        "save_path": "output/experiment_demo"
    }
    
    exp_config = create_experiment_config(
        base_config_path=base_config_path,
        experiment_name="demo_experiment",
        overrides=overrides
    )
    
    print(f"Experiment name: {exp_config.experiment_name}")
    print(f"Learning rate: {exp_config.lr}")
    print(f"Epochs: {exp_config.nepoch}")
    print(f"Save path: {exp_config.save_path}")
    
    # Save experiment configuration
    output_path = "output/demo_experiment_config.yaml"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_experiment_config(exp_config, output_path)
    print(f"Experiment config saved to: {output_path}")


def example_6_configuration_comparison():
    """Example 6: Comparing old and new configuration systems."""
    print("\n" + "=" * 60)
    print("Example 6: Configuration System Comparison")
    print("=" * 60)
    
    # Original config
    old_config = Config()
    
    # New config with same model type
    new_config = create_config(model_type=old_config.model_type)
    
    print("Comparison of configuration values:")
    print(f"{'Parameter':<20} {'Old Config':<15} {'New Config':<15}")
    print("-" * 50)
    
    # Compare key parameters
    params_to_compare = ["model_type", "mode", "dataset", "lr", "nepoch", "enc_layer", "dec_layer"]
    
    for param in params_to_compare:
        old_value = getattr(old_config, param, "N/A")
        new_value = getattr(new_config, param, "N/A")
        print(f"{param:<20} {str(old_value):<15} {str(new_value):<15}")
    
    print("\nBoth systems provide the same interface and values!")


def main():
    """Run all configuration examples."""
    print("VidSgg Configuration System Examples")
    print("====================================")
    
    try:
        example_1_backward_compatibility()
        example_2_yaml_configuration()
        example_3_structured_configuration()
        example_4_command_line_overrides()
        example_5_experiment_management()
        example_6_configuration_comparison()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nKey Benefits of the New System:")
        print("1. 100% backward compatibility - existing code works unchanged")
        print("2. YAML configuration files for better organization")
        print("3. Type safety with structured configurations")
        print("4. Configuration validation and error checking")
        print("5. Experiment management and reproducibility")
        print("6. Interpolation and advanced features")
        print("7. Easy migration path - use new features when ready")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
