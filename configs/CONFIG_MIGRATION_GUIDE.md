# Configuration System Migration Guide

This guide explains how to migrate from the old configuration system to the new OmegaConf-based system.

## Overview

The new configuration system provides:
- **Type safety**: Runtime validation with structured configs
- **YAML support**: Configuration files with interpolation
- **Better organization**: Model-specific configurations
- **Backward compatibility**: Existing code continues to work
- **Enhanced features**: Validation, merging, and utilities

## Quick Start

### Using YAML Configuration Files

Create a configuration file (e.g., `configs/my_experiment.yaml`):

```yaml
# @package _global_
defaults:
  - base
  - sttran

# Experiment-specific overrides
lr: 2e-5
nepoch: 20
save_path: "output/my_experiment"
```

Load it in your training script:

```python
from lib.config_omegaconf import ConfigManager

# Load from YAML file
config = ConfigManager(config_path="configs/my_experiment.yaml")

# Use configuration
print(f"Model: {config.model_type}")
print(f"Learning rate: {config.lr}")
```

### Using Structured Configurations

```python
from lib.config_omegaconf import create_config

# Create configuration for specific model
config = create_config(
    model_type="scenellm",
    lr=2e-5,
    nepoch=20,
    scenellm_training_stage="vqvae"
)

# Access configuration
print(f"LLM: {config.llm_name}")
print(f"LoRA rank: {config.lora_r}")
```

## Migration Steps

### 1. Update Imports

**Old:**
```python
from lib.config import Config
```

**New:**
```python
from lib.config_omegaconf import Config  # Backward compatible
# OR
from lib.config_omegaconf import ConfigManager  # New interface
```

### 2. Configuration Loading

**Old:**
```python
conf = Config()
```

**New (backward compatible):**
```python
conf = Config()  # Still works!
```

**New (recommended):**
```python
# From YAML file
conf = ConfigManager(config_path="configs/my_config.yaml")

# From structured config
conf = create_config(model_type="sttran", lr=2e-5)

# With command line overrides
conf = ConfigManager(
    config_path="configs/base.yaml",
    cli_args=["-lr", "2e-5", "-nepoch", "20"]
)
```

### 3. Configuration Access

**Old and New (both work):**
```python
# Attribute access
print(conf.model_type)
print(conf.lr)

# Dictionary access
print(conf["model_type"])
print(conf["lr"])

# Get with default
value = conf.get("nonexistent_key", "default_value")
```

**New features:**
```python
# Set values
conf.set("custom_param", "value")

# Update with merging
conf.update("nested.param", {"new": "value"}, merge=True)

# Check existence
if "param" in conf:
    print("Parameter exists")
```

### 4. Configuration Validation

**New:**
```python
from lib.config_utils import validate_config

errors = validate_config(conf._config, conf.model_type)
if errors:
    print(f"Configuration errors: {errors}")
```

### 5. Experiment Management

**New:**
```python
from lib.config_utils import create_experiment_config, save_experiment_config

# Create experiment configuration
exp_config = create_experiment_config(
    base_config_path="configs/sttran.yaml",
    experiment_name="my_experiment",
    overrides={"lr": 2e-5, "nepoch": 20}
)

# Save experiment configuration
save_experiment_config(exp_config, "output/my_experiment/config.yaml")
```

## Configuration File Structure

### Base Configuration (`configs/base.yaml`)
```yaml
# Common parameters for all models
mode: "predcls"
save_path: "output"
dataset: "action_genome"
data_path: "data/action_genome"
optimizer: "adamw"
lr: 1e-5
nepoch: 10
```

### Model-Specific Configuration (`configs/sttran.yaml`)
```yaml
# @package _global_
defaults:
  - base

model_type: "sttran"
use_matcher: false
```

### Experiment Configuration (`configs/experiments/my_experiment.yaml`)
```yaml
# @package _global_
defaults:
  - sttran

# Experiment-specific overrides
lr: 2e-5
nepoch: 20
save_path: "output/my_experiment"
```

## Advanced Features

### Interpolation

```yaml
# Use variables in configuration
base_path: "data"
dataset_path: "${base_path}/action_genome"
model_path: "${dataset_path}/models/checkpoint.pth"

# Environment variables
api_key: "${oc.env:API_KEY}"
timestamp: "${oc.env:EXPERIMENT_TIMESTAMP,${now:%Y%m%d_%H%M%S}}"
```

### Custom Resolvers

```yaml
# Path resolution
data_path: "${path:data/action_genome,${oc.env:PROJECT_ROOT}}"

# Model-specific parameters
embed_dim: "${model_param:scenellm,embed_dim,1024}"

# Dataset path resolution
dataset_path: "${dataset_path:action_genome,large,data}"
```

### Configuration Merging

```python
from lib.config_omegaconf import merge_configs

# Merge multiple configurations
config1 = ConfigManager(config_path="configs/base.yaml")
config2 = ConfigManager(config_path="configs/sttran.yaml")
config3 = ConfigManager(config_path="configs/experiments/my_exp.yaml")

merged_config = merge_configs(config1, config2, config3)
```

## Best Practices

### 1. Use YAML Files for Experiments
- Create separate YAML files for different experiments
- Use the `defaults` mechanism to inherit from base configurations
- Override only the parameters that differ

### 2. Validate Configurations
- Always validate configurations before training
- Use structured configs for type safety
- Check for missing mandatory values

### 3. Organize Configuration Files
```
configs/
├── base.yaml              # Common parameters
├── sttran.yaml           # STTRAN-specific
├── scenellm.yaml         # SceneLLM-specific
├── oed.yaml              # OED-specific
└── experiments/
    ├── small_dataset.yaml
    ├── scenellm_vqvae.yaml
    └── my_experiment.yaml
```

### 4. Use Interpolation for Flexibility
- Use environment variables for sensitive data
- Use interpolation for dynamic paths
- Use resolvers for computed values

### 5. Save Experiment Configurations
- Always save the configuration used for each experiment
- Include both YAML and Python script formats
- Document any manual overrides

## Troubleshooting

### Common Issues

1. **Missing mandatory values**
   ```python
   # Check for missing values
   missing = OmegaConf.missing_keys(config._config)
   if missing:
       print(f"Missing: {missing}")
   ```

2. **Type validation errors**
   ```python
   # Use structured configs for type safety
   config = create_config(model_type="sttran", lr="invalid")  # Will raise error
   ```

3. **Configuration not found**
   ```python
   # Check if config file exists
   if not Path("configs/my_config.yaml").exists():
       print("Configuration file not found")
   ```

### Debugging

```python
# Print configuration summary
from lib.config_utils import get_config_summary
summary = get_config_summary(config._config)
print(summary)

# Print full configuration
print(OmegaConf.to_yaml(config._config))

# Check interpolations
OmegaConf.resolve(config._config)
```

## Examples

### Training Script with New Configuration

```python
#!/usr/bin/env python3
"""Example training script using new configuration system."""

import os
import logging
from pathlib import Path
from lib.config_omegaconf import ConfigManager
from lib.config_utils import validate_config, save_experiment_config

def main():
    # Load configuration
    config = ConfigManager(config_path="configs/experiments/my_experiment.yaml")
    
    # Validate configuration
    errors = validate_config(config._config, config.model_type)
    if errors:
        print(f"Configuration errors: {errors}")
        return
    
    # Set up logging
    os.makedirs(config.save_path, exist_ok=True)
    log_file = os.path.join(config.save_path, "logfile.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    
    # Log configuration
    logger.info("Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
    
    # Save experiment configuration
    config_save_path = os.path.join(config.save_path, "config.yaml")
    save_experiment_config(config._config, config_save_path)
    
    # Training code here...
    logger.info(f"Starting training with model: {config.model_type}")
    logger.info(f"Learning rate: {config.lr}")
    logger.info(f"Epochs: {config.nepoch}")

if __name__ == "__main__":
    main()
```

### Command Line Usage

```bash
# Use YAML configuration with command line overrides
python train.py --config configs/sttran.yaml --lr 2e-5 --nepoch 20

# Use structured configuration
python train.py --model scenellm --scenellm_training_stage vqvae --lr 1e-4

# Use experiment configuration
python train.py --config configs/experiments/scenellm_vqvae.yaml
```

This migration guide provides a comprehensive overview of the new configuration system while maintaining backward compatibility with existing code.
