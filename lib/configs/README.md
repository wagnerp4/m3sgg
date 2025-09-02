# VidSgg Configuration System

This document provides a comprehensive overview of the VidSgg configuration system, which supports both the original configuration approach and a new OmegaConf-based system with **100% backward compatibility**.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Files](#configuration-files)
3. [Key Features](#key-features)
4. [Migration Guide](#migration-guide)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)
9. [File Structure](#file-structure)
10. [Integration Summary](#integration-summary)

## Quick Start

### Using the Original System (100% Backward Compatible)

```python
from lib.config import Config

# This works exactly as before - no changes needed
conf = Config()
print(f"Model: {conf.model_type}")
print(f"Learning rate: {conf.lr}")
```

### Using the New System (Optional)

```python
from lib.configs.config_omegaconf import ConfigManager, create_config

# From YAML file
config = ConfigManager(config_path="lib/configs/sttran.yaml")

# From structured config
config = create_config(model_type="scenellm", lr=2e-5)
```

## Configuration Files

The new system includes pre-configured YAML files in the `lib/configs/` directory:

- `lib/configs/base.yaml` - Common parameters for all models
- `lib/configs/sttran.yaml` - STTRAN-specific configuration
- `lib/configs/stket.yaml` - STKET-specific configuration
- `lib/configs/tempura.yaml` - Tempura-specific configuration
- `lib/configs/easg.yaml` - EASG-specific configuration
- `lib/configs/scenellm.yaml` - SceneLLM-specific configuration
- `lib/configs/oed.yaml` - OED-specific configuration
- `lib/configs/experiments/` - Example experiment configurations

## Key Features

### 1. Backward Compatibility
- All existing code continues to work unchanged
- Original `Config` class remains fully functional
- No breaking changes to existing scripts

### 2. YAML Configuration Files
- Human-readable configuration files
- Easy to version control and share
- Support for configuration inheritance

### 3. Type Safety
- Structured configuration classes with type validation
- Runtime type checking
- Better error messages for invalid configurations

### 4. Configuration Validation
- Automatic validation of configuration parameters
- Range checking for numeric values
- Path existence validation

### 5. Experiment Management
- Easy creation of experiment-specific configurations
- Automatic saving of experiment configurations
- Reproducibility support

### 6. Advanced Features
- Variable interpolation
- Environment variable support
- Configuration merging
- Custom resolvers

## Migration Guide

### Overview

The new configuration system provides:
- **Type safety**: Runtime validation with structured configs
- **YAML support**: Configuration files with interpolation
- **Better organization**: Model-specific configurations
- **Backward compatibility**: Existing code continues to work
- **Enhanced features**: Validation, merging, and utilities

### Migration Steps

#### 1. Update Imports

**Old:**
```python
from lib.config import Config
```

**New:**
```python
from lib.configs.config_omegaconf import Config  # Backward compatible
# OR
from lib.configs.config_omegaconf import ConfigManager  # New interface
```

#### 2. Configuration Loading

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
conf = ConfigManager(config_path="lib/configs/my_config.yaml")

# From structured config
conf = create_config(model_type="sttran", lr=2e-5)

# With command line overrides
conf = ConfigManager(
    config_path="lib/configs/base.yaml",
    cli_args=["-lr", "2e-5", "-nepoch", "20"]
)
```

#### 3. Configuration Access

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

#### 4. Configuration Validation

**New:**
```python
from lib.configs.config_utils import validate_config

errors = validate_config(conf._config, conf.model_type)
if errors:
    print(f"Configuration errors: {errors}")
```

#### 5. Experiment Management

**New:**
```python
from lib.configs.config_utils import create_experiment_config, save_experiment_config

# Create experiment configuration
exp_config = create_experiment_config(
    base_config_path="lib/configs/sttran.yaml",
    experiment_name="my_experiment",
    overrides={"lr": 2e-5, "nepoch": 20}
)

# Save experiment configuration
save_experiment_config(exp_config, "output/my_experiment/config.yaml")
```

### Migration Strategy

#### Phase 1: No Changes Required
- Continue using existing code as-is
- All current scripts work without modification
- Original `Config` class remains unchanged

#### Phase 2: Optional Adoption
- Start using YAML files for new experiments
- Gradually adopt new features as needed
- Mix old and new approaches as desired

#### Phase 3: Full Migration (Future)
- When ready, migrate to new system entirely
- Remove old configuration system
- Leverage all advanced features

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
from lib.configs.config_omegaconf import merge_configs

# Merge multiple configurations
config1 = ConfigManager(config_path="lib/configs/base.yaml")
config2 = ConfigManager(config_path="lib/configs/sttran.yaml")
config3 = ConfigManager(config_path="lib/configs/experiments/my_exp.yaml")

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
lib/configs/
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

## Examples

### Training Script with New Configuration

```python
#!/usr/bin/env python3
"""Example training script using new configuration system."""

import os
import logging
from pathlib import Path
from lib.configs.config_omegaconf import ConfigManager
from lib.configs.config_utils import validate_config, save_experiment_config

def main():
    # Load configuration
    config = ConfigManager(config_path="lib/configs/experiments/my_experiment.yaml")
    
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
python train.py --config lib/configs/sttran.yaml --lr 2e-5 --nepoch 20

# Use structured configuration
python train.py --model scenellm --scenellm_training_stage vqvae --lr 1e-4

# Use experiment configuration
python train.py --config lib/configs/experiments/scenellm_vqvae.yaml
```

### Configuration File Examples

#### Base Configuration (`lib/configs/base.yaml`)
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

#### Model-Specific Configuration (`lib/configs/sttran.yaml`)
```yaml
# @package _global_
defaults:
  - base

model_type: "sttran"
use_matcher: false
```

#### Experiment Configuration (`lib/configs/experiments/my_experiment.yaml`)
```yaml
# @package _global_
defaults:
  - sttran

# Experiment-specific overrides
lr: 2e-5
nepoch: 20
save_path: "output/my_experiment"
```

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
   if not Path("lib/configs/my_config.yaml").exists():
       print("Configuration file not found")
   ```

### Debugging

```python
# Print configuration summary
from lib.configs.config_utils import get_config_summary
summary = get_config_summary(config._config)
print(summary)

# Print full configuration
print(OmegaConf.to_yaml(config._config))

# Check interpolations
OmegaConf.resolve(config._config)
```

## File Structure

```
lib/configs/
├── base.yaml              # Common parameters
├── sttran.yaml           # STTRAN model config
├── stket.yaml            # STKET model config
├── tempura.yaml          # Tempura model config
├── easg.yaml             # EASG model config
├── scenellm.yaml         # SceneLLM model config
├── oed.yaml              # OED model config
├── experiments/
│   ├── action_genome_small.yaml
│   └── scenellm_vqvae.yaml
├── usage.py              # Usage examples
└── README.md             # This documentation

lib/
├── config.py             # Original config (unchanged)
└── configs/              # New configuration system
    ├── config_structured.py  # Structured config classes
    ├── config_omegaconf.py   # New OmegaConf-based system
    └── config_utils.py       # Configuration utilities
```

## Integration Summary

### What Was Implemented

#### 1. Core Components
- **`lib/configs/config_structured.py`** - Type-safe dataclass configurations for all model types
- **`lib/configs/config_omegaconf.py`** - Modern OmegaConf-based configuration manager
- **`lib/configs/config_utils.py`** - Utility functions for validation, interpolation, and experiment management

#### 2. Configuration Files
- **`lib/configs/base.yaml`** - Common parameters for all models
- **`lib/configs/sttran.yaml`** - STTRAN-specific configuration
- **`lib/configs/stket.yaml`** - STKET-specific configuration
- **`lib/configs/tempura.yaml`** - Tempura-specific configuration
- **`lib/configs/easg.yaml`** - EASG-specific configuration
- **`lib/configs/scenellm.yaml`** - SceneLLM-specific configuration
- **`lib/configs/oed.yaml`** - OED-specific configuration
- **`lib/configs/experiments/`** - Example experiment configurations

#### 3. Documentation and Examples
- **`README.md`** - This comprehensive documentation
- **`usage.py`** - Usage examples and demonstrations

### Key Benefits Achieved

1. **Zero Breaking Changes**: Existing code continues to work unchanged
2. **Better Organization**: YAML files are easier to manage and version control
3. **Type Safety**: Runtime validation catches configuration errors early
4. **Reproducibility**: Easy saving and sharing of experiment configurations
5. **Flexibility**: Support for complex configuration scenarios
6. **Developer Experience**: Better tooling and validation
7. **Future-Proof**: Modern configuration management foundation

### Testing Results

The integration has been thoroughly tested:
- ✅ Original `Config` class works unchanged
- ✅ New `ConfigManager` and structured configs work correctly
- ✅ YAML configuration loading functions properly
- ✅ Configuration validation and error checking works
- ✅ Experiment management utilities function correctly
- ✅ All examples run successfully

### Next Steps

1. **For Current Users**: No action required - everything works as before
2. **For New Features**: Consider using YAML configuration files
3. **For Experiments**: Use the experiment management utilities
4. **For Validation**: Enable configuration validation in new scripts

## Conclusion

The VidSgg configuration system provides a modern, flexible configuration management solution while maintaining complete backward compatibility. Users can continue using the existing system unchanged while having access to powerful new features when they're ready to adopt them. This approach ensures a smooth transition path and minimizes disruption to existing workflows.

The system is production-ready and has been thoroughly tested. All existing VidSgg functionality remains intact while providing a foundation for enhanced configuration management in the future.

---

*For more examples and detailed usage instructions, see `usage.py` in this directory.*
