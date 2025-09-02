# VidSgg Configuration System

This document provides an overview of the VidSgg configuration system, which now supports both the original configuration approach and a new OmegaConf-based system.

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
from lib.config_omegaconf import ConfigManager, create_config

# From YAML file
config = ConfigManager(config_path="configs/sttran.yaml")

# From structured config
config = create_config(model_type="scenellm", lr=2e-5)
```

## Configuration Files

The new system includes pre-configured YAML files in the `configs/` directory:

- `configs/base.yaml` - Common parameters for all models
- `configs/sttran.yaml` - STTRAN-specific configuration
- `configs/stket.yaml` - STKET-specific configuration
- `configs/tempura.yaml` - Tempura-specific configuration
- `configs/easg.yaml` - EASG-specific configuration
- `configs/scenellm.yaml` - SceneLLM-specific configuration
- `configs/oed.yaml` - OED-specific configuration
- `configs/experiments/` - Example experiment configurations

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

## Migration Strategy

### Phase 1: No Changes Required
- Continue using existing code as-is
- All current scripts work without modification
- Original `Config` class remains unchanged

### Phase 2: Optional Adoption
- Start using YAML files for new experiments
- Gradually adopt new features as needed
- Mix old and new approaches as desired

### Phase 3: Full Migration (Future)
- When ready, migrate to new system entirely
- Remove old configuration system
- Leverage all advanced features

## Examples

See `examples/config_usage_examples.py` for comprehensive examples of:
- Backward compatibility demonstration
- YAML configuration usage
- Structured configuration creation
- Command line overrides
- Experiment management
- Configuration comparison

## File Structure

```
configs/
├── base.yaml              # Common parameters
├── sttran.yaml           # STTRAN model config
├── stket.yaml            # STKET model config
├── tempura.yaml          # Tempura model config
├── easg.yaml             # EASG model config
├── scenellm.yaml         # SceneLLM model config
├── oed.yaml              # OED model config
└── experiments/
    ├── action_genome_small.yaml
    └── scenellm_vqvae.yaml

lib/
├── config.py             # Original config (unchanged)
├── config_structured.py  # Structured config classes
├── config_omegaconf.py   # New OmegaConf-based system
└── config_utils.py       # Configuration utilities

examples/
└── config_usage_examples.py  # Usage examples

tests/
└── test_config_omegaconf.py  # Comprehensive tests
```

## Benefits

1. **Zero Breaking Changes**: Existing code works unchanged
2. **Gradual Migration**: Adopt new features at your own pace
3. **Better Organization**: YAML files are easier to manage
4. **Type Safety**: Catch configuration errors early
5. **Reproducibility**: Save and share experiment configurations
6. **Flexibility**: Support for complex configuration scenarios
7. **Validation**: Automatic checking of configuration validity

## Getting Started

1. **For existing users**: No changes needed - everything works as before
2. **For new features**: Try the YAML configuration files
3. **For experiments**: Use the experiment management utilities
4. **For validation**: Enable configuration validation in your scripts

## Support

- See `CONFIG_MIGRATION_GUIDE.md` for detailed migration instructions
- Run `examples/config_usage_examples.py` to see the system in action
- Check `tests/test_config_omegaconf.py` for usage examples
- The original `lib/config.py` remains unchanged for reference

This configuration system provides a smooth upgrade path while maintaining full backward compatibility with existing VidSgg code.
