# Summarization Wrapper Classes

This module provides a comprehensive set of wrapper classes for different summarization models, with a focus on T5 and Pegasus models. The wrappers provide a unified interface and additional functionality for custom loading strategies.

## Overview

The summarization wrapper system consists of:

1. **BaseSummarizationWrapper**: Abstract base class providing unified interface
2. **T5SummarizationWrapper**: Wrapper for T5-based models
3. **PegasusSummarizationWrapper**: Wrapper for Pegasus-based models
4. **PegasusSeparateLoader**: Extension for separate tokenizer/model loading
5. **PegasusCustomConfig**: Extension for custom configuration options

## Quick Start

### Basic Usage

```python
from lib.summarization_wrapper import T5SummarizationWrapper, PegasusSummarizationWrapper

# T5 summarization
t5_wrapper = T5SummarizationWrapper("google-t5/t5-base")
summary = t5_wrapper.summarize("Your text here")

# Pegasus summarization
pegasus_wrapper = PegasusSummarizationWrapper("google/pegasus-xsum")
summary = pegasus_wrapper.summarize("Your text here")
```

### Separate Loading (Pegasus)

```python
from lib.summarization_wrapper import PegasusSeparateLoader

# Initialize separate loader
separate_loader = PegasusSeparateLoader("google/pegasus-xsum")

# Load components separately
tokenizer = separate_loader.load_tokenizer()
model = separate_loader.load_model()

# Generate summary
summary = separate_loader.summarize("Your text here")
```

### Custom Configuration (Pegasus)

```python
from lib.summarization_wrapper import PegasusCustomConfig

# Initialize with custom configuration
custom_config = PegasusCustomConfig("google/pegasus-xsum")

# Load with custom parameters
custom_config.load_with_config(
    config_kwargs={
        "max_position_embeddings": 1024,
        "num_attention_heads": 16
    },
    model_kwargs={
        "low_cpu_mem_usage": True
    }
)

# Generate summary with custom generation parameters
summary = custom_config.summarize(
    "Your text here",
    max_length=100,
    min_length=15,
    length_penalty=1.5,
    num_beams=6
)
```

## Class Details

### BaseSummarizationWrapper

Abstract base class that provides a unified interface for all summarization models.

**Key Methods:**
- `summarize(text, **kwargs)`: Summarize a single text
- `summarize_batch(texts, **kwargs)`: Summarize multiple texts

### T5SummarizationWrapper

Wrapper for T5-based summarization models.

**Features:**
- Automatic input formatting with "summarize:" prefix
- Optimized generation parameters for T5
- Support for sampling-based generation

**Default Parameters:**
- `max_length`: 100
- `min_length`: 20
- `num_beams`: 4
- `do_sample`: True
- `temperature`: 0.7

### PegasusSummarizationWrapper

Wrapper for Pegasus-based summarization models.

**Features:**
- Optimized for abstractive summarization
- Beam search by default
- Longer input handling (up to 1024 tokens)

**Default Parameters:**
- `max_length`: 128
- `min_length`: 20
- `num_beams`: 4
- `do_sample`: False
- `length_penalty`: 2.0

### PegasusSeparateLoader

Extension class that allows separate loading of tokenizer and model.

**Use Cases:**
- Custom loading strategies
- Memory optimization
- Incremental loading
- Custom error handling

**Key Methods:**
- `load_tokenizer(**kwargs)`: Load tokenizer separately
- `load_model(**kwargs)`: Load model separately
- `is_loaded()`: Check if components are loaded

### PegasusCustomConfig

Extension class for custom configuration options.

**Features:**
- Custom model configuration
- Flexible generation parameters
- Memory optimization options
- Custom tokenizer/model loading

**Key Methods:**
- `load_with_config(config_kwargs, model_kwargs)`: Load with custom config
- `set_generation_config(**kwargs)`: Set generation parameters

## Model Comparison

| Feature | T5 | Pegasus |
|---------|----|---------|
| Input Format | "summarize: text" | Direct text |
| Max Input Length | 512 tokens | 1024 tokens |
| Default Generation | Sampling | Beam Search |
| Best For | General purpose | Abstractive summarization |
| Memory Usage | Moderate | Higher |

## Usage Examples

### Batch Processing

```python
# Process multiple texts efficiently
texts = ["Text 1", "Text 2", "Text 3"]
summaries = wrapper.summarize_batch(texts)
```

### Custom Generation Parameters

```python
# Customize generation for specific use case
summary = wrapper.summarize(
    text,
    max_length=150,
    min_length=30,
    num_beams=8,
    length_penalty=1.5,
    repetition_penalty=1.3
)
```

### Error Handling

```python
try:
    summary = wrapper.summarize(text)
except Exception as e:
    print(f"Summarization failed: {e}")
    # Fallback to simpler approach
    summary = "Summary unavailable"
```

## Performance Tips

1. **Memory Management**: Use `PegasusSeparateLoader` for large models
2. **Batch Processing**: Use `summarize_batch()` for multiple texts
3. **Custom Config**: Use `PegasusCustomConfig` for memory optimization
4. **Device Management**: Models automatically use GPU if available

## Dependencies

- torch
- transformers
- typing (for type hints)

## Testing

Run the test script to verify functionality:

```bash
python scripts/test_summarization_wrappers.py
```

This will test all wrapper classes with sample texts and different configurations. 