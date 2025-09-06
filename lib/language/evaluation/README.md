# Language Module Evaluation

This package provides comprehensive evaluation tools for the language module's summarization capabilities, focusing on video caption generation using scene graph data.

## Overview

The evaluation framework includes:
- **Dataset Loading**: MSR-VTT dataset download and subset creation
- **Metrics**: ROUGE, BLEU, METEOR, and semantic similarity evaluation
- **Benchmarking**: Automated evaluation pipeline for multiple models
- **Results Management**: Structured result storage and analysis

## Installation

Install the required dependencies:

```bash
pip install -r lib/language/evaluation/requirements.txt
```

## Quick Start

### 1. Run Basic Benchmark

```bash
python lib/language/evaluation/run_benchmark.py \
    --checkpoint data/checkpoints/sgdet_test/model_best.tar \
    --subset-size 100 \
    --output output/benchmark_results.json
```

### 2. Run with Specific Models

```bash
python lib/language/evaluation/run_benchmark.py \
    --checkpoint data/checkpoints/sgdet_test/model_best.tar \
    --models t5_base pegasus_xsum \
    --subset-size 50
```

### 3. Run with Verbose Logging

```bash
python lib/language/evaluation/run_benchmark.py \
    --checkpoint data/checkpoints/sgdet_test/model_best.tar \
    --verbose
```

## Usage Examples

### Python API

```python
from lib.language.evaluation import SummarizationBenchmark

# Initialize benchmark
benchmark = SummarizationBenchmark(
    checkpoint_path="data/checkpoints/sgdet_test/model_best.tar",
    device="cuda:0"
)

# Load models
benchmark.load_models()

# Run evaluation
results = benchmark.run_scenario1_benchmark(subset_size=100)

# Print results
benchmark.print_results(results)

# Save results
benchmark.save_results(results, "output/results.json")
```

### Dataset Loading

```python
from lib.language.evaluation import MSRVTTLoader, create_subset

# Create subset
subset = create_subset(train_size=400, test_size=100)

# Or use loader directly
loader = MSRVTTLoader(cache_dir="data/msr_vtt")
dataset = loader.download_dataset()
subset = loader.create_subset(train_size=400, test_size=100)
```

### Metrics Computation

```python
from lib.language.evaluation import SummarizationMetrics

# Initialize metrics
metrics = SummarizationMetrics()

# Compute all metrics
predictions = ["A person is walking in the park."]
references = ["A person walks through the park."]

results = metrics.compute_all_metrics(predictions, references)
print(metrics.format_results(results))
```

## Benchmark Scenarios

### Scenario 1: Video Caption Generation (Implemented)
- **Input**: Scene graphs from videos
- **Output**: Text summaries/captions
- **Models**: T5-base, Pegasus-xsum, Pegasus-custom
- **Metrics**: ROUGE-1/2/L, BLEU-1/2/3/4, METEOR, Semantic Similarity

### Scenario 2: Temporal Summarization (Planned)
- **Input**: Multi-frame scene graphs
- **Output**: Temporal summaries
- **Focus**: Temporal coherence and information aggregation

## Configuration

### Command Line Arguments

- `--checkpoint`: Path to STTran checkpoint (required)
- `--subset-size`: Number of test samples (default: 100)
- `--models`: List of models to evaluate (default: all)
- `--device`: Device for inference (default: cuda:0)
- `--cache-dir`: Dataset cache directory (default: data/msr_vtt)
- `--output`: Output file path (default: output/summarization_benchmark_results.json)
- `--verbose`: Enable verbose logging

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Specify GPU devices
- `HF_HOME`: Hugging Face cache directory
- `TRANSFORMERS_CACHE`: Transformers cache directory

## Output Format

Results are saved in JSON format with the following structure:

```json
{
  "timestamp": "2024-01-15 10:30:00",
  "device": "cuda:0",
  "checkpoint_path": "data/checkpoints/sgdet_test/model_best.tar",
  "results": {
    "t5_base": {
      "rouge1": 0.342,
      "rouge2": 0.187,
      "rougeL": 0.298,
      "bleu1": 0.456,
      "bleu2": 0.234,
      "bleu3": 0.123,
      "bleu4": 0.067,
      "meteor": 0.234,
      "semantic_similarity": 0.789
    },
    "pegasus_xsum": {
      // ... similar structure
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Dataset Download Fails**: Check internet connection and disk space
3. **Model Loading Errors**: Verify checkpoint path and model compatibility
4. **Missing Dependencies**: Install requirements.txt

### Debug Mode

Run with verbose logging to see detailed information:

```bash
python lib/language/evaluation/run_benchmark.py --verbose
```

## Contributing

To add new evaluation metrics or benchmark scenarios:

1. Extend `SummarizationMetrics` class for new metrics
2. Add new scenarios to `SummarizationBenchmark` class
3. Update command line interface in `run_benchmark.py`
4. Add tests and documentation

## License

This evaluation framework is part of the VidSGG project and follows the same license terms.
