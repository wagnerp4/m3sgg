## Migration:
Decided switching name from default ML task descriptor to
the first name iteration for the framework.
Old name: vidsgg (Video Scene Graph Generation)
New name: m3sgg (Modular, multi-modal Scene Graph Generation)
Furthermore a large scale restructuring plan is explained below.
Task: Start realizing the phases one-by-one.

### Phase 1) General new structure (ongoing)
```
M3SGG/
├── assets/
├── docs/                          # Update .rst files
├── examples/                      # Instantiate notebooks
├── fasterRCNN/                    # Recompile detector
└── data/                          # Data and checkpoints
├── src/
│   └── m3sgg/                     # Main package
│       ├── __init__.py
│       ├── core/                  # Core functionality
│       │   ├── __init__.py
│       │   ├── config/
│       │   │   ├──__init__.py
│       │   │   ├── config.py      # Current lib/config.py
│       │   │   ├── configs/       # Current lib/configs content
│       │   ├── models/            # Model implementations
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── sttran.py
│       │   │   ├── tempura.py
│       │   │   ├── scenellm.py
│       │   │   ├── vlm.py
│       │   │   └── oed.py
│       │   ├── detectors/         # Object detectors
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── faster_rcnn.py # Equivalent to current object_detector.py
│       │   │   └── easg.py
│       │   └── evaluation/        # Evaluation tools
│       │       ├── __init__.py
│       │       └── metrics.py
│       ├── datasets/              # Dataset implementations
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── action_genome.py
│       │   └── easg.py
│       ├── utils/                 # Utilities
│       │   ├── __init__.py
│       │   ├── io.py
│       │   └── visualization.py
│       └── language/              # Language models
│           ├── __init__.py
│           ├── summarization.py
│           └── vlm.py
├── scripts/                       # Application scripts
│   ├── apps/
│   ├── data_scripts/              # Remove
│   ├── evaluation/
│   └── model_scripts/             # Remove
│   ├── training/
├── tests/                         # Rework completely
│   ├── simple_test.py
│   ├── ...
```

### Modularize Trainer
- class Trainer
    - def train_loop
    =
    - def train_iter
    - def train_epoch 
    - def train_step
- class Evaluation
    - def eval_loop

### Improved Testing System:
automatize with pytest
```
tests/
├── conftest.py                    # Shared fixtures
├── unit/                          # Unit tests
│   ├── test_models/
│   ├── test_datasets/
│   └── test_utils/
├── integration/                   # Integration tests
│   ├── test_training_pipeline.py
│   └── test_evaluation_pipeline.py
├── fixtures/                      # Test data
│   ├── sample_videos/
│   └── sample_annotations/
└── performance/                   # Performance tests
    └── test_memory_usage.py
```

### Add Example Notebooks:
Add in /examples.
Notebooks that demonstrate video + scene graph -> summarization.


### Improved Configuration System.
```
│   └── m3sgg/
│       ├── core/
│       │   ├── config/                    # Unified config directory
│       │   │   ├── __init__.py
│       │   │   ├── legacy.py              # Old argparse-based config.py
│       │   │   ├── modern.py              # OmegaConf-based ConfigManager
│       │   │   ├── unified.py             # Unified interface
│       │   │   ├── structured/            # Structured config definitions
│       │   │   │   ├── __init__.py
│       │   │   │   ├── base.py
│       │   │   │   ├── sttran.py
│       │   │   │   ├── tempura.py
│       │   │   │   ├── scenellm.py
│       │   │   │   └── oed.py
│       │   │   └── presets/               # YAML preset files
│       │   │       ├── base.yaml
│       │   │       ├── sttran.yaml
│       │   │       ├── tempura.yaml
│       │   │       ├── scenellm.yaml
│       │   │       ├── oed.yaml
│       │   │       └── experiments/
│       │   │           ├── action_genome_small.yaml
│       │   │           └── oed_optimal.yaml
│       │   ├── models/
│       │   ├── detectors/
│       │   └── evaluation/
│       └── ...
```

### Improved CLI System.

Should result into something like this:
``` bash
# Run training
m3sgg train  # runs currently selected config (above) with scripts/training/training.py

# Run eval
m3sgg eval

# Run streamlit
m3sgg app
```

### Improved Dependency Management

- Problem: Inconsistent import management with manual sys.path manipulation
- Solution: src/ - Centralized import management with proper __init__.py files

- Lazy loading and dependency isolation for heavy dependencies like transformers/DGL
- Mixed dependency management (uv + pyproject.toml + manual installs)
- Installation profiles for different use cases
- Categorized dependencies with optional extras

- Solution:
```toml
[project]
name = "mmsgg"
version = "0.1.0"
description = "Video Scene Graph Generation Framework"
requires-python = ">=3.10"

# Core dependencies (always required)
dependencies = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "matplotlib>=3.4.0",
    "opencv-python>=4.5.0",
    "tqdm>=4.62.0",
    "scipy>=1.9.0",
    "pillow>=8.3.0",
    "scikit-learn>=1.1.0",
    "easydict>=1.13",
    "PyYAML>=6.0.1",
    "omegaconf>=2.3.0",
    "filelock>=3.18.0",
    "typing-extensions>=4.10.0",
    "sympy>=1.13.1",
    "networkx>=3.4.2,<3.5",
    "jinja2>=3.1.6",
    "fsspec>=2025.5.1",
    "requests>=2.31.0",
    "h5py>=3.14.0",
    "imageio>=2.37.0",
    "python-dotenv>=1.1.0",
]

# Optional dependencies for different use cases
[project.optional-dependencies]
# PyTorch (user must choose CPU or CUDA version)
pytorch-cpu = [
    "torch>=1.9.0",
    "torchvision>=0.10.0",
    "torchaudio>=0.9.0",
]
pytorch-cuda = [
    "torch>=1.9.0",
    "torchvision>=0.10.0", 
    "torchaudio>=0.9.0",
]

# GUI applications
gui = [
    "streamlit>=1.29.0",
    "streamlit-chat>=0.1.1",
    "plotly>=5.17.0",
    "PyQt5>=5.15.0",
]

# SceneLLM specific
scenellm = [
    "transformers>=4.20.0",
    "peft>=0.4.0",
    "pot>=0.9.0",
    "dgl>=1.0.0",
    "accelerate>=0.20.0",
]

# VLM specific  
vlm = [
    "transformers>=4.20.0",
    "timm>=0.4.12",
    "accelerate>=0.20.0",
]

# Development and testing
dev = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "black>=22.0",
    "isort>=5.0",
    "flake8>=4.0",
    "mypy>=0.900",
]

# Documentation
docs = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "sphinx-autodoc-typehints>=1.12.0",
    "sphinx-copybutton>=0.5.0",
    "sphinxcontrib-napoleon>=0.7",
]

# All optional dependencies
all = [
    "vidsgg[gui,scenellm,vlm,dev,docs]"
]
```

```
# Create environment-specific lock files
uv lock --extra pytorch-cuda --extra gui --extra scenellm
uv lock --extra pytorch-cpu --extra gui
uv lock --extra pytorch-cuda --extra vlm
```
