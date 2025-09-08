Usage Guide
===========

This comprehensive guide covers all usage patterns for M3SGG, including the new configuration system, different calling methods, applications, and examples.

Quick Start
-----------

M3SGG provides multiple ways to interact with the framework:

1. **CLI Commands** - Direct command-line interface
2. **Configuration Files** - YAML-based configuration system
3. **Python API** - Programmatic interface
4. **Applications** - GUI and web interfaces
5. **Jupyter Notebooks** - Interactive examples

Configuration System
--------------------

M3SGG features a modern, flexible configuration system with multiple approaches:

YAML Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~

Use structured YAML files for reproducible experiments:

.. code-block:: yaml

   # configs/sttran_predcls.yaml
   mode: predcls
   model_type: sttran
   dataset: action_genome
   data_path: data/action_genome
   datasize: large
   
   # Training parameters
   lr: 1e-4
   nepoch: 100
   batch_size: 1
   optimizer: adamw
   
   # Model architecture
   enc_layer: 1
   dec_layer: 3
   
   # System settings
   device: cuda:0
   seed: 42
   num_workers: 4

Structured Configuration Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use type-safe configuration classes for programmatic access:

.. code-block:: python

   from m3sgg.core.config.structured.sttran import STTranConfig
   
   config = STTranConfig(
       mode="predcls",
       lr=1e-4,
       nepoch=100,
       enc_layer=1,
       dec_layer=3
   )

Unified Configuration Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The unified interface supports both legacy and modern systems:

.. code-block:: python

   from m3sgg.core.config.unified import UnifiedConfig
   
   # Modern configuration
   config = UnifiedConfig(
       config_path="configs/sttran.yaml",
       model_type="sttran",
       use_modern=True
   )
   
   # Legacy configuration
   config = UnifiedConfig(
       cli_args=["-mode", "predcls", "-model", "sttran"],
       use_modern=False
   )

Command Line Interface
----------------------

M3SGG CLI
~~~~~~~~~

The modern CLI provides a clean interface:

.. code-block:: bash

   # Install CLI
   pip install -e .
   
   # Basic usage
   m3sgg train --config configs/sttran.yaml
   m3sgg eval --model-path output/model.pth
   m3sgg app  # Launch Streamlit app

CLI Options
~~~~~~~~~~~

.. code-block:: bash

   m3sgg train [OPTIONS]
   
   Options:
     --config, -c PATH     Path to configuration file
     --model, -m TEXT      Model type (sttran/tempura/scenellm/stket/oed/vlm)
     --dataset, -d TEXT    Dataset (action_genome/EASG)
     --mode TEXT           Training mode (predcls/sgcls/sgdet)
     --epochs, -e INT      Number of training epochs
     --lr FLOAT            Learning rate
     --batch-size, -b INT  Batch size
     --device TEXT         Device (cuda:0/cpu)
     --output, -o PATH     Output directory
     --checkpoint PATH     Path to checkpoint file
     --verbose, -v         Enable verbose logging

Legacy Training Scripts
~~~~~~~~~~~~~~~~~~~~~~~

For backward compatibility, legacy scripts are still supported:

.. code-block:: bash

   # Training
   python scripts/training/training.py -mode predcls -model sttran -data_path data/action_genome
   
   # Evaluation
   python scripts/evaluation/test.py -m predcls -model_path output/model.pth
   
   # EASG training
   python scripts/training/easg/train_with_EASG.py -mode easgcls -model sttran

Applications
------------

Streamlit Web Application
~~~~~~~~~~~~~~~~~~~~~~~~~

Interactive web interface for video scene graph generation:

.. code-block:: bash

   # Launch Streamlit app
   python scripts/apps/streamlit.py
   
   # Or use CLI
   m3sgg app

Features:
* **Video Upload**: Upload and process custom videos
* **Model Selection**: Choose from available trained models
* **Real-time Processing**: Generate scene graphs on-the-fly
* **Interactive Visualization**: Explore results with interactive plots
* **Export Options**: Save results in multiple formats
* **Chat Interface**: Natural language interaction with results

PyQt Desktop Application
~~~~~~~~~~~~~~~~~~~~~~~~

Desktop GUI for advanced users:

.. code-block:: bash

   python scripts/apps/pyqt.py

Features:
* **Native Performance**: Full desktop application experience
* **Advanced Controls**: Fine-grained parameter adjustment
* **Batch Processing**: Process multiple videos efficiently
* **Custom Visualizations**: Advanced plotting and analysis tools
* **Model Management**: Easy model switching and comparison

Jupyter Notebook Examples
-------------------------

Interactive examples in the `examples/` directory:

1. **Basic Video Scene Graph Generation** (`01_basic_video_scene_graph_generation.ipynb`)
   - Complete pipeline from video to scene graph
   - Error handling and troubleshooting
   - Configurable parameters and results analysis

2. **Scene Graph to Text Summarization** (`02_scene_graph_to_text_summarization.ipynb`)
   - Convert scene graphs to natural language
   - Multiple summarization models (T5, Pegasus)
   - Advanced prompting strategies

3. **End-to-End Video to Summary Pipeline** (`03_end_to_end_video_to_summary.ipynb`)
   - Integrated `VideoToSummaryPipeline` class
   - Modular design with error handling
   - Combined visualization and export

4. **Advanced VLM Scene Graph Generation** (`04_advanced_vlm_scene_graph_generation.ipynb`)
   - Vision-Language Model integration
   - Few-shot learning and reasoning
   - Chain-of-thought prompting

5. **Model Comparison and Evaluation** (`05_model_comparison_and_evaluation.ipynb`)
   - Comprehensive evaluation framework
   - Model comparison and ranking
   - Performance analysis and visualization

Running Examples
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start Jupyter Lab
   jupyter lab
   
   # Or Jupyter Notebook
   jupyter notebook
   
   # Navigate to examples/ directory and open notebooks

Python API
----------

Programmatic Interface
~~~~~~~~~~~~~~~~~~~~~~

Use M3SGG as a Python library:

.. code-block:: python

   from m3sgg.core.config.unified import UnifiedConfig
   from m3sgg.core.training.trainer import Trainer
   from m3sgg.datasets.action_genome import ActionGenomeDataset
   
   # Load configuration
   config = UnifiedConfig(config_path="configs/sttran.yaml")
   
   # Create dataset
   dataset = ActionGenomeDataset(
       data_path=config.data_path,
       split="train",
       mode=config.mode
   )
   
   # Initialize trainer
   trainer = Trainer(config)
   
   # Train model
   trainer.train(dataset)

Model Factory
~~~~~~~~~~~~~

Create models programmatically:

.. code-block:: python

   from m3sgg.core.training.model_factory import create_model
   
   # Create STTran model
   model = create_model("sttran", config)
   
   # Create Tempura model
   model = create_model("tempura", config)

Dataset Factory
~~~~~~~~~~~~~~~

Load datasets dynamically:

.. code-block:: python

   from m3sgg.datasets.factory import create_dataset
   
   # Create Action Genome dataset
   dataset = create_dataset("action_genome", config)
   
   # Create EASG dataset
   dataset = create_dataset("easg", config)

Training Modes
--------------

PredCLS (Predicate Classification)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predict relationships given ground truth objects:

.. code-block:: bash

   # CLI
   m3sgg train --mode predcls --model sttran
   
   # Legacy
   python scripts/training/training.py -mode predcls -model sttran

SGCLS (Scene Graph Classification)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predict both objects and relationships given bounding boxes:

.. code-block:: bash

   # CLI
   m3sgg train --mode sgcls --model sttran
   
   # Legacy
   python scripts/training/training.py -mode sgcls -model sttran

SGDET (Scene Graph Detection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

End-to-end object detection and relationship prediction:

.. code-block:: bash

   # CLI
   m3sgg train --mode sgdet --model sttran
   
   # Legacy
   python scripts/training/training.py -mode sgdet -model sttran

Model-Specific Usage
--------------------

STTran
~~~~~~

Spatial-Temporal Transformer baseline:

.. code-block:: yaml

   # configs/sttran.yaml
   model_type: sttran
   enc_layer: 1
   dec_layer: 3
   lr: 1e-4

Tempura
~~~~~~~

Uncertainty-aware temporal modeling:

.. code-block:: yaml

   # configs/tempura.yaml
   model_type: tempura
   obj_head: gmm
   rel_head: gmm
   K: 3
   obj_mem_compute: true
   rel_mem_compute: true

SceneLLM
~~~~~~~~

Large language model integration:

.. code-block:: yaml

   # configs/scenellm.yaml
   model_type: scenellm
   scenellm_training_stage: stage1
   llm_model: gemma3-270M
   fusion_layers: 3

STKET
~~~~~

Knowledge-enhanced transformer:

.. code-block:: yaml

   # configs/stket.yaml
   model_type: stket
   N_layer: 1
   enc_layer_num: 1
   dec_layer_num: 1
   use_spatial_prior: true
   use_temporal_prior: true

OED
~~~

Object-Entity Disentanglement:

.. code-block:: yaml

   # configs/oed.yaml
   model_type: oed
   oed_variant: multi
   num_queries: 100

VLM
~~~

Vision-Language Model:

.. code-block:: yaml

   # configs/vlm.yaml
   model_type: vlm
   vlm_model: blip2
   reasoning_type: chain_of_thought

Configuration Presets
---------------------

Use predefined configuration presets:

.. code-block:: bash

   # Quick test configuration
   m3sgg train --config configs/presets/quick_test.yaml
   
   # Production configuration
   m3sgg train --config configs/presets/production.yaml

Available Presets
~~~~~~~~~~~~~~~~~

* **Quick Test**: Fast training for testing
* **Production**: Optimized for best performance
* **Debug**: Verbose logging and error checking
* **Research**: Full feature set for experimentation

Advanced Usage
--------------

Custom Training Loops
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from m3sgg.core.training.trainer import Trainer
   
   class CustomTrainer(Trainer):
       def train_epoch(self, dataloader):
           # Custom training logic
           pass
   
   trainer = CustomTrainer(config)
   trainer.train()

Model Evaluation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from m3sgg.core.training.evaluation import Evaluator
   
   evaluator = Evaluator(config)
   results = evaluator.evaluate(model, dataloader)
   print(f"Recall@20: {results['recall@20']:.2f}")

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from m3sgg.utils.batch_processor import BatchProcessor
   
   processor = BatchProcessor(config)
   results = processor.process_videos(video_paths)

Performance Optimization
------------------------

GPU Memory Management
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # configs/optimized.yaml
   batch_size: 1
   gradient_accumulation_steps: 4
   mixed_precision: true
   gradient_checkpointing: true

Data Loading Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # configs/optimized.yaml
   num_workers: 8
   pin_memory: true
   persistent_workers: true
   prefetch_factor: 2

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Configuration Errors**
   * Verify YAML syntax and indentation
   * Check required parameters are present
   * Validate parameter types and ranges

**Import Errors**
   * Ensure m3sgg package is installed: `pip install -e .`
   * Check Python path includes src directory
   * Verify all dependencies are installed

**CUDA Issues**
   * Check GPU availability: `torch.cuda.is_available()`
   * Verify CUDA version compatibility
   * Use CPU fallback: `--device cpu`

**Memory Issues**
   * Reduce batch size
   * Use gradient accumulation
   * Enable gradient checkpointing
   * Use mixed precision training

**Data Loading Issues**
   * Verify dataset paths and structure
   * Check file permissions
   * Ensure sufficient disk space
   * Validate data format and annotations

Getting Help
~~~~~~~~~~~~

* **Documentation**: Check the comprehensive API documentation
* **Examples**: Run through the Jupyter notebook examples
* **Issues**: Report bugs and ask questions on GitHub
* **Community**: Join discussions and get help from the community

Next Steps
----------

* :doc:`training` - Detailed training procedures and best practices
* :doc:`evaluation` - Comprehensive evaluation metrics and analysis
* :doc:`models` - Deep dive into model architectures and implementations
* :doc:`api` - Complete API reference documentation