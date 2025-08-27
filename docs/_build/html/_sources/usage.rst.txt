Usage Guide
===========

This guide covers the basic usage patterns for DLHM VidSGG, including training, evaluation, and running the demo application.

Quick Start
-----------

Basic Training Command
~~~~~~~~~~~~~~~~~~~~~~

The simplest way to start training:

.. code-block:: bash

   python train.py -mode predcls -datasize large -data_path data/action_genome -model sttran

This command trains the STTran model on Action Genome dataset in PredCLS mode.

Basic Evaluation Command
~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate a trained model:

.. code-block:: bash

   python test.py -m predcls -datasize large -data_path data/action_genome -model_path output/model.pth

Command Line Interface
----------------------

Training Script (train.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main training script supports various arguments:

.. code-block:: bash

   python train.py [OPTIONS]

Key options:

* ``-mode {predcls,sgcls,sgdet}`` - SGG evaluation mode
* ``-datasize {small,large}`` - Dataset size variant
* ``-data_path PATH`` - Path to dataset directory
* ``-model {sttran,tempura,scenellm,stket}`` - Model to use
* ``-lr FLOAT`` - Learning rate (default: 1e-4)
* ``-batch_size INT`` - Batch size (default: 1)
* ``-epochs INT`` - Number of training epochs

Example training commands:

.. code-block:: bash

   # STTran model with different modes
   python train.py -mode predcls -datasize large -data_path data/action_genome -model sttran
   python train.py -mode sgcls -datasize large -data_path data/action_genome -model sttran
   python train.py -mode sgdet -datasize large -data_path data/action_genome -model sttran

   # Other models
   python train.py -mode predcls -datasize large -data_path data/action_genome -model tempura
   python train.py -mode predcls -datasize large -data_path data/action_genome -model scenellm

Test Script (test.py)
~~~~~~~~~~~~~~~~~~~~~

For model evaluation:

.. code-block:: bash

   python test.py [OPTIONS]

Key options:

* ``-m {predcls,sgcls,sgdet}`` - Evaluation mode
* ``-datasize {small,large}`` - Dataset size
* ``-data_path PATH`` - Dataset path
* ``-model_path PATH`` - Path to trained model

Demo Application
----------------

GUI Application
~~~~~~~~~~~~~~~

Launch the interactive demo:

.. code-block:: bash

   python scripts/core/gui.py

The GUI provides:

* Video loading and playback
* Real-time scene graph generation
* Visualization of detected objects and relationships
* Interactive exploration of results

Features:

* **Video Selection**: Load videos from dataset or custom files
* **Model Selection**: Switch between different trained models
* **Visualization Options**: Customize display of bounding boxes and relationships
* **Export Results**: Save generated scene graphs and visualizations

Scene Graph Generation Modes
-----------------------------

The framework supports three main evaluation modes:

PredCLS (Predicate Classification)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Input**: Ground truth object bounding boxes and labels
* **Task**: Predict relationships between objects
* **Usage**: Focuses purely on relationship prediction accuracy

.. code-block:: bash

   python train.py -mode predcls -data_path data/action_genome -model sttran

SGCLS (Scene Graph Classification)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Input**: Ground truth object bounding boxes
* **Task**: Predict both object labels and relationships
* **Usage**: Evaluates object classification and relationship prediction

.. code-block:: bash

   python train.py -mode sgcls -data_path data/action_genome -model sttran

SGDET (Scene Graph Detection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Input**: Raw video frames
* **Task**: Detect objects and predict relationships end-to-end
* **Usage**: Most challenging mode, full pipeline evaluation

.. code-block:: bash

   python train.py -mode sgdet -data_path data/action_genome -model sttran

Model Selection
---------------

Available Models
~~~~~~~~~~~~~~~~

STTran (Spatial-Temporal Transformer)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The baseline model for video scene graph generation:

.. code-block:: bash

   python train.py -model sttran -mode predcls

Tempura
^^^^^^^

Temporal relationship modeling with uncertainty:

.. code-block:: bash

   python train.py -model tempura -mode predcls

SceneLLM
^^^^^^^^

Large language model integration for scene understanding:

.. code-block:: bash

   python train.py -model scenellm -mode predcls

STKET
^^^^^

Spatial-temporal knowledge-enhanced transformer:

.. code-block:: bash

   python train.py -model stket -mode predcls

Configuration
-------------

Model Configuration
~~~~~~~~~~~~~~~~~~~

Models can be configured through configuration files or command-line arguments:

.. code-block:: python

   # Example configuration
   config = {
       'learning_rate': 1e-4,
       'batch_size': 1,
       'hidden_dim': 512,
       'num_epochs': 100
   }

Dataset Configuration
~~~~~~~~~~~~~~~~~~~~~

Configure dataset paths and preprocessing options:

.. code-block:: bash

   export DATA_PATH=/path/to/datasets
   export MODEL_PATH=/path/to/models

Output Structure
----------------

Training outputs are organized as follows:

.. code-block:: text

   output/
   ├── action_genome/
   │   ├── sttran_predcls_YYYYMMDD_HHMMSS/
   │   │   ├── logfile.txt
   │   │   ├── checkpoint.tar
   │   │   └── predictions.csv
   │   └── tempura_sgdet_YYYYMMDD_HHMMSS/
   │       ├── logfile.txt
   │       ├── checkpoint.tar
   │       └── predictions.csv
   └── EASG/
       └── sttran_easgcls_YYYYMMDD_HHMMSS/
           ├── logfile.txt
           ├── checkpoint.tar
           └── predictions.csv

Log files contain training progress, checkpoint files store model states, and prediction files contain evaluation results.

Performance Tips
----------------

Training Optimization
~~~~~~~~~~~~~~~~~~~~~

* **GPU Memory**: Reduce batch size if encountering CUDA out of memory errors
* **Mixed Precision**: Use automatic mixed precision for faster training
* **Data Loading**: Increase number of workers for faster data loading

.. code-block:: bash

   # Example optimized training command
   python train.py -model sttran -mode predcls -batch_size 2 -num_workers 4

Evaluation Optimization
~~~~~~~~~~~~~~~~~~~~~~~

* **Checkpoint Selection**: Use the best validation checkpoint for evaluation
* **Batch Processing**: Process multiple samples simultaneously when possible

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Training Loss Not Decreasing**
   * Check learning rate (try 1e-5 for fine-tuning)
   * Verify dataset loading and preprocessing
   * Ensure proper model configuration

**Low Evaluation Scores**
   * Verify ground truth data format
   * Check evaluation metrics implementation
   * Compare with baseline results

**Memory Issues**
   * Reduce batch size
   * Use gradient accumulation
   * Enable gradient checkpointing

Next Steps
----------

* :doc:`training` - Detailed training procedures and best practices
* :doc:`evaluation` - Comprehensive evaluation metrics and analysis
* :doc:`models` - Deep dive into model architectures and implementations
