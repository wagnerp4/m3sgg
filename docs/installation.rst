Installation
============

This guide explains how to install and set up M3SGG on your system.

Requirements
------------

System Requirements
~~~~~~~~~~~~~~~~~~~

* **Operating System**: Linux, macOS, or Windows
* **Python**: 3.10.0 or higher
* **CUDA**: Compatible GPU with CUDA support (recommended for training)
* **Memory**: At least 16GB RAM (32GB recommended for large datasets)
* **Storage**: At least 50GB free space for datasets and models

Installation Methods
--------------------

Using pip (editable):
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e .

Optional extras:
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # summarization extras
   pip install -e .[summarization]

PyTorch is not pinned in dependencies. Install the variant that fits your system:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # CPU
   pip install torch torchvision
   
   # CUDA (example for cu121)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

GUI Dependencies (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the demo application:

.. code-block:: text

   PyQt5>=5.15.0
   opencv-python>=4.5.0
   matplotlib>=3.5.0

Installation Steps
------------------

1. Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone <repository-url>
   cd m3sgg

2. Create virtual environment (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1

3. Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -e .

4. Install DGL (if using SceneLLM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visit https://www.dgl.ai/pages/start.html and choose the appropriate DGL version with CUDA support for your system.

.. code-block:: bash

   pip install dgl-cu117  # Example for CUDA 11.7

5. Setup FasterRCNN backbone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow compilation instructions from `https://github.com/jwyang/faster-rcnn.pytorch`.

Download the pretrained FasterRCNN model:

.. code-block:: bash

   # Download from: https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing
   # Place at: fasterRCNN/models/faster_rcnn_ag.pth

Dataset Setup
-------------

Action Genome dataset
~~~~~~~~~~~~~~~~~~~~~

1. Download the Action Genome dataset from https://www.actiongenome.org/#download
2. Process using the ActionGenome Toolkit: https://github.com/JingweiJ/ActionGenome
3. Organize the dataset structure:

.. code-block:: text

   data/
   └── action_genome/
       ├── annotations/  # GT annotations
       ├── frames/       # Sampled frames
       └── videos/       # Original videos

4. Download the additional filter file:

.. code-block:: bash

   # Download object_bbox_and_relationship_filtersmall.pkl
   # From: https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing
   # Place in: src/m3sgg/utils/

EASG dataset
~~~~~~~~~~~~

TODO: Add EASG dataset setup instructions

Visual Genome dataset
~~~~~~~~~~~~~~~~~~~~~

TODO: Add Visual Genome dataset setup instructions

Verification
------------

Test Your Installation
~~~~~~~~~~~~~~~~~~~~~~~

Run a simple test to verify the installation:

.. code-block:: bash

   python -c "import torch; print(\"PyTorch version:\", torch.__version__)"
   python -c "import m3sgg; print(\"M3SGG imported successfully\")"

Test GPU Support (if available):

.. code-block:: bash

   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

Run Tests
~~~~~~~~~

Execute the test suite:

.. code-block:: bash

   python -m pytest tests/

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'dgl'**
   Install DGL with appropriate CUDA version from https://www.dgl.ai/

**CUDA out of memory**
   Reduce batch size in configuration or use CPU-only mode

**FasterRCNN compilation errors**
   Ensure proper C++ compiler is installed and CUDA paths are set correctly

**Missing dataset files**
   Verify dataset download and placement according to the directory structure

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~

For optimal performance:

* **Training**: NVIDIA RTX 3090 or equivalent (24GB VRAM recommended)
* **Inference**: NVIDIA GTX 1080 or equivalent (8GB VRAM minimum)
* **CPU-only**: Possible but significantly slower

Next Steps
----------

After successful installation, proceed to:

* :doc:`usage` - Learn basic usage patterns
* :doc:`datasets` - Understand dataset formats and preparation
* :doc:`training` - Start training your first model
