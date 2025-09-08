Datasets
========

This section provides detailed information about the datasets supported by M3SGG.

Supported Datasets
------------------

Action Genome
~~~~~~~~~~~~~

The Action Genome dataset is the primary dataset for video scene graph generation.

**Overview**

* **Type**: Video Scene Graph Dataset
* **Domain**: Human activities and object interactions
* **Size**: ~10,000 videos with dense annotations
* **Format**: MP4 videos with JSON annotations

**Dataset Structure**

.. code-block:: text

   action_genome/
   ├── annotations/     # Ground truth scene graph annotations
   │   ├── train/
   │   ├── val/
   │   └── test/
   ├── frames/         # Extracted video frames
   │   ├── video_001/
   │   ├── video_002/
   │   └── ...
   └── videos/         # Original video files
       ├── video_001.mp4
       ├── video_002.mp4
       └── ...

**Annotation Format**

Each annotation file contains:

.. code-block:: javascript

   {
     "video_id": "video_001",
     "frame_annotations": [
       {
         "frame_id": 1,
         "objects": [
           {
             "object_id": 1,
             "bbox": [100, 50, 200, 150],
             "class": "person",
             "attributes": ["adult", "standing"]
           }
         ],
         "relationships": [
           {
             "subject_id": 1,
             "object_id": 2,
             "predicate": "holding"
           }
         ]
       }
     ]
   }

**Download and Setup**

1. Visit https://www.actiongenome.org/#download
2. Download the complete dataset
3. Process using the ActionGenome Toolkit
4. Place in ``data/action_genome/`` directory

EASG Dataset
~~~~~~~~~~~~

The EASG (Enhanced Action Scene Graph) dataset provides additional annotations and features.

**Overview**

* **Type**: Enhanced Video Scene Graph Dataset
* **Domain**: Extended human activities with fine-grained annotations
* **Features**: Additional semantic features and temporal annotations

**Dataset Structure**

.. code-block:: text

   EASG/
   ├── EASG/
   │   ├── annotations/
   │   └── features/
   ├── frames/
   ├── features_verb.pt
   ├── verb_features.pt
   └── model_final.pth

**Setup Instructions**

TODO: Add detailed EASG setup instructions

Visual Genome Dataset
~~~~~~~~~~~~~~~~~~~~~

Visual Genome provides static image scene graphs that can be used for pretraining.

**Overview**

* **Type**: Static Image Scene Graph Dataset
* **Domain**: General object relationships in images
* **Size**: ~100,000 images with scene graph annotations

**Setup Instructions**

TODO: Add Visual Genome integration details

Dataset Processing
------------------

Data Preprocessing
~~~~~~~~~~~~~~~~~~

The framework includes several preprocessing utilities:

**Frame Extraction**

.. code-block:: python

   from m3sgg.datasets.action_genome import ActionGenomeDataset
   
   # Initialize dataset
   dataset = ActionGenomeDataset(
       data_path="data/action_genome",
       split="train",
       mode="predcls"
   )

**Annotation Processing**

.. code-block:: python

   # Load and process annotations
   annotations = dataset.load_annotations()
   processed_data = dataset.preprocess_annotations(annotations)

**Feature Extraction**

.. code-block:: python

   # Extract visual features
   features = dataset.extract_features(video_path)

Data Loading
~~~~~~~~~~~~

**Basic Usage**

.. code-block:: python

   from torch.utils.data import DataLoader
   from m3sgg.datasets.action_genome import ActionGenomeDataset
   
   # Create dataset
   dataset = ActionGenomeDataset(
       data_path="data/action_genome",
       split="train",
       mode="predcls"
   )
   
   # Create data loader
   dataloader = DataLoader(
       dataset,
       batch_size=1,
       shuffle=True,
       num_workers=4
   )
   
   # Iterate through data
   for batch in dataloader:
       frames, annotations, metadata = batch
       # Process batch...

**Advanced Configuration**

.. code-block:: python

   # Custom dataset configuration
   dataset = ActionGenomeDataset(
       data_path="data/action_genome",
       split="train",
       mode="predcls",
       filter_duplicate_relations=True,
       filter_multiple_preds=False,
       frame_sample_rate=1
   )

Dataset Statistics
------------------

Action Genome Statistics
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Action Genome Dataset Statistics
   :widths: 25 25 25 25
   :header-rows: 1

   * - Split
     - Videos
     - Frames
     - Relationships
   * - Train
     - 7,842
     - 476,583
     - 1,752,524
   * - Validation
     - 1,960
     - 119,145
     - 438,131
   * - Test
     - 1,960
     - 119,170
     - 438,384

**Object Classes**

The dataset includes 35 object categories:

.. code-block:: text

   person, chair, table, cup, plate, food, bag, bed, book, laptop,
   phone, tv, remote, mouse, keyboard, bottle, wine_glass, fork,
   knife, spoon, bowl, banana, apple, sandwich, orange, broccoli,
   carrot, hot_dog, pizza, donut, cake, refrigerator, oven,
   microwave, toaster

**Relationship Predicates**

The dataset includes 25 relationship types:

.. code-block:: text

   looking_at, not_looking_at, unsure, above, beneath, in_front_of,
   behind, on_the_side_of, in, carrying, covered_by, drinking_from,
   eating, have_it_on_the_back, holding, leaning_on, lying_on,
   not_contacting, other_relationship, sitting_on, standing_on,
   touching, twisting, wearing, wiping

Quality Assurance
-----------------

Data Validation
~~~~~~~~~~~~~~~

The framework includes validation utilities:

.. code-block:: python

   from utils.validation import validate_dataset
   
   # Validate dataset integrity
   validation_report = validate_dataset("data/action_genome")
   print(validation_report)

**Common Validation Checks**

* File existence and accessibility
* Annotation format consistency
* Bounding box validity
* Frame-annotation alignment
* Missing or corrupted files

Performance Considerations
--------------------------

Loading Optimization
~~~~~~~~~~~~~~~~~~~~

* **Caching**: Enable feature caching for faster loading
* **Parallel Loading**: Use multiple workers for data loading
* **Memory Management**: Monitor memory usage with large datasets

.. code-block:: python

   # Optimized data loading
   dataloader = DataLoader(
       dataset,
       batch_size=4,
       shuffle=True,
       num_workers=8,
       pin_memory=True,
       persistent_workers=True
   )

Storage Requirements
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Storage Requirements
   :widths: 30 35 35
   :header-rows: 1

   * - Dataset
     - Raw Size
     - Processed Size
   * - Action Genome
     - ~500GB
     - ~200GB
   * - EASG
     - ~100GB
     - ~50GB
   * - Visual Genome
     - ~15GB
     - ~10GB

Custom Datasets
---------------

Adding New Datasets
~~~~~~~~~~~~~~~~~~~

To add support for a new dataset:

1. Create a new dataloader class inheriting from base dataset
2. Implement required methods: ``__init__``, ``__len__``, ``__getitem__``
3. Add dataset-specific preprocessing functions
4. Update configuration files

.. code-block:: python

   from dataloader.base import BaseDataset
   
   class CustomDataset(BaseDataset):
       def __init__(self, data_path, split, mode):
           super().__init__(data_path, split, mode)
           # Custom initialization
       
       def __getitem__(self, idx):
           # Load and return data sample
           pass
       
       def __len__(self):
           # Return dataset size
           pass

Dataset Conversion
~~~~~~~~~~~~~~~~~~

Utilities for converting between dataset formats:

.. code-block:: bash

   # Convert from custom format to Action Genome format
   python scripts/datasets/convert_dataset.py --input custom_data --output action_genome_format

Next Steps
----------

* :doc:`training` - Learn how to train models on these datasets
* :doc:`api/dataloader` - Detailed API documentation for data loading
* :doc:`evaluation` - Understand evaluation metrics and procedures
