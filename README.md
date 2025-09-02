# VidSGG

This repository offers a framework for training Video Scene Graph Generation (VidSGG) models based on transformer-based
Deep Learning. A `feature module` and `language module` offer the ability to pre- and
post-process input and output data of the core training algorithm represented by the `sgg module`. 
The code builds on top of the paper [Spatial-Temporal Transformer for Dynamic Scene Graph Generation, ICCV 2021] with it's associated repository (https://github.com/yrcong/STTran).

A streamlit demo application allows to analyze the real-time capabilities of such systems, 
testing the entire inference process in an e2e environment using self-trained weights. 
A possible training workflow could look like this: 
- Feature module: FasterRCNN
- SGG module: STTran
- NLP module: Gemma3

Each of these module is conceptually a collection of exchangeable submodules. An Alternative example could be: V2L features -> Tempura -> Pegasus. Please refer to the [chapter:modules] for further information about the items.

**Disclaimer**: This README.md serves as a summarization of the actual documentation found on https://wagnerp4.github.io/VidSgg/. Please refer to it for a detailed installation guide, an exhaustive overview, and the api reference.

# Project Structure

```
VidSgg/
├── scripts/
│   ├── core/                    # Main execution scripts
│   │   ├── apps/                # Application interfaces
│   │   │   ├── streamlit.py     # Streamlit web application
│   │   │   ├── demovideo.py     # Demo video visualization
│   │   │   ├── pyqt.py          # PyQt desktop application
│   │   │   └── run_streamlit.ps1 # Streamlit launcher script
│   │   ├── training/            # Training scripts
│   │   │   ├── train.py         # Main training script
│   │   │   ├── train_with_EASG.py # EASG-specific training
│   │   │   ├── run_easg.py      # EASG training runner
│   │   │   ├── run_easg_rnd_search.py # EASG random search
│   │   │   ├── train_scenellm_example.py # SceneLLM training
│   │   │   ├── batch_train.ps1  # Batch training script (PowerShell)
│   │   │   └── batch_train.sh   # Batch training script (Bash)
│   │   └── evaluation/          # Evaluation scripts
│   │       └── test.py          # Model evaluation script
│   ├── datasets/                # Dataset processing scripts
│   │   ├── countPrior.py        # Prior counting utilities
│   │   ├── debug_dataset.py     # Dataset debugging tools
│   │   └── analyze_*.py         # Dataset analysis scripts
│   └── models/                  # Model-specific scripts
│       ├── debug_rcnn_*.py      # R-CNN debugging tools
│       ├── tempura_*.py         # Tempura model utilities
│       └── download_*.py        # Model download scripts
├── lib/                         # Core library modules
├── datasets/                    # Dataset implementations
├── utils/                       # Utility functions
└── data/                        # Data and checkpoints
```

Hello. We are still working. Nothing to see here...

# Installation
- Python 3.10+ (tested with Python 3.10.0)
- CUDA-compatible GPU (recommended)
- For the original setup please refer to https://github.com/yrcong/STTran.

**Manual Installation (Required)**: Before installing the main dependencies, you need to manually install PyTorch (https://pytorch.org/get-started/locally/) and DGL (https://www.dgl.ai/pages/start.html) with the correct CUDA version for your system:

**Main Dependencies:** Install the remaining dependencies using `uv sync`

# Model Checkpoints
- Feature Modules: A pretrained FasterRCNN model for Action Genome can be download [here](https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing). To use it extract it to this path `fasterRCNN/models/faster_rcnn_ag.pth`. Additional details can be found at https://github.com/jwyang/faster-rcnn.pytorch.
- SGG Modules: We provide a [[checkpoint-link](https://drive.google.com/drive/folders/12yc-D4n3Ine7jWX2cDlBMX6zFl4s2yyt?usp=drive_link)] for the best performing SGG model to try with the streamlit app. Please put it under `data/checkpoints/best_model.tar` or provide it as path in streamlit. If you want to try a different model to assess it's performance, follow the below training guide and place the checkpoint at `data/checkpoints`.
- NLP-Modules:
    - T5/Pegasus (Summarization)
    - Gemma3 270M (Action Anticipation, Language Modelling)

# Dataset
Action Genome: We use the dataset [Action Genome](https://www.actiongenome.org/#download) to train/evaluate our method. Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome).  In the experiments for SGCLS/SGDET, we only keep bounding boxes with short edges larger than 16 pixels. Please download the file [object_bbox_and_relationship_filtersmall.pkl](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing) and put it in the ```datasets``` directory. The directories of the dataset should look like:
```
|-- action_genome
    |-- annotations   #gt annotations
    |-- frames        #sampled frames
    |-- videos        #original videos
```

# Training and Evaluation
Here are some example training commands using all different SGG modes, with the same dataset and model. The entire list of arguments can be refered to in the documentation or under `VidSgg/lib/config.py`.

```python
# Mode: PredCLS
python scripts/core/training/train.py -mode predcls -datasize large -data_path data/action_genome -model sttran 
# Mode: SGCLS
python scripts/core/training/train.py -mode sgcls -datasize large -data_path data/action_genome -model sttran
# Mode: SGdetCLS
python scripts/core/training/train.py -mode sgdet -datasize large -data_path data/action_genome -model sttran

# EASG-specific training
python scripts/core/training/train_with_EASG.py -mode easgcls -datasize large -data_path data/EASG -model sttran
# EASG random search
python scripts/core/training/run_easg_rnd_search.py
```

```powershell
# PowerShell batch training
.\scripts\core\training\batch_train.ps1
```

```bash
# Bash batch training
./scripts/core/training/batch_train.sh
```

Here are some example evaluation commands using all different SGG modes, with test.py.

```python
# Mode: PredCLS
python scripts/core/evaluation/test.py -m predcls -datasize large -data_path $DATAPATH -model_path $MODELPATH
# Mode: SGCLS
python scripts/core/evaluation/test.py -m sgcls -datasize large -data_path $DATAPATH -model_path $MODELPATH
# Mode: SGdetCLS
python scripts/core/evaluation/test.py -m sgdet -datasize large -data_path $DATAPATH -model_path $MODELPATH
```

<!-- # Demo Applications
The project provides multiple interfaces for different use cases:

### Web Interface (Streamlit)
- **File**: `scripts/core/apps/streamlit.py`
- **Features**: Modern web-based interface for real-time analysis
- **Best for**: Interactive exploration and demonstration

### Desktop Interface (PyQt)
- **File**: `scripts/core/apps/pyqt.py`
- **Features**: Native desktop application
- **Best for**: Offline usage and advanced users

### Demo Video
- **File**: `scripts/core/apps/demovideo.py`
- **Features**: Video visualization and analysis
- **Best for**: Batch processing and video analysis -->

<!-- ## Video Demo -->
<!-- Click to view the demo video:
[![Demo Video](https://img.shields.io/badge/📹_Watch_Demo_Video-blue?style=for-the-badge)](https://github.com/your-username/VidSgg/raw/main/assets/demo.mp4) -->

# Streamlit Web App
Required deps: pip install streamlit streamlit_chat
Start the modern web-based interface:
```powershell
streamlit run scripts/core/apps/streamlit.py
```
Or use the provided launcher script:
```powershell
.\scripts\core\apps\run_streamlit.ps1
```
The application will open at `http://localhost:8501`.

**Note:** PyTorch and DGL must be installed manually with the correct CUDA version (see Installation section above).

# Hardware
Training was run on a single  NVIDIA RTX 3090 TI GPU for both training and testing.
