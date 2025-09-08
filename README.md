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
- Language module: Gemma3

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

**PyTorch Installation (Required)**: PyTorch and torchvision are not included as dependencies to allow users to choose the appropriate version for their system. Install them manually before installing the main dependencies:

**Complete Installation Workflow:**
```bash
uv venv --python 3.10.0
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\activate.ps1
uv pip install -e .
```

# Model Checkpoints
- Feature Module: A pretrained FasterRCNN model for Action Genome can be download [here](https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing). To use it extract it to this path `fasterRCNN/models/faster_rcnn_ag.pth`. Additional details can be found at https://github.com/jwyang/faster-rcnn.pytorch.
- SGG Module: We provide a [[checkpoint-link](https://drive.google.com/drive/folders/12yc-D4n3Ine7jWX2cDlBMX6zFl4s2yyt?usp=drive_link)] for some models with auto-detection to try with the streamlit app. Please put it under `data/checkpoints/best_model.tar` or provide it as path in streamlit. If you want to try a different model to assess it's performance, follow the below training guide and place the checkpoint at `data/checkpoints`.
- Language Module:
    - Summarization: T5, Pegasus
    - General tasks: Gemma3

# Dataset
Action Genome: We use the dataset [Action Genome](https://www.actiongenome.org/#download) to train/evaluate our method. Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome).  In the experiments for SGCLS/SGDET, we only keep bounding boxes with short edges larger than 16 pixels. Please download the file [object_bbox_and_relationship_filtersmall.pkl](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing) and put it in the ```datasets``` directory.

# Training and Evaluation
Here is a quick command to get you going in the training, using the default model on easy mode:
```bash
python scripts/training/train.py -mode predcls -datasize large -data_path data/action_genome -model sttran
```
For detailed training commands, model-specific configurations, batch training scripts, and evaluation procedures, see the complete documentation [Training Documentation](https://wagnerp4.github.io/VidSgg/training.html) containing a complete training guide with all modes, models, and advanced configurations.

# Streamlit Web App
Watch the VidSgg demo video:

<div align="center">
  <a href="https://youtu.be/VeVcv9HD2t8">
    <img src="https://img.youtube.com/vi/VeVcv9HD2t8/maxresdefault.jpg" alt="VidSgg Demo Video" style="width:100%;max-width:800px;border-radius:8px;box-shadow:0 4px 8px rgba(0,0,0,0.1);">
  </a>
  <br>
  <p><strong>Click the thumbnail above to watch the full demo on YouTube</strong></p>
</div>

Start the modern web-based interface:
```powershell
streamlit run scripts/apps/streamlit.py
```
The application will open at `http://localhost:8501`.

**Note:** PyTorch must be installed manually with the correct CUDA version (see Installation section above).

# Hardware
Training was run on a single  NVIDIA RTX 3090 TI GPU for both training and testing.

# Local Development
```shell
# Run all tests
python -m pytest tests/

# Run specific categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/performance/

# With coverage
python -m pytest tests/ --cov=src/m3sgg --cov-report=html

# Build docs
cd docs && .\make.bat html
Start-Process ".\_build\html\index.html"
```