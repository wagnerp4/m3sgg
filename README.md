# VidSGG

# Overview
Our code uses the [Spatial-Temporal Transformer for Dynamic Scene Graph Generation, ICCV 2021] repository (https://github.com/yrcong/STTran) as a baseline. On top of that, new models, datasets, and processing functionality were added, as explained in the respective sections below.

## Installation

### Prerequisites
- Python 3.10+ (tested with Python 3.10.0)
- CUDA-compatible GPU (recommended)

### Manual Installation (Required)
Before installing the main dependencies, you need to manually install PyTorch and DGL with the correct CUDA version for your system:

**PyTorch Installation:**
```bash
# For CUDA 11.8
pip install torch>=1.12.0 torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch>=1.12.0 torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (not recommended for training)
pip install torch>=1.12.0 torchvision>=0.13.0 --index-url https://download.pytorch.org/whl/cpu
```

**DGL Installation:**
Visit https://www.dgl.ai/pages/start.html and choose the DGL version with CUDA support that matches your PyTorch installation.

### Main Dependencies
Install the remaining dependencies using uv:
```bash
uv sync
or:
uv venv
uv pip install -e .
```

## General Usage
- Certain C module dependencies and related functionality which was outdated
in STTran-like repositories was replaced with python alternatives.
- For the original setup please refer to https://github.com/yrcong/STTran.

## Backbone
For the object detector part, please follow the compilation from https://github.com/jwyang/faster-rcnn.pytorch
We provide a pretrained FasterRCNN model for Action Genome. Please download [here](https://drive.google.com/file/d/1-u930Pk0JYz3ivS6V_HNTM1D5AxmN5Bs/view?usp=sharing) and put it in 
```
fasterRCNN/models/faster_rcnn_ag.pth
```

## SGG Models
- STTran
- DSG-DETR
- STKET
- Tempura
- OED
- SceneLLM

We provide a [checkpoint-link] for the best performing SGG model to try with the streamlit app.
Please put it under 'data/checkpoints/best_model.pth'. 
If you want to try a different model to assess it's performance, follow the below training guide and
place the checkpoint at 'data/checkpoints'.

## NLP Models
- T5/Pegasus (Summarization)
- Gemma3 270M (Action Anticipation, Language Modelling)

## Dataset
Action Genome: We use the dataset [Action Genome](https://www.actiongenome.org/#download) to train/evaluate our method. Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome). The directories of the dataset should look like:
```
|-- action_genome
    |-- annotations   #gt annotations
    |-- frames        #sampled frames
    |-- videos        #original videos
```

 In the experiments for SGCLS/SGDET, we only keep bounding boxes with short edges larger than 16 pixels. 
 Please download the file [object_bbox_and_relationship_filtersmall.pkl](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view?usp=sharing) and put it in the ```dataloader```

## Training
Here are some example training commands using all different SGG modes, with the same dataset and model. 

- Provide all available parameters, which are necessary instead of three commands just one

+ For PredCLS: 
```
python train.py -mode predcls -datasize large -data_path data/action_genome -model sttran 
```
+ For SGCLS: 
```
python train.py -mode sgcls -datasize large -data_path data/action_genome -model sttran 
```
+ For SGDET: 
```
python train.py -mode sgdet -datasize large -data_path data/action_genome -model sttran 
```

## Evaluation
Here are some example evaluation commands using all different SGG modes, with test.py.

+ For PredCLS ([trained Model](https://drive.google.com/file/d/18oFR8hfH3W84AYjR1yktsjQKeIlKbilo/view?usp=sharing)): 
```
python test.py -m predcls -datasize large -data_path $DATAPATH -model_path $MODELPATH
```
+ For SGCLS ([trained Model](https://drive.google.com/file/d/1E3fTGyh7Uhcsy7nBfrrY0t3jIi88uclF/view?usp=sharing)): : 
```
python test.py -m sgcls -datasize large -data_path $DATAPATH -model_path $MODELPATH
```
+ For SGDET ([trained Model](https://drive.google.com/file/d/19qW2x61eXBhQ2x3liJSRmKOF6zKqtYjV/view?usp=sharing)): : 
```
python test.py -m sgdet -datasize large -data_path $DATAPATH -model_path $MODELPATH
```

## Output
```
output/
├── action_genome/
│   ├── sttran_predcls_20241201_143022/
│   │   ├── logfile.txt
│   │   ├── checkpoint.tar
│   │   └── predictions.csv
│   └── tempura_sgdet_20241201_150045/
│       ├── logfile.txt
│       ├── checkpoint.tar
│       └── predictions.csv
└── EASG/
    └── sttran_easgcls_20241201_160000/
        ├── logfile.txt
        ├── checkpoint.tar
        └── predictions.csv
```

# Demo Applications:

## Streamlit Web App (Recommended)
Required deps: pip install streamlit streamlit_chat
Start the modern web-based interface:
```powershell
streamlit run app.py
```
The application will open at `http://localhost:8501`.

## PyQt5 GUI (Legacy)
```
python scripts/core/gui.py
```

## Dependencies
The following dependencies are automatically handled via `pyproject.toml`:
```
streamlit>=1.29.0
plotly>=5.17.0
opencv-python>=4.5.0
matplotlib>=3.4.0
numpy>=1.21.0
scipy>=1.9.0
pillow>=8.3.0
tqdm>=4.62.0
transformers>=4.20.0
peft>=0.4.0
pot>=0.9.0
cython
cffi
msgpack
tensorboardX
```

**Note:** PyTorch and DGL must be installed manually with the correct CUDA version (see Installation section above).

## Hardware
Training was run on a single  NVIDIA RTX 3090 TI GPU for both training and testing.
