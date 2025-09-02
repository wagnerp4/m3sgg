# VidSGG

# Overview
Our code uses the [Spatial-Temporal Transformer for Dynamic Scene Graph Generation, ICCV 2021] repository (https://github.com/yrcong/STTran) as a baseline. On top of that, new models, datasets, and processing functionality were added, as explained in the respective sections below.

## Installation

- Python 3.10+ (tested with Python 3.10.0)
- CUDA-compatible GPU (recommended)
- For the original setup please refer to https://github.com/yrcong/STTran.

### Manual Installation (Required)
Before installing the main dependencies, you need to manually install PyTorch (https://pytorch.org/get-started/locally/) and DGL (https://www.dgl.ai/pages/start.html) with the correct CUDA version for your system:

### Main Dependencies
Install the remaining dependencies using uv:
```bash
uv sync
```

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

We provide a [[checkpoint-link](https://drive.google.com/drive/folders/12yc-D4n3Ine7jWX2cDlBMX6zFl4s2yyt?usp=drive_link)] for the best performing SGG model to try with the streamlit app.
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
Here are some example training commands using all different SGG modes, with the same dataset and model. The entire list of arguments can be refered to in the documentation or under `VidSgg/lib/config.py`.
```python
# Mode: PredCLS
python train.py -mode predcls -datasize large -data_path data/action_genome -model sttran 
# Mode: SGCLS
python train.py -mode sgcls -datasize large -data_path data/action_genome -model sttran
# Mode: SGdetCLS
python train.py -mode sgdet -datasize large -data_path data/action_genome -model sttran
```

## Evaluation
Here are some example evaluation commands using all different SGG modes, with test.py.
```python
# Mode: PredCLS
python test.py -m predcls -datasize large -data_path $DATAPATH -model_path $MODELPATH
# Mode: SGCLS
python test.py -m sgcls -datasize large -data_path $DATAPATH -model_path $MODELPATH
# Mode: SGdetCLS
python test.py -m sgdet -datasize large -data_path $DATAPATH -model_path $MODELPATH
```

## Checkpoints

<!-- TODO: Create table with checkpoints -->

<!-- ## Output Format
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
``` -->

# Demo Applications:

## Video Demo

<!-- Click to view the demo video:
[![Demo Video](https://img.shields.io/badge/📹_Watch_Demo_Video-blue?style=for-the-badge)](assets/demo.mp4) -->

<video width="800" controls>
  <source src="assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<!-- After uploading to GitHub, use this format:
https://github.com/user-attachments/assets/demo.mp4
-->

<!-- Convert demo.mp4 to GIF and use:
![Demo](assets/demo.gif)
-->

## Streamlit Web App
Required deps: pip install streamlit streamlit_chat
Start the modern web-based interface:
```powershell
streamlit run app.py
```
The application will open at `http://localhost:8501`.

**Note:** PyTorch and DGL must be installed manually with the correct CUDA version (see Installation section above).

## Hardware
Training was run on a single  NVIDIA RTX 3090 TI GPU for both training and testing.
