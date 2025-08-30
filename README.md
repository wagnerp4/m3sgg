# DLHM VidSGG

# Overview
Our code uses the [Spatial-Temporal Transformer for Dynamic Scene Graph Generation, ICCV 2021] repository (https://github.com/yrcong/STTran) as a baseline. On top of that, new models, datasets, and processing functionality were added, as explained in the respective sections below.

## General Usage
- Versions: python=3.10.0, torch>=1.9.0 and torchvision>=0.10.0.
- https://www.dgl.ai/pages/start.html chose dgl version with cuda
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
- SceneLLM (new)

We provide a [checkpoint-link] for the best performing SGG model. Please put it under 'data/checkpoints/best_model.pth'.

## NLP Models
- T5/Pegasus (summarization)
- gemma3 270M (action anticipation, language modeling)
- feature processing (?)

## Dataset
Action Genome: We use the dataset [Action Genome](https://www.actiongenome.org/#download) to train/evaluate our method. Please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome). The directories of the dataset should look like:
```
|-- action_genome
    |-- annotations   #gt annotations
    |-- frames        #sampled frames
    |-- videos        #original videos
```

EASG dataset: TODO

Visual Genome: TODO

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
Start the modern web-based interface:
```powershell
streamlit run app.py
```
The application will open at `http://localhost:8501` and features:
- **Video Upload**: Support for MP4, AVI, MOV, and MKV formats
- **Real-time Processing**: Live scene graph generation with progress tracking
- **Interactive Visualizations**: Dynamic charts showing objects and relationships over time
- **Model Selection**: Choose from available trained checkpoints
- **Export Options**: Download results in multiple formats

### Getting Started with Streamlit App
1. Ensure you have a trained model checkpoint in the `output/` directory
2. Run `streamlit run app.py` 
   - The app uses pre-configured settings (port 8501, 200MB upload limit)
   - Modify `.streamlit/config.toml` if you need different settings (see `.streamlit/config.toml.template` for all options)
3. Select your model from the sidebar dropdown
4. Upload a video file (MP4, AVI, MOV, MKV)
5. Click "Generate Scene Graph" to start processing
6. View real-time results and interactive visualizations

## PyQt5 GUI (Legacy)
```
python scripts/core/gui.py
```

## Dependencies
The following dependencies are automatically handled via `pyproject.toml`:
```
streamlit>=1.29.0
plotly>=5.17.0
PyQt5>=5.15.0
opencv-python>=4.5.0
matplotlib>=3.5.0
numpy>=1.21.0
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.3.0
tqdm>=4.62.0 
```

## Hardware
Training was run on a single  NVIDIA RTX 3090 TI GPU for both training and testing.
