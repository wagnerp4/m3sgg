# Data Directory Structure

This directory contains datasets, models, and related data files for the M3SGG project.

## Directory Structure

```
data/
├── action_genome216/                    # Action Genome dataset
├── action_genome1600/                 # Action Genome 200 subset
├── action_genome/                # Action Genome 9000 subset
│   ├── annotations
│   │   ├── frame_list.txt
│   │   ├── object_bbox_and_relationships.pkl
│   │   ├── object_classes.txt
│   │   ├── person_bbox.pkl
│   │   ├── relationship_classes.txt
│   ├── frames
│   │   ├── 0A8CF.mp4/
│   ├── videos
│   │   ├── 0A8CF.mp4
├── cache/                           # Cached data files
│   ├── glove.6B.200d.pkl
│   └── glove.6B.300d.pkl
├── checkpoints/                     # Model checkpoints
│   └── action_genome/sgdet_test/model_best.tar
├── EASG/                           # EASG dataset
│   ├── EASG/
│   ├── EASG_unict_master_final.json
│   ├── features_verb.pt
│   ├── frames/
│   ├── model_final.pth
│   ├── roi_feats_train.pkl
│   ├── roi_feats_val.pkl
│   ├── slowfast_8x8_R101.pyth
│   └── verb_features.pt
├── mock_dataset/                    # Mock dataset for testing
│   ├── metadata.json
│   ├── test_data.json
│   └── train_data.json
├── msr_vtt/                        # MSR-VTT dataset
├── TestAnnos/                      # Test annotations
├── TrainAnnos/                     # Training annotations
├── action_genome9000_detailed_report.txt
├── action_genome9000_stats.json
├── glove.6B.100d.txt               # GloVe word embeddings (100d)
├── glove.6B.200d                   # GloVe word embeddings (200d)
├── glove.6B.200d.pt                # PyTorch format GloVe embeddings
├── glove.6B.200d.txt               # GloVe word embeddings (200d text)
├── glove.6B.300d.txt               # GloVe word embeddings (300d)
├── glove.6B.50d.txt                # GloVe word embeddings (50d)
├── object_bbox_and_relationship_filtersmall.pkl
├── scene_graph_demo.mp4            # Demo video
├── SceneLLM.pdf                    # SceneLLM paper
├── TestPrior.json                  # Test prior data
└── TrainPrior.json                 # Training prior data
```

## File Descriptions

- **GloVe Embeddings**: Pre-trained word embeddings in various dimensions (50d, 100d, 200d, 300d)
- **Action Genome**: Multi-modal dataset for scene graph generation
- **EASG**: Extended Action Scene Graph dataset
- **MSR-VTT**: Microsoft Research Video to Text dataset
- **Checkpoints**: Saved model weights and training states
- **Annotations**: Training and test annotation files
- **Cache**: Cached preprocessed data for faster loading
