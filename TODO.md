# TODO List

## OED Integration Status

### ✅ Completed
- [x] **OED Analysis**: Analyzed OED repository structure and understood core components
- [x] **Integration Strategy**: Developed comprehensive integration strategy for OED into existing codebase
- [x] **Single vs Multi Decision**: Determined to integrate both OED variants (single and multi)
- [x] **Configuration Integration**: Added OED-specific parameters to existing config system
- [x] **Core Model Integration**: Created lib/oed/ directory structure with all core model files
- [x] **Training Pipeline Integration**: Integrated OED model type into existing training pipeline

### 🔄 In Progress
- [ ] **Testing and Validation**: Test OED integration with existing codebase
- [ ] **Documentation**: Update documentation to include OED model

### 📋 Remaining Tasks
- [ ] **Data Loading Adaptation**: Ensure OED models work with existing Action Genome data loader
- [ ] **Evaluation Integration**: Integrate OED evaluation metrics with existing evaluation system
- [ ] **Performance Optimization**: Optimize OED model performance and memory usage
- [ ] **Model Checkpointing**: Implement proper checkpoint saving/loading for OED models
- [ ] **Hyperparameter Tuning**: Fine-tune OED hyperparameters for optimal performance

## OED Model Architecture

### Core Components
- **OEDMulti**: Multi-frame variant with Progressively Refined Module (PRM)
- **OEDSingle**: Single-frame variant for baseline comparison
- **Transformer**: Cascaded decoders (HOPD + Interaction)
- **Backbone**: ResNet-based feature extraction
- **Criterion**: Hungarian matcher + focal loss for relations

### Key Features
- One-stage end-to-end training
- DETR-style architecture
- Temporal context aggregation without additional trackers
- Support for both single and multi-frame processing

## Usage

### Training OED Models
```bash
# Single-frame variant
python train.py --model_type oed --oed_variant single

# Multi-frame variant (recommended)
python train.py --model_type oed --oed_variant multi --use_matcher
```

### Configuration Parameters
- `num_queries`: Number of query slots (default: 100)
- `dec_layers_hopd`: HOPD decoder layers (default: 6)
- `dec_layers_interaction`: Interaction decoder layers (default: 6)
- `oed_variant`: Model variant ("single" or "multi")
- `oed_use_matching`: Whether to use Hungarian matcher

## Notes
- OED models require the existing Hungarian matcher from the codebase
- Both variants support the same loss functions and evaluation metrics
- Multi-frame variant is the main contribution and recommended for production use