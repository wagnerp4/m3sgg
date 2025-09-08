# Detector Factory

This module provides a unified interface for creating different object detection models.

## Architecture

```
src/m3sgg/core/detectors/
├── factory.py              # Main detector factory
├── faster_rcnn/           # Faster R-CNN implementation
├── easg/                  # EASG-specific detector
├── vit/                   # ViT-based detectors (future)
├── detr/                  # DETR-based detectors (future)
├── yolo/                  # YOLO-based detectors (future)
└── resnet/                # ResNet-based detectors (future)
```

## Usage

### Basic Usage

```python
from m3sgg.core.detectors.factory import DetectorFactory

# Create detector factory
detector_factory = DetectorFactory(config, dataset_train, device, logger)

# Create detector
detector = detector_factory.create_detector()
```

### Configuration

Set the detector type in your configuration:

```python
config.detector_type = "faster_rcnn"  # Default
config.detector_type = "vit"          # ViT-based detector
config.detector_type = "detr"         # DETR-based detector
config.detector_type = "yolo"         # YOLO-based detector
config.detector_type = "resnet"       # ResNet-based detector
```

## Adding New Detector Types

### 1. Create Detector Implementation

Create a new directory for your detector type:

```python
# src/m3sgg/core/detectors/my_detector/detector.py
import torch
from m3sgg.core.detectors.factory import BaseDetector

class MyDetector(BaseDetector):
    def __init__(self, object_classes, mode, **kwargs):
        super().__init__()
        # Initialize your detector
        
    def forward(self, *args, **kwargs):
        # Implement forward pass
        return {"detections": detections, "features": features}
    
    def get_feature_extractor(self):
        # Return feature extraction backbone
        return self.backbone
    
    def get_classifier(self):
        # Return classification head
        return self.classifier
```

### 2. Add Factory Method

Add a new method to `DetectorFactory`:

```python
def _create_my_detector(self) -> torch.nn.Module:
    """Create MyDetector instance.
    
    :return: MyDetector instance
    :rtype: torch.nn.Module
    """
    from m3sgg.core.detectors.my_detector.detector import MyDetector
    
    detector = MyDetector(
        object_classes=self.dataset_train.object_classes,
        mode=self.config.mode,
        # Add your specific parameters
        my_param=self.config.my_param,
    ).to(device=self.device)
    
    self.logger.info("Created MyDetector")
    detector.eval()
    return detector
```

### 3. Update Factory Logic

Add your detector type to the main factory method:

```python
def create_detector(self) -> torch.nn.Module:
    detector_type = getattr(self.config, 'detector_type', 'faster_rcnn')
    
    if detector_type == "faster_rcnn":
        return self._create_faster_rcnn_detector()
    elif detector_type == "my_detector":  # Add this line
        return self._create_my_detector()  # Add this line
    # ... other detector types
    else:
        raise ValueError(f"Detector type '{detector_type}' not supported")
```

### 4. Add Configuration Parameters

Add detector-specific parameters to your configuration:

```python
# In your config class or detector_config.py
my_param: float = 0.5
my_other_param: str = "default_value"
```

## Current Detector Types

### Faster R-CNN (Default)
- **Type**: `faster_rcnn`
- **Status**: ✅ Implemented
- **Description**: Standard Faster R-CNN with ResNet backbone

### EASG Detector
- **Type**: `faster_rcnn` (with EASG dataset)
- **Status**: ✅ Implemented
- **Description**: EASG-specific Faster R-CNN variant

### ViT Detector (Future)
- **Type**: `vit`
- **Status**: 🚧 Planned
- **Description**: Vision Transformer-based detector

### DETR Detector (Future)
- **Type**: `detr`
- **Status**: 🚧 Planned
- **Description**: Detection Transformer-based detector

### YOLO Detector (Future)
- **Type**: `yolo`
- **Status**: 🚧 Planned
- **Description**: YOLO-based detector

### ResNet Detector (Future)
- **Type**: `resnet`
- **Status**: 🚧 Planned
- **Description**: ResNet-based detector variants

## Benefits

1. **Unified Interface**: All detectors follow the same interface
2. **Easy Extension**: Simple to add new detector types
3. **Configuration-Driven**: Detector selection via configuration
4. **Backward Compatibility**: Existing code continues to work
5. **Type Safety**: Abstract base class ensures consistent interface
6. **Logging**: Built-in logging for detector creation
7. **Device Management**: Automatic device placement
