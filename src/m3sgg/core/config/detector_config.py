"""
Detector configuration parameters.

This module extends the base configuration with detector-specific parameters
for different object detection models.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectorConfig:
    """Configuration class for object detection models.
    
    This class contains parameters specific to different detector types
    including Faster R-CNN, ViT, DETR, YOLO, and ResNet variants. It provides
    a comprehensive set of configuration options for various detector architectures
    and their specific requirements.
    
    :param detector_type: Type of detector to use
    :type detector_type: str
    :param vit_backbone: ViT backbone architecture
    :type vit_backbone: str
    :param vit_pretrained: Whether to use pretrained ViT weights
    :type vit_pretrained: bool
    :param vit_dropout: Dropout rate for ViT
    :type vit_dropout: float
    :param vit_attention_dropout: Attention dropout rate for ViT
    :type vit_attention_dropout: float
    :param detr_backbone: DETR backbone architecture
    :type detr_backbone: str
    :param detr_num_queries: Number of query slots for DETR
    :type detr_num_queries: int
    :param detr_num_encoder_layers: Number of encoder layers for DETR
    :type detr_num_encoder_layers: int
    :param detr_num_decoder_layers: Number of decoder layers for DETR
    :type detr_num_decoder_layers: int
    :param detr_d_model: Model dimension for DETR
    :type detr_d_model: int
    :param yolo_model_size: YOLO model size variant
    :type yolo_model_size: str
    :param yolo_confidence: Confidence threshold for YOLO
    :type yolo_confidence: float
    :param yolo_iou_threshold: IoU threshold for YOLO NMS
    :type yolo_iou_threshold: float
    :param yolo_max_detections: Maximum number of detections for YOLO
    :type yolo_max_detections: int
    :param resnet_backbone: ResNet backbone architecture
    :type resnet_backbone: str
    :param resnet_pretrained: Whether to use pretrained ResNet weights
    :type resnet_pretrained: bool
    :param resnet_fpn: Whether to use Feature Pyramid Network
    :type resnet_fpn: bool
    :param detector_confidence_threshold: Confidence threshold for detections
    :type detector_confidence_threshold: float
    :param detector_nms_threshold: NMS threshold for detections
    :type detector_nms_threshold: float
    :param detector_max_detections: Maximum number of detections
    :type detector_max_detections: int
    :param detector_input_size: Input image size (height, width)
    :type detector_input_size: tuple
    :param feature_pooling: Type of feature pooling
    :type feature_pooling: str
    :param feature_pool_size: Feature pool size
    :type feature_pool_size: tuple
    :param feature_stride: Feature stride
    :type feature_stride: int
    :param detector_lr: Learning rate for detector
    :type detector_lr: float
    :param detector_weight_decay: Weight decay for detector
    :type detector_weight_decay: float
    :param detector_momentum: Momentum for detector optimizer
    :type detector_momentum: float
    :param detector_horizontal_flip: Whether to use horizontal flip augmentation
    :type detector_horizontal_flip: bool
    :param detector_vertical_flip: Whether to use vertical flip augmentation
    :type detector_vertical_flip: bool
    :param detector_rotation: Rotation angle for augmentation
    :type detector_rotation: float
    :param detector_scale_jitter: Scale jitter range for augmentation
    :type detector_scale_jitter: tuple
    :param detector_color_jitter: Color jitter strength for augmentation
    :type detector_color_jitter: float
    :param detector_use_fpn: Whether to use Feature Pyramid Network
    :type detector_use_fpn: bool
    :param detector_use_gn: Whether to use Group Normalization
    :type detector_use_gn: bool
    :param detector_use_dcn: Whether to use Deformable Convolutions
    :type detector_use_dcn: bool
    """
    
    # Detector type selection
    detector_type: str = "faster_rcnn"  # faster_rcnn, vit, detr, yolo, resnet
    
    # ViT detector parameters
    vit_backbone: str = "vit_base_patch16_224"
    vit_pretrained: bool = True
    vit_dropout: float = 0.1
    vit_attention_dropout: float = 0.0
    
    # DETR detector parameters
    detr_backbone: str = "resnet50"
    detr_num_queries: int = 100
    detr_num_encoder_layers: int = 6
    detr_num_decoder_layers: int = 6
    detr_d_model: int = 256
    
    # YOLO detector parameters
    yolo_model_size: str = "yolov8n"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    yolo_confidence: float = 0.25
    yolo_iou_threshold: float = 0.45
    yolo_max_detections: int = 1000
    
    # ResNet detector parameters
    resnet_backbone: str = "resnet50"  # resnet18, resnet34, resnet50, resnet101, resnet152
    resnet_pretrained: bool = True
    resnet_fpn: bool = True
    
    # Common detector parameters
    detector_confidence_threshold: float = 0.5
    detector_nms_threshold: float = 0.5
    detector_max_detections: int = 100
    detector_input_size: tuple = (800, 1333)  # (height, width)
    
    # Feature extraction parameters
    feature_pooling: str = "roi_align"  # roi_align, roi_pool, adaptive_avg_pool
    feature_pool_size: tuple = (7, 7)
    feature_stride: int = 16
    
    # Training parameters
    detector_lr: float = 1e-4
    detector_weight_decay: float = 1e-4
    detector_momentum: float = 0.9
    
    # Data augmentation
    detector_horizontal_flip: bool = True
    detector_vertical_flip: bool = False
    detector_rotation: float = 0.0
    detector_scale_jitter: tuple = (0.8, 1.2)
    detector_color_jitter: float = 0.1
    
    # Model-specific parameters
    detector_use_fpn: bool = True
    detector_use_gn: bool = False  # Group normalization
    detector_use_dcn: bool = False  # Deformable convolutions
    
    def get_detector_name(self) -> str:
        """Get a descriptive name for the detector configuration.
        
        :return: Detector configuration name
        :rtype: str
        """
        if self.detector_type == "vit":
            return f"ViT-{self.vit_backbone}"
        elif self.detector_type == "detr":
            return f"DETR-{self.detr_backbone}-{self.detr_num_queries}q"
        elif self.detector_type == "yolo":
            return f"YOLO-{self.yolo_model_size}"
        elif self.detector_type == "resnet":
            return f"ResNet-{self.resnet_backbone}"
        else:
            return "FasterRCNN"
    
    def validate_config(self) -> bool:
        """Validate the detector configuration.
        
        :return: True if configuration is valid
        :rtype: bool
        :raises ValueError: If configuration is invalid
        """
        valid_detector_types = ["faster_rcnn", "vit", "detr", "yolo", "resnet"]
        if self.detector_type not in valid_detector_types:
            raise ValueError(f"Invalid detector_type: {self.detector_type}. Must be one of {valid_detector_types}")
        
        if self.detector_confidence_threshold < 0 or self.detector_confidence_threshold > 1:
            raise ValueError("detector_confidence_threshold must be between 0 and 1")
        
        if self.detector_nms_threshold < 0 or self.detector_nms_threshold > 1:
            raise ValueError("detector_nms_threshold must be between 0 and 1")
        
        if self.detector_max_detections <= 0:
            raise ValueError("detector_max_detections must be positive")
        
        return True
