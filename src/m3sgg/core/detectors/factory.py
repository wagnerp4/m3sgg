"""
Detector factory for creating different object detection models.

This module provides a unified interface for creating various object detection
models including Faster R-CNN, ViT-based detectors, DETR, YOLO, and ResNet variants.
"""

import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

import torch


class DetectorFactory:
    """Factory class for creating object detection models based on configuration.
    
    This factory provides a unified interface for creating different object detection
    models including Faster R-CNN, ViT-based detectors, DETR, YOLO, and ResNet variants.
    It handles detector instantiation logic and provides a consistent interface
    across different detector types for improved modularity and maintainability.
    
    :param config: Configuration object containing detector parameters
    :type config: Config
    :param dataset_train: Training dataset for extracting class information
    :type dataset_train: Dataset
    :param device: Device to place the detector on
    :type device: torch.device
    :param logger: Optional logger instance
    :type logger: Optional[logging.Logger]
    """
    
    def __init__(self, config, dataset_train, device, logger: Optional[logging.Logger] = None):
        """Initialize the detector factory.
        
        :param config: Configuration object containing detector parameters
        :type config: Config
        :param dataset_train: Training dataset for extracting class information
        :type dataset_train: Dataset
        :param device: Device to place the detector on
        :type device: torch.device
        :param logger: Optional logger instance
        :type logger: Optional[logging.Logger]
        """
        self.config = config
        self.dataset_train = dataset_train
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
    
    def create_detector(self) -> torch.nn.Module:
        """Create a detector instance based on the configuration.
        
        :return: Instantiated detector
        :rtype: torch.nn.Module
        :raises ValueError: If detector type is not supported
        """
        detector_type = getattr(self.config, 'detector_type', 'faster_rcnn')
        
        if detector_type == "faster_rcnn":
            return self._create_faster_rcnn_detector()
        elif detector_type == "vit":
            return self._create_vit_detector()
        elif detector_type == "detr":
            return self._create_detr_detector()
        elif detector_type == "yolo":
            return self._create_yolo_detector()
        elif detector_type == "resnet":
            return self._create_resnet_detector()
        else:
            raise ValueError(f"Detector type '{detector_type}' not supported")
    
    def _create_faster_rcnn_detector(self) -> torch.nn.Module:
        """Create Faster R-CNN detector.
        
        :return: Faster R-CNN detector instance
        :rtype: torch.nn.Module
        """
        if self.config.dataset == "EASG":
            from m3sgg.core.detectors.easg.object_detector_EASG import detector as detector_EASG
            detector = detector_EASG(
                train=True,
                object_classes=self.dataset_train.obj_classes,
                use_SUPPLY=True,
                mode=self.config.mode,
            ).to(device=self.device)
            self.logger.info("Created EASG Faster R-CNN detector")
        else:
            from m3sgg.core.detectors.faster_rcnn import detector
            detector = detector(
                train=True,
                object_classes=self.dataset_train.object_classes,
                use_SUPPLY=True,
                mode=self.config.mode,
            ).to(device=self.device)
            self.logger.info("Created Faster R-CNN detector")
        
        detector.eval()
        return detector
    
    def _create_vit_detector(self) -> torch.nn.Module:
        """Create ViT-based detector.
        
        :return: ViT detector instance
        :rtype: torch.nn.Module
        :raises NotImplementedError: If ViT detector is not implemented
        """
        try:
            from m3sgg.core.detectors.vit.detector import ViTDetector
            detector = ViTDetector(
                object_classes=self.dataset_train.object_classes,
                mode=self.config.mode,
                backbone=self.config.vit_backbone,
                pretrained=self.config.vit_pretrained,
            ).to(device=self.device)
            self.logger.info(f"Created ViT detector with backbone: {self.config.vit_backbone}")
            detector.eval()
            return detector
        except ImportError:
            raise NotImplementedError("ViT detector not yet implemented")
    
    def _create_detr_detector(self) -> torch.nn.Module:
        """Create DETR-based detector.
        
        :return: DETR detector instance
        :rtype: torch.nn.Module
        :raises NotImplementedError: If DETR detector is not implemented
        """
        try:
            from m3sgg.core.detectors.detr.detector import DETRDetector
            detector = DETRDetector(
                object_classes=self.dataset_train.object_classes,
                mode=self.config.mode,
                backbone=self.config.detr_backbone,
                num_queries=self.config.detr_num_queries,
            ).to(device=self.device)
            self.logger.info(f"Created DETR detector with backbone: {self.config.detr_backbone}")
            detector.eval()
            return detector
        except ImportError:
            raise NotImplementedError("DETR detector not yet implemented")
    
    def _create_yolo_detector(self) -> torch.nn.Module:
        """Create YOLO-based detector.
        
        :return: YOLO detector instance
        :rtype: torch.nn.Module
        :raises NotImplementedError: If YOLO detector is not implemented
        """
        try:
            from m3sgg.core.detectors.yolo.detector import YOLODetector
            detector = YOLODetector(
                object_classes=self.dataset_train.object_classes,
                mode=self.config.mode,
                model_size=self.config.yolo_model_size,
                confidence_threshold=self.config.yolo_confidence,
            ).to(device=self.device)
            self.logger.info(f"Created YOLO detector with model size: {self.config.yolo_model_size}")
            detector.eval()
            return detector
        except ImportError:
            raise NotImplementedError("YOLO detector not yet implemented")
    
    def _create_resnet_detector(self) -> torch.nn.Module:
        """Create ResNet-based detector.
        
        :return: ResNet detector instance
        :rtype: torch.nn.Module
        :raises NotImplementedError: If ResNet detector is not implemented
        """
        try:
            from m3sgg.core.detectors.resnet.detector import ResNetDetector
            detector = ResNetDetector(
                object_classes=self.dataset_train.object_classes,
                mode=self.config.mode,
                backbone=self.config.resnet_backbone,
                pretrained=self.config.resnet_pretrained,
            ).to(device=self.device)
            self.logger.info(f"Created ResNet detector with backbone: {self.config.resnet_backbone}")
            detector.eval()
            return detector
        except ImportError:
            raise NotImplementedError("ResNet detector not yet implemented")


class BaseDetector(ABC):
    """Abstract base class for all detector implementations.
    
    This abstract base class ensures consistent interface across different
    detector types. All detector implementations must inherit from this class
    and implement the required abstract methods to maintain compatibility
    with the detector factory and training pipeline.
    
    :param ABC: Abstract base class from abc module
    :type ABC: ABC
    """
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        """Forward pass of the detector.
        
        This method performs the forward pass of the detector and returns
        a dictionary containing detection results including bounding boxes,
        class predictions, and confidence scores.
        
        :param args: Variable length argument list
        :type args: Any
        :param kwargs: Arbitrary keyword arguments
        :type kwargs: Any
        :return: Dictionary containing detection results
        :rtype: Dict[str, Any]
        """
        pass
    
    @abstractmethod
    def get_feature_extractor(self) -> torch.nn.Module:
        """Get the feature extraction backbone.
        
        Returns the feature extraction backbone module used by the detector.
        This is typically the CNN or transformer backbone that extracts
        features from input images.
        
        :return: Feature extraction module
        :rtype: torch.nn.Module
        """
        pass
    
    @abstractmethod
    def get_classifier(self) -> torch.nn.Module:
        """Get the classification head.
        
        Returns the classification head module used by the detector.
        This module is responsible for predicting object classes
        based on the extracted features.
        
        :return: Classification module
        :rtype: torch.nn.Module
        """
        pass
