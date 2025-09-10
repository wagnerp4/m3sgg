"""Video processing utilities for Streamlit integration.

This module provides video processing capabilities for the Streamlit web interface,
including model initialization, frame processing, and scene graph generation.
It supports multiple model types and datasets with automatic model detection.

Classes:
    StreamlitVideoProcessor: Main video processor class for Streamlit integration
"""

import json
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

try:
    from m3sgg.utils.model_detector import get_model_info_from_checkpoint

    MODEL_DETECTOR_AVAILABLE = True
except ImportError:
    MODEL_DETECTOR_AVAILABLE = False

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class StreamlitVideoProcessor:
    """Video processor for Streamlit integration with VidSgg pipeline.

    This class provides video processing capabilities for the Streamlit web interface,
    including model initialization, frame processing, and scene graph generation.
    It supports multiple model types (STTran, STKET, TEMPURA, SceneLLM, OED) and
    datasets (Action Genome, EASG) with automatic model detection.

    :param model_path: Path to the model checkpoint file
    :type model_path: str
    """

    def __init__(self, model_path: str, progress_callback=None):
        """Initialize the video processor with model path.

        :param model_path: Path to the model checkpoint file
        :type model_path: str
        :param progress_callback: Optional callback function for progress updates
        :type progress_callback: callable, optional
        """
        # Check CUDA availability and set device accordingly
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Using CUDA device for processing")
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, using CPU device")

        # Don't disable CUDA if we're using it
        if self.device.type == "cpu":
            # Disable DirectML and other accelerators only for CPU
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

            # Monkey patch torch.cuda to prevent accelerator access
            self._patch_torch_cuda()

        self.model_path = model_path
        self.object_classes = self._load_object_classes()
        self.relationship_classes = self._load_relationship_classes()
        self.setup_models(progress_callback)

    def _patch_torch_cuda(self):
        """Monkey patch torch.cuda to prevent accelerator access.

        This method overrides PyTorch CUDA methods to force CPU usage,
        preventing device errors in Streamlit environments when CUDA is not available.
        """
        import torch

        # Store original methods
        self._original_cuda = torch.cuda.is_available
        self._original_cuda_device = torch.cuda.device

        # Override methods to always return False/CPU
        torch.cuda.is_available = lambda: False
        torch.cuda.device = lambda *args, **kwargs: self.device

        # Override tensor.cuda() method
        def patched_cuda(self, *args, **kwargs):
            return self.to(self.device)

        torch.Tensor.cuda = patched_cuda

    def _load_object_classes(self):
        """Load object classes from file as fallback.

        :return: List of object class names
        :rtype: list
        """
        from m3sgg.core.constants import get_object_classes

        return get_object_classes("data/action_genome")

    def _load_relationship_classes(self):
        """Load relationship classes from file as fallback.

        :return: List of relationship class names
        :rtype: list
        """
        from m3sgg.core.constants import get_relationship_classes

        return get_relationship_classes("data/action_genome")

    def setup_models(self, progress_callback=None):
        """Initialize models for video processing with automatic model detection.

        This method detects the model type from the checkpoint file and initializes
        the appropriate object detector and scene graph generation model. It supports
        multiple model types including STTran, STKET, TEMPURA, SceneLLM, and OED.

        :param progress_callback: Optional callback function for progress updates
        :type progress_callback: callable, optional
        :raises ValueError: If model type cannot be detected from checkpoint
        :raises FileNotFoundError: If model checkpoint file is not found
        :raises Exception: If model initialization fails
        """
        try:
            if progress_callback:
                progress_callback("detecting_model", 0.1)
            from m3sgg.datasets.action_genome import AG
            from m3sgg.datasets.easg import EASG
            from m3sgg.core.config.config import Config
            from m3sgg.utils.matcher import HungarianMatcher

            from m3sgg.core.detectors.faster_rcnn import detector
            from m3sgg.core.detectors.easg.object_detector_EASG import (
                detector as detector_EASG,
            )

            # Detect model type from checkpoint
            if MODEL_DETECTOR_AVAILABLE:
                model_info = get_model_info_from_checkpoint(self.model_path)
                detected_model_type = model_info["model_type"]
                detected_dataset = model_info["dataset"]

                if not detected_model_type:
                    raise ValueError(
                        f"Could not detect model type from checkpoint: {self.model_path}"
                    )

                print(f"Detected model type: {detected_model_type}")
                print(f"Detected dataset: {detected_dataset}")
            else:
                # Fallback to default values if model detector is not available
                detected_model_type = "sttran"  # Default fallback
                detected_dataset = "action_genome"  # Default fallback
                print(
                    f"Model detector unavailable, using defaults: {detected_model_type}, {detected_dataset}"
                )

            self.conf = Config()
            self.conf.mode = "sgdet"

            # Set dataset-specific configuration
            if detected_dataset == "EASG":
                self.conf.data_path = "data/EASG"
                self.conf.dataset = "EASG"
            else:
                self.conf.data_path = "data/action_genome"
                self.conf.dataset = "action_genome"

            # Initialize dataset (suppress print statements during initialization)
            if progress_callback:
                progress_callback("loading_dataset", 0.3)

            import sys
            from io import StringIO

            # Capture stdout to suppress dataset initialization prints
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                if detected_dataset == "EASG":
                    self.dataset = EASG(
                        mode="test",
                        data_path=self.conf.data_path,
                    )
                else:
                    self.dataset = AG(
                        mode="test",
                        datasize=self.conf.datasize,
                        data_path=self.conf.data_path,
                        filter_nonperson_box_frame=True,
                        filter_small_box=True,
                    )
            finally:
                # Restore stdout
                sys.stdout = old_stdout

            # Object Detector
            if progress_callback:
                progress_callback("loading_fasterrcnn", 0.5)

            if detected_dataset == "EASG":
                self.object_detector = detector_EASG(
                    train=False,
                    object_classes=self.dataset.obj_classes,
                    use_SUPPLY=True,
                    mode=self.conf.mode,
                ).to(device=self.device)
            else:
                self.object_detector = detector(
                    train=False,
                    object_classes=self.dataset.object_classes,
                    use_SUPPLY=True,
                    mode=self.conf.mode,
                ).to(device=self.device)
            self.object_detector.eval()

            # Initialize SGG model based on detected type
            if progress_callback:
                progress_callback("creating_sgg_model", 0.7)

            self.model = self._create_model_from_type(
                detected_model_type, detected_dataset
            )

            # Load checkpoint
            if progress_callback:
                progress_callback("loading_model_weights", 0.8)

            if os.path.exists(self.model_path):
                ckpt = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(ckpt["state_dict"], strict=False)
                # Explicitly move model to CPU
                self.model = self.model.to(self.device)
                print(f"Loaded {detected_model_type} model from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file {self.model_path} not found!")

            self.model.eval()

            if progress_callback:
                progress_callback("initializing_matcher", 0.9)

            self.matcher = HungarianMatcher(0.5, 1, 1, 0.5).to(device=self.device)
            self.matcher.eval()

            if progress_callback:
                progress_callback("model_initialization_complete", 1.0)

        except Exception as e:
            import traceback

            error_traceback = traceback.format_exc()
            print(f"Failed to setup models: {e}")
            print(f"Full traceback: {error_traceback}")
            if STREAMLIT_AVAILABLE:
                st.error(f"Failed to setup models: {e}")
                st.error(f"Full traceback: {error_traceback}")
            raise

    def _create_model_from_type(self, model_type: str, dataset_type: str):
        """Create model instance based on detected type and dataset.

        :param model_type: Detected model type
        :type model_type: str
        :param dataset_type: Detected dataset type
        :type dataset_type: str
        :return: Initialized model
        :rtype: nn.Module
        """
        print(f"Creating model of type: {model_type} for dataset: {dataset_type}")
        print(f"Using device: {self.device}")

        try:
            if model_type == "sttran":
                if dataset_type == "EASG":
                    from m3sgg.core.detectors.easg.sttran_EASG import (
                        STTran as STTran_EASG,
                    )

                    return STTran_EASG(
                        mode=self.conf.mode,
                        obj_classes=self.dataset.obj_classes,
                        verb_classes=self.dataset.verb_classes,
                        edge_class_num=len(self.dataset.edge_classes),
                        enc_layer_num=self.conf.enc_layer,
                        dec_layer_num=self.conf.dec_layer,
                    ).to(device=self.device)
                else:
                    from m3sgg.core.models.sttran import STTran

                    return STTran(
                        mode=self.conf.mode,
                        attention_class_num=len(self.dataset.attention_relationships),
                        spatial_class_num=len(self.dataset.spatial_relationships),
                        contact_class_num=len(self.dataset.contacting_relationships),
                        obj_classes=self.dataset.object_classes,
                        enc_layer_num=self.conf.enc_layer,
                        dec_layer_num=self.conf.dec_layer,
                    ).to(device=self.device)

            elif model_type == "stket":
                from m3sgg.core.models.stket import STKET

                trainPrior = (
                    json.load(open("data/TrainPrior.json", "r"))
                    if os.path.exists("data/TrainPrior.json")
                    else None
                )
                return STKET(
                    mode=self.conf.mode,
                    attention_class_num=len(self.dataset.attention_relationships),
                    spatial_class_num=len(self.dataset.spatial_relationships),
                    contact_class_num=len(self.dataset.contacting_relationships),
                    obj_classes=self.dataset.object_classes,
                    N_layer_num=getattr(self.conf, "N_layer", 1),
                    enc_layer_num=getattr(self.conf, "enc_layer_num", 1),
                    dec_layer_num=getattr(self.conf, "dec_layer_num", 1),
                    pred_contact_threshold=getattr(
                        self.conf, "pred_contact_threshold", 0.5
                    ),
                    window_size=getattr(self.conf, "window_size", 4),
                    trainPrior=trainPrior,
                    use_spatial_prior=getattr(self.conf, "use_spatial_prior", False),
                    use_temporal_prior=getattr(self.conf, "use_temporal_prior", False),
                ).to(device=self.device)

            elif model_type == "tempura":
                from m3sgg.core.models.tempura.tempura import TEMPURA

                return TEMPURA(
                    mode=self.conf.mode,
                    attention_class_num=len(self.dataset.attention_relationships),
                    spatial_class_num=len(self.dataset.spatial_relationships),
                    contact_class_num=len(self.dataset.contacting_relationships),
                    obj_classes=self.dataset.object_classes,
                    enc_layer_num=self.conf.enc_layer,
                    dec_layer_num=self.conf.dec_layer,
                    obj_mem_compute=getattr(self.conf, "obj_mem_compute", None),
                    rel_mem_compute=getattr(self.conf, "rel_mem_compute", None),
                    take_obj_mem_feat=getattr(self.conf, "take_obj_mem_feat", False),
                    mem_fusion=getattr(self.conf, "mem_fusion", None),
                    selection=getattr(self.conf, "mem_feat_selection", None),
                    selection_lambda=getattr(self.conf, "mem_feat_lambda", 0.5),
                    obj_head=getattr(self.conf, "obj_head", "gmm"),
                    rel_head=getattr(self.conf, "rel_head", "gmm"),
                    K=getattr(self.conf, "K", None),
                ).to(device=self.device)

            elif model_type == "scenellm":
                from m3sgg.core.models.scenellm.scenellm import SceneLLM

                return SceneLLM(self.conf, self.dataset).to(device=self.device)

            elif model_type == "oed":
                # Default to multi-frame OED, could be enhanced to detect single vs multi
                from m3sgg.core.models.oed.oed_multi import OEDMulti

                return OEDMulti(self.conf, self.dataset).to(device=self.device)

            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except Exception as e:
            import traceback

            error_traceback = traceback.format_exc()
            print(f"Error creating model: {e}")
            print(f"Full traceback: {error_traceback}")
            raise

    def preprocess_frame(self, frame):
        """Preprocess frame for model input.

        :param frame: Input video frame as numpy array
        :type frame: np.ndarray
        :return: Preprocessed frame tensor ready for model input
        :rtype: torch.Tensor
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (600, 600))
        frame_tensor = (
            torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0)
        )
        frame_tensor = frame_tensor / 255.0
        # Apply ImageNet normalization
        bgr_means = torch.tensor([102.9801, 115.9465, 122.7717]).view(3, 1, 1)
        frame_tensor = frame_tensor * 255.0 - bgr_means
        return frame_tensor.to(self.device)

    def process_frame(self, frame, progress_callback=None):
        """Process a single frame and extract scene graph.

        :param frame: Input video frame as numpy array
        :type frame: np.ndarray
        :param progress_callback: Optional callback function for progress updates
        :type progress_callback: callable, optional
        :return: Tuple containing (results_dict, entry_dict, pred_dict)
        :rtype: tuple
        """
        try:
            # Progress callback: Frame preprocessing started
            if progress_callback:
                progress_callback("preprocessing", 0.1)

            im_data = self.preprocess_frame(frame)
            im_info = torch.tensor([[600, 600, 1.0]], dtype=torch.float32).to(
                self.device
            )

            # Progress callback: Object detection starting
            if progress_callback:
                progress_callback("detection", 0.3)

            with torch.no_grad():
                if self.conf.dataset == "EASG":
                    gt_grounding = []  # Empty grounding for inference
                    entry = self.object_detector(
                        im_data, im_info, gt_grounding, im_all=None
                    )
                    # Add verb features if available
                    if hasattr(self.dataset, "verb_feats"):
                        entry["features_verb"] = torch.zeros(1, 2048).to(self.device)
                else:
                    # Action Genome format
                    gt_boxes = torch.zeros([1, 1, 5]).to(self.device)
                    num_boxes = torch.zeros([1], dtype=torch.int64).to(self.device)
                    empty_annotation = []
                    entry = self.object_detector(
                        im_data,
                        im_info,
                        gt_boxes,
                        num_boxes,
                        empty_annotation,
                        im_all=None,
                    )

                if self.conf.mode == "sgdet" and self.conf.dataset != "EASG":
                    # Store original format
                    original_scores = (
                        entry["scores"].clone() if "scores" in entry else None
                    )
                    original_boxes = entry["boxes"].clone()

                    if "distribution" in entry:
                        entry["scores"] = entry["distribution"]  # Use 2D distribution
                    if len(entry["boxes"].shape) > 1 and entry["boxes"].shape[1] > 4:
                        entry["boxes"] = entry["boxes"][
                            :, 1:
                        ]  # Remove first column (batch index)

                    from m3sgg.utils.track import get_sequence

                    get_sequence(
                        entry,
                        [],  # empty annotation for sgdet mode
                        self.matcher,
                        torch.tensor([600, 600]).to(self.device),
                        "sgdet",
                    )

                    # Restore original format
                    if original_scores is not None:
                        entry["scores"] = original_scores
                    entry["boxes"] = original_boxes

                # Progress callback: Scene graph generation starting
                if progress_callback:
                    progress_callback("scene_graph", 0.7)

                # Scene graph generation
                pred = self.model(entry)

                # Progress callback: Frame processing completed
                if progress_callback:
                    progress_callback("completed", 1.0)

            return self.extract_results(entry, pred), entry, pred

        except Exception as e:
            # print(f"Frame processing error: {e}")
            return {"objects": 0, "relationships": 0, "error": str(e)}, None, None

    def simple_draw_bounding_boxes(self, frame, entry):
        """Simple bounding box drawing method using PIL.

        :param frame: Input video frame
        :type frame: np.ndarray
        :param entry: Model entry containing detection data
        :type entry: dict
        :return: Frame with drawn bounding boxes
        :rtype: np.ndarray
        """
        try:
            from .drawing_methods import simple_draw_bounding_boxes

            return simple_draw_bounding_boxes(frame, entry)
        except ImportError:
            # Fallback to alternative method
            from .drawing_methods import simple_draw_bounding_boxes_cv2_alternative

            return simple_draw_bounding_boxes_cv2_alternative(frame, entry)

    def simple_create_scene_graph_frame(self, frame, entry, pred):
        """Simple scene graph drawing method using matplotlib.

        :param frame: Input video frame
        :type frame: np.ndarray
        :param entry: Model entry containing detection data
        :type entry: dict
        :param pred: Model predictions containing relationship data
        :type pred: dict
        :return: Frame with scene graph visualization
        :rtype: np.ndarray
        """
        try:
            from .drawing_methods import simple_create_scene_graph_frame

            return simple_create_scene_graph_frame(frame, entry, pred)
        except ImportError:
            # Fallback to basic method
            return frame

    def draw_bounding_boxes(self, frame, entry, confidence_threshold=0.1):
        """Draw bounding boxes on frame.

        :param frame: Input video frame
        :type frame: np.ndarray
        :param entry: Model entry containing detection data
        :type entry: dict
        :param confidence_threshold: Minimum confidence for drawing boxes, defaults to 0.1
        :type confidence_threshold: float, optional
        :return: Frame with drawn bounding boxes and labels
        :rtype: np.ndarray
        """
        if entry is None or "boxes" not in entry:
            return frame

        frame_with_boxes = frame.copy()
        boxes = entry["boxes"]

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()

        labels = None
        scores = None

        if "pred_labels" in entry:
            labels = entry["pred_labels"]
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
        elif "labels" in entry:
            labels = entry["labels"]
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
        if "pred_scores" in entry:
            scores = entry["pred_scores"]
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
        elif "scores" in entry:
            scores = entry["scores"]
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
        if scores is not None:
            pass
        if "distribution" in entry:
            distribution = entry["distribution"]
            if isinstance(distribution, torch.Tensor):
                if scores is None:
                    scores = torch.max(distribution, dim=1)[0].cpu().numpy()
        if scores is None and boxes is not None:
            scores = np.ones(len(boxes))
        if labels is None and boxes is not None:
            labels = np.ones(len(boxes), dtype=int)
        if scores is not None and len(scores) > 0:
            high_conf_mask = scores > confidence_threshold
            if boxes is not None:
                boxes = boxes[high_conf_mask]
            if labels is not None:
                labels = labels[high_conf_mask]
            scores = scores[high_conf_mask]

        for i, box in enumerate(boxes):
            if len(box) >= 4:
                if len(box) == 5:
                    x1, y1, x2, y2 = box[1:5].astype(int)
                else:
                    x1, y1, x2, y2 = box[:4].astype(int)

                # assuming model uses 600x600
                h, w = frame.shape[:2]
                x1 = int(x1 * w / 600)
                y1 = int(y1 * h / 600)
                x2 = int(x2 * w / 600)
                y2 = int(y2 * h / 600)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                if x2 > x1 and y2 > y1:
                    if i < len(scores):
                        conf_score = scores[i]
                        if conf_score > 0.7:
                            color = (0, 255, 0)  # Green for high confidence
                        elif conf_score > 0.3:
                            color = (0, 255, 255)  # Yellow for medium confidence
                        else:
                            color = (0, 165, 255)  # Orange for low confidence
                    else:
                        color = (0, 255, 0)  # Default green

                    thickness = 3 if (i < len(scores) and scores[i] > 0.5) else 2
                    cv2.rectangle(
                        frame_with_boxes, (x1, y1), (x2, y2), color, thickness
                    )

                    if i < len(labels):
                        object_classes = None
                        if hasattr(self, "AG_dataset") and hasattr(
                            self.AG_dataset, "object_classes"
                        ):
                            object_classes = self.AG_dataset.object_classes
                        elif hasattr(self, "dataset") and hasattr(
                            self.dataset, "object_classes"
                        ):
                            object_classes = self.dataset.object_classes
                        elif hasattr(self, "dataset") and hasattr(
                            self.dataset, "obj_classes"
                        ):
                            object_classes = self.dataset.obj_classes
                        elif hasattr(self, "object_classes") and self.object_classes:
                            object_classes = self.object_classes
                        if object_classes and labels[i] < len(object_classes):
                            label_text = f"{object_classes[labels[i]]}"
                        else:
                            label_text = f"obj_{labels[i]}"
                        if i < len(scores):
                            label_text += f" {scores[i]:.2f}"

                        (text_width, text_height), baseline = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        cv2.rectangle(
                            frame_with_boxes,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width + 5, y1),
                            color,
                            -1,
                        )
                        cv2.putText(
                            frame_with_boxes,
                            label_text,
                            (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                else:
                    pass
            else:
                pass
        #
        return frame_with_boxes

    def extract_results(self, entry, pred):
        """Extract meaningful results from model outputs.

        :param entry: Model entry containing detection data
        :type entry: dict
        :param pred: Model predictions containing relationship data
        :type pred: dict
        :return: Dictionary with object count, relationship count, and confidence
        :rtype: dict
        """
        try:
            results = {"objects": 0, "relationships": 0, "confidence": 0.0}

            if "pred_scores" in entry:
                scores = entry["pred_scores"].cpu().numpy()
                # Count objects with confidence > lower threshold to match drawing
                results["objects"] = int(np.sum(scores > 0.1))
                results["confidence"] = (
                    float(np.mean(scores[scores > 0.1]))
                    if results["objects"] > 0
                    else 0.0
                )

            if "attention_distribution" in pred:
                attention_scores = pred["attention_distribution"].cpu().numpy()
                results["relationships"] = int(
                    np.sum(attention_scores.max(axis=-1) > 0.1)
                )

            return results

        except Exception as e:
            # print(f"Result extraction error: {e}")
            return {"objects": 0, "relationships": 0, "error": str(e)}

    def extract_bbox_centers(self, entry) -> List[Tuple[float, float]]:
        """Extract bounding box centers for scene graph node positioning

        :param entry: Model entry containing bounding box data
        :type entry: dict
        :return: List of (x, y) coordinates for bounding box centers
        :rtype: List[Tuple[float, float]]
        """
        if entry is None or "boxes" not in entry:
            return []

        boxes = entry["boxes"]
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()

        centers = []
        for box in boxes:
            if len(box) >= 4:
                # Handle batch dimension if present (5-element box: [batch_idx, x1, y1, x2, y2])
                if len(box) == 5:
                    x1, y1, x2, y2 = box[1:5]
                else:
                    x1, y1, x2, y2 = box[:4]

                # Calculate center coordinates
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                centers.append((float(center_x), float(center_y)))

        return centers

    def extract_bbox_info(
        self, entry, confidence_threshold=0.1
    ) -> List[Dict[str, Any]]:
        """Extract bounding box information for table display

        :param entry: Model entry containing detection data
        :type entry: dict
        :param confidence_threshold: Minimum confidence for inclusion, defaults to 0.1
        :type confidence_threshold: float, optional
        :return: List of bbox dictionaries with object names and confidence scores
        :rtype: List[Dict[str, Any]]
        """
        bbox_info = []

        if entry is None or "boxes" not in entry:
            return bbox_info

        boxes = entry["boxes"]
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()

        # Get labels and scores
        labels = None
        scores = None

        if "pred_labels" in entry:
            labels = entry["pred_labels"]
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
        elif "labels" in entry:
            labels = entry["labels"]
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

        if "pred_scores" in entry:
            scores = entry["pred_scores"]
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
        elif "scores" in entry:
            scores = entry["scores"]
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

        # Handle distribution case
        if "distribution" in entry and scores is None:
            distribution = entry["distribution"]
            if isinstance(distribution, torch.Tensor):
                scores = torch.max(distribution, dim=1)[0].cpu().numpy()

        # Create dummy data if missing
        if scores is None and boxes is not None:
            scores = np.ones(len(boxes))
        if labels is None and boxes is not None:
            labels = np.ones(len(boxes), dtype=int)

        # Filter by confidence threshold
        if scores is not None and len(scores) > 0:
            high_conf_mask = scores > confidence_threshold
            boxes = boxes[high_conf_mask] if boxes is not None else boxes
            labels = labels[high_conf_mask] if labels is not None else labels
            scores = scores[high_conf_mask]

        # Extract bbox information
        for i, box in enumerate(boxes):
            if len(box) >= 4:
                # Handle batch dimension if present
                if len(box) == 5:
                    x1, y1, x2, y2 = box[1:5]
                else:
                    x1, y1, x2, y2 = box[:4]

                # Get object class name
                object_name = "unknown"
                if labels is not None and i < len(labels):
                    # Check for dataset object classes (both AG_dataset and dataset attributes)
                    object_classes = None
                    if hasattr(self, "AG_dataset") and hasattr(
                        self.AG_dataset, "object_classes"
                    ):
                        object_classes = self.AG_dataset.object_classes
                    elif hasattr(self, "dataset") and hasattr(
                        self.dataset, "object_classes"
                    ):
                        object_classes = self.dataset.object_classes
                    elif hasattr(self, "dataset") and hasattr(
                        self.dataset, "obj_classes"
                    ):
                        object_classes = self.dataset.obj_classes
                    elif hasattr(self, "object_classes") and self.object_classes:
                        object_classes = self.object_classes

                    if object_classes and labels[i] < len(object_classes):
                        object_name = object_classes[labels[i]]
                    else:
                        object_name = f"class_{labels[i]}"

                # Get confidence score
                confidence = (
                    scores[i] if scores is not None and i < len(scores) else 0.0
                )

                bbox_info.append(
                    {
                        "object_name": object_name,
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    }
                )

        return bbox_info

    def extract_relationships(self, entry, pred) -> List[Dict[str, Any]]:
        """Extract relationship information for scene graph edges

        :param entry: Model entry containing pair indices and labels
        :type entry: dict
        :param pred: Model predictions containing relationship distributions
        :type pred: dict
        :return: List of relationship dictionaries with subject, object, and predicate info
        :rtype: List[Dict[str, Any]]
        """
        relationships = []

        if entry is None or pred is None:
            return relationships

        # Get pair indices (subject-object pairs)
        if "pair_idx" not in entry:
            return relationships

        pair_idx = entry["pair_idx"]
        if isinstance(pair_idx, torch.Tensor):
            pair_idx = pair_idx.cpu().numpy()

        # Get relationship predictions
        attention_dist = None
        spatial_dist = None

        if "attention_distribution" in pred:
            attention_dist = pred["attention_distribution"]
            if isinstance(attention_dist, torch.Tensor):
                attention_dist = attention_dist.cpu().numpy()

        if "spatial_distribution" in pred:
            spatial_dist = pred["spatial_distribution"]
            if isinstance(spatial_dist, torch.Tensor):
                spatial_dist = spatial_dist.cpu().numpy()

        # Get object labels for better relationship descriptions
        labels = None
        if "pred_labels" in entry:
            labels = entry["pred_labels"]
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

        # Process each relationship pair
        for i, (subj_idx, obj_idx) in enumerate(pair_idx):
            rel_info = {
                "subject_idx": int(subj_idx),
                "object_idx": int(obj_idx),
                "confidence": 0.0,
                "predicate": "interacts_with",  # Default predicate
            }

            # Get subject and object labels if available
            if labels is not None:
                if subj_idx < len(labels):
                    rel_info["subject_class"] = int(labels[subj_idx])
                if obj_idx < len(labels):
                    rel_info["object_class"] = int(labels[obj_idx])

            # Get the best relationship prediction
            if attention_dist is not None and i < len(attention_dist):
                # Get the attention relationship with highest confidence
                attention_scores = attention_dist[i]
                best_attention_idx = np.argmax(attention_scores)
                rel_info["attention_confidence"] = float(
                    attention_scores[best_attention_idx]
                )
                rel_info["attention_type"] = int(best_attention_idx)
                rel_info["confidence"] = max(
                    rel_info["confidence"], rel_info["attention_confidence"]
                )

            if spatial_dist is not None and i < len(spatial_dist):
                # Get the spatial relationship with highest confidence
                spatial_scores = spatial_dist[i]
                best_spatial_idx = np.argmax(spatial_scores)
                rel_info["spatial_confidence"] = float(spatial_scores[best_spatial_idx])
                rel_info["spatial_type"] = int(best_spatial_idx)
                rel_info["confidence"] = max(
                    rel_info["confidence"], rel_info["spatial_confidence"]
                )

            # Only include relationships with reasonable confidence
            if rel_info["confidence"] > 0.1:
                relationships.append(rel_info)

        return relationships

    def get_object_name(self, class_idx: int) -> str:
        """Get human-readable object name from class index.

        This method retrieves the human-readable name for an object class
        by looking up the class index in the dataset's object classes list.
        It handles multiple fallback sources for object class names.

        :param class_idx: Object class index
        :type class_idx: int
        :return: Human-readable object name
        :rtype: str
        """
        try:
            # Try to get object names from dataset
            if hasattr(self, "AG_dataset") and hasattr(
                self.AG_dataset, "object_classes"
            ):
                object_classes = self.AG_dataset.object_classes
            elif hasattr(self, "dataset") and hasattr(self.dataset, "object_classes"):
                object_classes = self.dataset.object_classes
            else:
                # Fallback object names - use Action Genome classes
                from m3sgg.core.constants import get_object_classes

                object_classes = get_object_classes("data/action_genome")

            if class_idx < len(object_classes):
                return object_classes[class_idx]
            else:
                return f"object_{class_idx}"

        except Exception as e:
            print(f"Error getting object name: {e}")
            return f"object_{class_idx}"

    def get_relationship_name(self, rel_type: int, rel_category: str) -> str:
        """Get human-readable relationship name from type and category.

        This method retrieves the human-readable name for a relationship type
        by looking up the type index in the appropriate relationship category
        (attention, spatial, or contacting relationships).

        :param rel_type: Relationship type index
        :type rel_type: int
        :param rel_category: Relationship category ('attention', 'spatial', 'contacting')
        :type rel_category: str
        :return: Human-readable relationship name
        :rtype: str
        """
        try:
            # Try to get relationship names from dataset
            relationships = None
            if hasattr(self, "AG_dataset") and hasattr(
                self.AG_dataset, f"{rel_category}_relationships"
            ):
                relationships = getattr(
                    self.AG_dataset, f"{rel_category}_relationships"
                )
            elif hasattr(self, "dataset") and hasattr(
                self.dataset, f"{rel_category}_relationships"
            ):
                relationships = getattr(self.dataset, f"{rel_category}_relationships")
            elif hasattr(self, "dataset") and hasattr(
                self.dataset, "relationship_classes"
            ):
                # Fallback: use the full relationship_classes list
                relationships = self.dataset.relationship_classes
                if rel_category == "attention":
                    relationships = relationships[0:3]
                elif rel_category == "spatial":
                    relationships = relationships[3:9]
                elif rel_category == "contacting":
                    relationships = relationships[9:]
            elif hasattr(self, "relationship_classes") and self.relationship_classes:
                # Final fallback: use loaded relationship classes
                relationships = self.relationship_classes
                if rel_category == "attention":
                    relationships = relationships[0:3]
                elif rel_category == "spatial":
                    relationships = relationships[3:9]
                elif rel_category == "contacting":
                    relationships = relationships[9:]
            else:
                # Ultimate fallback: use relationship classes by category
                from m3sgg.core.constants import get_relationship_classes_by_category

                relationships = get_relationship_classes_by_category(
                    rel_category, "data/action_genome"
                )

            if relationships and rel_type < len(relationships):
                return relationships[rel_type]
            else:
                return f"rel_{rel_type}"
        except (IndexError, AttributeError, Exception) as e:
            print(f"Error getting relationship name: {e}")
            return f"rel_{rel_type}"

    def create_scene_graph_frame(self, frame, entry, pred, frame_scale_factor=1.0):
        """Create a frame with 2D scene graph overlay.

        This method creates a visual representation of the scene graph by drawing
        nodes (objects) as circles and edges (relationships) as lines between them.
        Different colors and sizes are used to distinguish between human and object
        nodes, and relationship confidence is indicated by line thickness and color.

        :param frame: Original video frame
        :type frame: np.ndarray
        :param entry: Model entry containing detection data
        :type entry: dict
        :param pred: Model predictions containing relationship data
        :type pred: dict
        :param frame_scale_factor: Scale factor for coordinate conversion, defaults to 1.0
        :type frame_scale_factor: float, optional
        :return: Frame with scene graph overlay
        :rtype: np.ndarray
        """
        if entry is None or pred is None:
            return frame

        frame_with_sg = frame.copy()
        h, w = frame.shape[:2]

        # Extract node positions (bbox centers)
        centers = self.extract_bbox_centers(entry)
        if not centers:
            return frame

        # Scale centers to frame coordinates (assuming model uses 600x600)
        scaled_centers = []
        for cx, cy in centers:
            scaled_x = int(cx * w / 600)
            scaled_y = int(cy * h / 600)
            scaled_centers.append((scaled_x, scaled_y))

        # Extract relationships
        relationships = self.extract_relationships(entry, pred)

        # Draw relationship lines first (so they appear behind nodes)
        for rel in relationships:
            subj_idx = rel["subject_idx"]
            obj_idx = rel["object_idx"]

            if subj_idx < len(scaled_centers) and obj_idx < len(scaled_centers):
                subj_pos = scaled_centers[subj_idx]
                obj_pos = scaled_centers[obj_idx]

                # Draw line between subject and object
                confidence = rel.get("confidence", 0.0)

                # Line color based on confidence
                if confidence > 0.7:
                    line_color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.4:
                    line_color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    line_color = (0, 165, 255)  # Orange for low confidence

                # Line thickness based on confidence
                thickness = max(1, int(confidence * 3))

                cv2.line(frame_with_sg, subj_pos, obj_pos, line_color, thickness)

                # Add relationship label at midpoint
                mid_x = (subj_pos[0] + obj_pos[0]) // 2
                mid_y = (subj_pos[1] + obj_pos[1]) // 2

                # Create relationship label with decoded name
                rel_label = "interacts_with"  # Default
                if "attention_type" in rel:
                    rel_label = self.get_relationship_name(
                        rel["attention_type"], "attention"
                    )
                elif "spatial_type" in rel:
                    rel_label = self.get_relationship_name(
                        rel["spatial_type"], "spatial"
                    )
                elif "contacting_type" in rel:
                    rel_label = self.get_relationship_name(
                        rel["contacting_type"], "contacting"
                    )

                # Shorten long relationship names for display
                if len(rel_label) > 12:
                    rel_label = rel_label[:9] + "..."

                if confidence > 0.3:  # Only show labels for decent confidence
                    # Background for text
                    (text_w, text_h), baseline = cv2.getTextSize(
                        rel_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )
                    cv2.rectangle(
                        frame_with_sg,
                        (mid_x - text_w // 2 - 2, mid_y - text_h - 2),
                        (mid_x + text_w // 2 + 2, mid_y + 2),
                        (0, 0, 0),
                        -1,
                    )

                    # Text
                    cv2.putText(
                        frame_with_sg,
                        rel_label,
                        (mid_x - text_w // 2, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )

        # Draw nodes (circles at bbox centers)
        for i, (cx, cy) in enumerate(scaled_centers):
            # Node color - different for humans vs objects
            labels = entry.get("pred_labels", None)
            if labels is not None and isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

            if labels is not None and i < len(labels):
                # Human nodes are blue, object nodes are red
                if labels[i] == 1:  # Human class
                    node_color = (255, 100, 100)  # Light blue
                    node_radius = 8
                else:
                    node_color = (100, 100, 255)  # Light red
                    node_radius = 6
            else:
                node_color = (128, 128, 128)  # Gray for unknown
                node_radius = 6

            # Draw filled circle for node
            cv2.circle(frame_with_sg, (cx, cy), node_radius, node_color, -1)
            # Draw border
            cv2.circle(frame_with_sg, (cx, cy), node_radius, (255, 255, 255), 1)

            # Add node label (object class)
            if labels is not None and i < len(labels):
                # Try to get object class names from the dataset
                object_classes = None
                if hasattr(self, "AG_dataset") and hasattr(
                    self.AG_dataset, "object_classes"
                ):
                    object_classes = self.AG_dataset.object_classes
                elif hasattr(self, "dataset") and hasattr(
                    self.dataset, "object_classes"
                ):
                    object_classes = self.dataset.object_classes
                elif hasattr(self, "dataset") and hasattr(self.dataset, "obj_classes"):
                    object_classes = self.dataset.obj_classes
                elif hasattr(self, "object_classes") and self.object_classes:
                    object_classes = self.object_classes

                if object_classes and labels[i] < len(object_classes):
                    class_name = object_classes[labels[i]]
                    # Shortened class name for display
                    short_name = class_name[:8] if len(class_name) > 8 else class_name

                    # Text background
                    (text_w, text_h), baseline = cv2.getTextSize(
                        short_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )
                    cv2.rectangle(
                        frame_with_sg,
                        (cx - text_w // 2 - 2, cy - node_radius - text_h - 4),
                        (cx + text_w // 2 + 2, cy - node_radius),
                        (0, 0, 0),
                        -1,
                    )

                    # Text
                    cv2.putText(
                        frame_with_sg,
                        short_name,
                        (cx - text_w // 2, cy - node_radius - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
            else:
                # Fallback node number
                node_text = f"{i}"
                cv2.putText(
                    frame_with_sg,
                    node_text,
                    (cx - 5, cy - node_radius - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

        return frame_with_sg
