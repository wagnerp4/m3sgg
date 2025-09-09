import base64
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from streamlit_chat import message


# Try to import streamlit-player for enhanced video display
try:
    from streamlit_player import st_player
    STREAMLIT_PLAYER_AVAILABLE = True
    print("streamlit-player imported successfully")
except ImportError as e:
    print(f"streamlit-player not available: {e}")
    STREAMLIT_PLAYER_AVAILABLE = False

# Add the project root and src directory to Python path so m3sgg can be imported
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import class definitions from centralized constants module
# noinspection PyPep8
from m3sgg.core.constants import (
    get_object_classes,
    get_relationship_classes,
    get_relationship_classes_by_category
)

# M3SGG package is now properly installed and can be imported directly
try:
    from m3sgg.utils.track import get_sequence
    print("Successfully imported get_sequence")
except ImportError as e:
    print(f"Warning: Could not import get_sequence: {e}")

try:
    from m3sgg.utils.model_detector import get_model_info_from_checkpoint
    MODEL_DETECTOR_AVAILABLE = True
    print("Model detector imported successfully")
except ImportError as e:
    print(f"Could not import model_detector: {e}")
    MODEL_DETECTOR_AVAILABLE = False

try:
    from m3sgg.datasets.action_genome import AG
    print("Successfully imported AG dataset")
except ImportError as e:
    print(f"Warning: Could not import AG dataset: {e}")

try:
    from m3sgg.core.config.config import Config
    print("Successfully imported Config")
except ImportError as e:
    print(f"Warning: Could not import Config: {e}")

try:
    from m3sgg.utils.matcher import HungarianMatcher
    print("Successfully imported HungarianMatcher")
except ImportError as e:
    print(f"Warning: Could not import HungarianMatcher: {e}")

try:
    from m3sgg.language.conversation import SceneGraphChatInterface
    CHAT_INTERFACE_AVAILABLE = True
    print("Successfully imported SceneGraphChatInterface")
except ImportError as e:
    print(f"Warning: Could not import SceneGraphChatInterface: {e}")
    CHAT_INTERFACE_AVAILABLE = False

st.set_page_config(
    page_title="M3Sgg",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded", # collapsed
)


if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


if st.session_state.dark_mode:
    st.markdown(
        """
        <style>
            .main-header {
                font-size: 3rem;
                color: #4a9eff;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #2d3748;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 5px solid #4a9eff;
                color: #e2e8f0;
            }
            .stAlert {
                margin-top: 1rem;
            }
            .stApp {
                background-color: #1a202c;
                color: #e2e8f0;
            }
            .stSidebar {
                background-color: #2d3748;
            }
            .stSelectbox > div > div {
                background-color: #2d3748;
                color: #e2e8f0;
            }
            .stTextInput > div > div > input {
                background-color: #2d3748;
                color: #e2e8f0;
                border-color: #4a5568;
            }
            .stTextArea > div > div > textarea {
                background-color: #2d3748;
                color: #e2e8f0;
                border-color: #4a5568;
            }
            .stButton > button {
                background-color: #4a9eff;
                color: #1a202c;
                border: none;
            }
            .stButton > button:hover {
                background-color: #3182ce;
            }
            .stTabs [data-baseweb="tab-list"] {
                background-color: #2d3748;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #2d3748;
                color: #e2e8f0;
            }
            .stTabs [aria-selected="true"] {
                background-color: #4a9eff;
                color: #1a202c;
            }
            .stDataFrame {
                background-color: #2d3748;
                color: #e2e8f0;
            }
            .stExpander {
                background-color: #2d3748;
                color: #e2e8f0;
            }
            .stMarkdown {
                color: #e2e8f0;
            }
            .stSuccess {
                background-color: #2d5016;
                color: #9ae6b4;
            }
            .stError {
                background-color: #742a2a;
                color: #feb2b2;
            }
            .stWarning {
                background-color: #744210;
                color: #fbd38d;
            }
            .stInfo {
                background-color: #2c5282;
                color: #90cdf4;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
            .main-header {
                font-size: 3rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 5px solid #1f77b4;
            }
            .stAlert {
                margin-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


class StreamlitVideoProcessor:
    def __init__(self, model_path: str):
        """Video processor for Streamlit integration with VidSgg pipeline"""
        # Force CPU and disable accelerators to avoid device errors
        self.device = torch.device("cpu")
        
        # Disable DirectML and other accelerators
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Monkey patch torch.cuda to prevent accelerator access
        self._patch_torch_cuda()
        
        self.model_path = model_path
        self.object_classes = self._load_object_classes()
        self.relationship_classes = self._load_relationship_classes()
        self.setup_models()
    
    def _patch_torch_cuda(self):
        """Monkey patch torch.cuda to prevent accelerator access"""
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
        """Load object classes from file as fallback"""
        return get_object_classes("data/action_genome")
    
    def _load_relationship_classes(self):
        """Load relationship classes from file as fallback"""
        return get_relationship_classes("data/action_genome")

    def setup_models(self):
        """Initialize models for video processing with automatic model detection"""
        try:
            from m3sgg.datasets.action_genome import AG
            from m3sgg.datasets.easg import EASG
            from m3sgg.core.config.config import Config
            from m3sgg.utils.matcher import HungarianMatcher

            from m3sgg.core.detectors.faster_rcnn import detector
            from m3sgg.core.detectors.easg.object_detector_EASG import detector as detector_EASG

            # Detect model type from checkpoint
            if MODEL_DETECTOR_AVAILABLE:
                model_info = get_model_info_from_checkpoint(self.model_path)
                detected_model_type = model_info["model_type"]
                detected_dataset = model_info["dataset"]
                
                if not detected_model_type:
                    raise ValueError(f"Could not detect model type from checkpoint: {self.model_path}")
                
                print(f"Detected model type: {detected_model_type}")
                print(f"Detected dataset: {detected_dataset}")
            else:
                # Fallback to default values if model detector is not available
                detected_model_type = "sttran"  # Default fallback
                detected_dataset = "action_genome"  # Default fallback
                print(f"Model detector unavailable, using defaults: {detected_model_type}, {detected_dataset}")

            self.conf = Config()
            self.conf.mode = "sgdet"
            
            # Set dataset-specific configuration
            if detected_dataset == "EASG":
                self.conf.data_path = "data/EASG"
                self.conf.dataset = "EASG"
            else:
                self.conf.data_path = "data/action_genome"
                self.conf.dataset = "action_genome"

            # Initialize dataset
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

            # Object Detector
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
            self.model = self._create_model_from_type(detected_model_type, detected_dataset)

            # Load checkpoint
            if os.path.exists(self.model_path):
                ckpt = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(ckpt["state_dict"], strict=False)
                # Explicitly move model to CPU
                self.model = self.model.to(self.device)
                print(f"Loaded {detected_model_type} model from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file {self.model_path} not found!")

            self.model.eval()
            self.matcher = HungarianMatcher(0.5, 1, 1, 0.5).to(device=self.device)
            self.matcher.eval()

        except Exception as e:
            st.error(f"Failed to setup models: {e}")
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
        if model_type == "sttran":
            if dataset_type == "EASG":
                from m3sgg.core.detectors.easg.sttran_EASG import STTran as STTran_EASG
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
                pred_contact_threshold=getattr(self.conf, "pred_contact_threshold", 0.5),
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

    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
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

    def process_frame(self, frame):
        """Process a single frame and extract scene graph"""
        try:
            im_data = self.preprocess_frame(frame)
            im_info = torch.tensor([[600, 600, 1.0]], dtype=torch.float32).to(
                self.device
            )

            with torch.no_grad():
                # Handle different datasets
                if self.conf.dataset == "EASG":
                    # EASG uses different input format
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
                        im_data, im_info, gt_boxes, num_boxes, empty_annotation, im_all=None
                    )

                if "boxes" in entry and entry["boxes"] is not None:
                    # print(f"Raw detections: {entry['boxes'].shape[0]} boxes")
                    if "pred_scores" in entry:
                        # raw_scores = entry["pred_scores"].cpu().numpy()
                        # print(
                        #     f"Raw scores range: {raw_scores.min():.3f} - {raw_scores.max():.3f}"
                        # )
                        # print(f"Scores above 0.1: {(raw_scores > 0.1).sum()}")
                        # print(f"Scores above 0.3: {(raw_scores > 0.3).sum()}")
                        pass

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

                    get_sequence(
                        entry,
                        [],  # empty annotation for sgdet mode
                        self.matcher,
                        torch.tensor([600, 600]).to(
                            self.device
                        ),  # TODO: generalize image dimensions
                        "sgdet",
                    )

                    # Restore original format
                    if original_scores is not None:
                        entry["scores"] = original_scores
                    entry["boxes"] = original_boxes
                    # print(f"After get_sequence: boxes shape = {entry['boxes'].shape}")

                # Scene graph generation
                pred = self.model(entry)

            return self.extract_results(entry, pred), entry, pred

        except Exception as e:
            # print(f"Frame processing error: {e}")
            return {"objects": 0, "relationships": 0, "error": str(e)}, None, None

    def simple_draw_bounding_boxes(self, frame, entry):
        """Simple bounding box drawing method using PIL"""
        try:
            from simple_drawing_methods import simple_draw_bounding_boxes
            return simple_draw_bounding_boxes(frame, entry)
        except ImportError:
            # Fallback to alternative method
            from simple_drawing_methods import simple_draw_bounding_boxes_cv2_alternative
            return simple_draw_bounding_boxes_cv2_alternative(frame, entry)

    def simple_create_scene_graph_frame(self, frame, entry, pred):
        """Simple scene graph drawing method using matplotlib"""
        try:
            from simple_drawing_methods import simple_create_scene_graph_frame
            return simple_create_scene_graph_frame(frame, entry, pred)
        except ImportError:
            # Fallback to basic method
            return frame

    def draw_bounding_boxes(self, frame, entry, confidence_threshold=0.1):
        """Draw bounding boxes on frame"""
        if entry is None or "boxes" not in entry:
            return frame

        frame_with_boxes = frame.copy()
        boxes = entry["boxes"]

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()

        # Get labels and scores - try different fields
        labels = None
        scores = None

        # Try to get the most comprehensive set of detections
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

        # Debug: Show entry fields and their contents
        # print(f"DRAW_BBOX: Entry fields: {list(entry.keys())}")
        # print(
        #     f"DRAW_BBOX: Boxes shape: {boxes.shape if hasattr(boxes, 'shape') else len(boxes)}"
        # )
        if labels is not None:
            # print(
            #     f"DRAW_BBOX: Labels shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}"
            # )
            # Add detailed object list logging
            if hasattr(self, "AG_dataset") and labels is not None:
                # print("DRAW_BBOX: === DETECTED OBJECTS LIST ===")
                for i, label_idx in enumerate(labels):
                    if label_idx < len(self.AG_dataset.object_classes):
                        # object_name = self.AG_dataset.object_classes[label_idx]
                        # score_str = (
                        #     f" (score: {scores[i]:.3f})"
                        #     if scores is not None and i < len(scores)
                        #     else ""
                        # )
                        # print(f"  {i+1}. {object_name}{score_str}")
                        pass
                    else:
                        # print(f"  {i+1}. unknown_class_{label_idx}")
                        pass
                # print("DRAW_BBOX: === END OBJECT LIST ===")
        if scores is not None:
            # print(
            #     f"DRAW_BBOX: Scores shape: {scores.shape if hasattr(scores, 'shape') else len(scores)}"
            # )
            # print(f"DRAW_BBOX: All scores: {scores}")
            # print(
            #     f"DRAW_BBOX: Score statistics - Min: {scores.min():.3f}, Max: {scores.max():.3f}, Mean: {scores.mean():.3f}"
            # )
            pass

        # Check if we have distribution instead
        if "distribution" in entry:
            distribution = entry["distribution"]
            if isinstance(distribution, torch.Tensor):
                # print(f"Distribution shape: {distribution.shape}")
                # Use distribution to get scores
                if scores is None:
                    scores = torch.max(distribution, dim=1)[0].cpu().numpy()
                    # print(f"Generated scores from distribution: {scores}")

        # If we still don't have scores, create dummy ones
        if scores is None and boxes is not None:
            scores = np.ones(len(boxes))
            # print(f"Using dummy scores for {len(boxes)} boxes")

        # If we don't have labels, create dummy ones
        if labels is None and boxes is not None:
            labels = np.ones(len(boxes), dtype=int)
            # print(f"Using dummy labels for {len(boxes)} boxes")

        # Debug: Show all detections before filtering
        if scores is not None and len(scores) > 0:
            # print(f"Total detections before filtering: {len(scores)}")

            # Filter detections with lower confidence threshold
            high_conf_mask = scores > confidence_threshold
            # print(
            #     f"Detections after confidence filtering (>{confidence_threshold}): {high_conf_mask.sum()}"
            # )

            if boxes is not None:
                boxes = boxes[high_conf_mask]
            if labels is not None:
                labels = labels[high_conf_mask]
            scores = scores[high_conf_mask]

            # print(f"Final boxes to draw: {len(boxes) if boxes is not None else 0}")

        # Draw boxes
        # print(
        #     f"DRAW_BBOX: About to draw {len(boxes) if boxes is not None else 0} boxes"
        # )
        for i, box in enumerate(boxes):
            # print(f"DRAW_BBOX: Processing box {i+1}/{len(boxes)}: {box}")
            if len(box) >= 4:
                # Handle batch dimension if present
                if len(box) == 5:
                    x1, y1, x2, y2 = box[1:5].astype(int)
                    # print(
                    #     f"DRAW_BBOX: Box {i+1} (with batch): ({x1}, {y1}, {x2}, {y2})"
                    # )
                else:
                    x1, y1, x2, y2 = box[:4].astype(int)
                    # print(f"DRAW_BBOX: Box {i+1} (no batch): ({x1}, {y1}, {x2}, {y2})")

                # Scale to frame size (assuming model uses 600x600)
                h, w = frame.shape[:2]
                x1 = int(x1 * w / 600)
                y1 = int(y1 * h / 600)
                x2 = int(x2 * w / 600)
                y2 = int(y2 * h / 600)

                # print(
                #     f"DRAW_BBOX: Box {i+1} scaled to frame ({w}x{h}): ({x1}, {y1}, {x2}, {y2})"
                # )

                # Ensure coordinates are valid
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                # print(f"DRAW_BBOX: Box {i+1} after clipping: ({x1}, {y1}, {x2}, {y2})")

                if x2 > x1 and y2 > y1:
                    # print(f"DRAW_BBOX:  Drawing box {i+1} - valid coordinates")
                    # Choose color based on confidence
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

                    # Draw rectangle with thickness based on confidence
                    thickness = 3 if (i < len(scores) and scores[i] > 0.5) else 2
                    cv2.rectangle(
                        frame_with_boxes, (x1, y1), (x2, y2), color, thickness
                    )

                    # Add label if available
                    if i < len(labels):
                        # Try to get object class names from the dataset
                        object_classes = None
                        if hasattr(self, "AG_dataset") and hasattr(self.AG_dataset, "object_classes"):
                            object_classes = self.AG_dataset.object_classes
                        elif hasattr(self, "dataset") and hasattr(self.dataset, "object_classes"):
                            object_classes = self.dataset.object_classes
                        elif hasattr(self, "dataset") and hasattr(self.dataset, "obj_classes"):
                            object_classes = self.dataset.obj_classes
                        elif hasattr(self, "object_classes") and self.object_classes:
                            object_classes = self.object_classes
                        
                        if object_classes and labels[i] < len(object_classes):
                            label_text = f"{object_classes[labels[i]]}"
                        else:
                            label_text = f"obj_{labels[i]}"

                        if i < len(scores):
                            label_text += f" {scores[i]:.2f}"

                        # Background for better text visibility
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

                    # print(f"DRAW_BBOX:  Successfully drew box {i+1}")
                else:
                    # print(
                    #     f"DRAW_BBOX:  Skipped box {i+1} - invalid coordinates (x2 <= x1 or y2 <= y1)"
                    # )
                    pass
            else:
                # print(
                #     f"DRAW_BBOX:  Skipped box {i+1} - insufficient coordinates (need at least 4)"
                # )
                pass

        # print(
        #     f"DRAW_BBOX: === Finished drawing {len(boxes) if boxes is not None else 0} boxes ==="
        # )
        return frame_with_boxes

    def extract_results(self, entry, pred):
        """Extract meaningful results from model outputs"""
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
                    if hasattr(self, "AG_dataset") and hasattr(self.AG_dataset, "object_classes"):
                        object_classes = self.AG_dataset.object_classes
                    elif hasattr(self, "dataset") and hasattr(self.dataset, "object_classes"):
                        object_classes = self.dataset.object_classes
                    elif hasattr(self, "dataset") and hasattr(self.dataset, "obj_classes"):
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
        """Get human-readable object name from class index
        
        :param class_idx: Object class index
        :type class_idx: int
        :return: Human-readable object name
        :rtype: str
        """
        try:
            # Try to get object names from dataset
            if hasattr(self, "AG_dataset") and hasattr(self.AG_dataset, "object_classes"):
                object_classes = self.AG_dataset.object_classes
            elif hasattr(self, "dataset") and hasattr(self.dataset, "object_classes"):
                object_classes = self.dataset.object_classes
            else:
                # Fallback object names - use Action Genome classes
                object_classes = get_object_classes("data/action_genome")
            
            if class_idx < len(object_classes):
                return object_classes[class_idx]
            else:
                return f"object_{class_idx}"
                
        except Exception as e:
            print(f"Error getting object name: {e}")
            return f"object_{class_idx}"

    def get_relationship_name(self, rel_type: int, rel_category: str) -> str:
        """Get human-readable relationship name from type and category

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
            if hasattr(self, "AG_dataset") and hasattr(self.AG_dataset, f"{rel_category}_relationships"):
                relationships = getattr(self.AG_dataset, f"{rel_category}_relationships")
            elif hasattr(self, "dataset") and hasattr(self.dataset, f"{rel_category}_relationships"):
                relationships = getattr(self.dataset, f"{rel_category}_relationships")
            elif hasattr(self, "dataset") and hasattr(self.dataset, "relationship_classes"):
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
                relationships = get_relationship_classes_by_category(rel_category, "data/action_genome")
            
            if relationships and rel_type < len(relationships):
                return relationships[rel_type]
            else:
                return f"rel_{rel_type}"
        except (IndexError, AttributeError, Exception) as e:
            print(f"Error getting relationship name: {e}")
            return f"rel_{rel_type}"

    def create_scene_graph_frame(self, frame, entry, pred, frame_scale_factor=1.0):
        """Create a frame with 2D scene graph overlay

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
                if hasattr(self, "AG_dataset") and hasattr(self.AG_dataset, "object_classes"):
                    object_classes = self.AG_dataset.object_classes
                elif hasattr(self, "dataset") and hasattr(self.dataset, "object_classes"):
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


def get_best_video_format() -> str:
    """Get the best video format for browser compatibility
    
    :return: Best video format extension
    :rtype: str
    """
    # Try formats in order of browser compatibility
    formats_to_try = [".mp4", ".avi", ".mov"]
    return formats_to_try[0]  # Default to mp4


def convert_video_for_browser(video_path: str) -> str:
    """Convert video to browser-friendly format if needed
    
    :param video_path: Path to input video file
    :type video_path: str
    :return: Path to converted video file (or original if conversion not needed)
    :rtype: str
    """
    try:
        print(f"Starting video conversion for: {video_path}")
        
        # Check if video is already browser-friendly
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            # Check codec
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            print(f"Original video fourcc: {fourcc}")
            cap.release()
            
            # H264 and avc1 codecs (most browser-compatible)
            if fourcc == 875967048.0:  # H264
                print("Video is already H264, no conversion needed")
                return video_path
            elif fourcc == 875967080.0:  # avc1
                print("Video is already avc1, no conversion needed")
                return video_path
        
        # Convert to H264 if not already
        print(f"Converting video to browser-friendly format: {video_path}")
        
        # Create output path
        base_path = video_path.rsplit(".", 1)[0]
        output_path = f"{base_path}_browser_friendly.mp4"
        print(f"Output path: {output_path}")
        
        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open input video")
            return video_path
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {fps} FPS, {width}x{height}, {frame_count} frames")
        
        # Create output with mp4v codec (case-sensitive)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if out.isOpened():
            print("Video writer opened successfully")
            frame_num = 0
            # Copy all frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_num += 1
                if frame_num % 10 == 0:
                    print(f"Processed {frame_num} frames")
            
            out.release()
            cap.release()
            print(f"Conversion completed, processed {frame_num} frames")
            
            # Verify output
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                test_cap.release()
                if ret and test_frame is not None:
                    print(f"Successfully converted video: {output_path}")
                    print(f"Converted video size: {os.path.getsize(output_path)} bytes")
                    return output_path
                else:
                    print("Converted video verification failed")
            else:
                print("Failed to open converted video for verification")
            
            # If conversion failed, return original
            try:
                os.unlink(output_path)
                print("Removed failed conversion file")
            except:
                pass
        else:
            print("Failed to open video writer")
        
        cap.release()
        print("Returning original video path")
        return video_path
        
    except Exception as e:
        print(f"Error converting video: {e}")
        import traceback
        traceback.print_exc()
        return video_path


def validate_video_file(video_path: str) -> bool:
    """Validate that a video file is readable and contains valid frames
    
    :param video_path: Path to video file to validate
    :type video_path: str
    :return: True if video is valid, False otherwise
    :rtype: bool
    """
    try:
        if not os.path.exists(video_path):
            return False
            
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size < 1000:  # Minimum 1KB
            return False
            
        # Test with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
            
        # Try to read at least one frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
        
    except Exception:
        return False


def create_processed_video_with_bboxes(
    video_path: str, model_path: str, max_frames: int = 30
) -> tuple[bool, str]:
    """Create a new video file with bounding boxes drawn on frames"""
    try:
        import tempfile
        import os
        
        # Create project-local temp directory
        temp_dir = os.path.join("data", "temp_vid")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create temporary file in project temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=temp_dir) as tmp_file:
            output_path = tmp_file.name
        
        processor = StreamlitVideoProcessor(model_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return False, ""

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Try different codecs in order of web compatibility
        # Prioritize mp4v (case-sensitive) for better browser compatibility
        codecs_to_try = ["avc1", "mp4v", "XVID", "MJPG", "H264"]
        
        out = None
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"Successfully opened video writer with codec: {codec}")
                    break
                else:
                    out.release()
                    out = None
            except Exception as e:
                print(f"Failed to open video writer with codec {codec}: {e}")
                if out:
                    out.release()
                    out = None

        if not out or not out.isOpened():
            raise ValueError("Could not initialize video writer with any codec")

        frame_count = 0

        while frame_count < max_frames and cap.isOpened():
            if not (ret := cap.read())[0]:
                break
            frame = ret[1]

            # Process frame with SGG
            frame_results, entry, pred = processor.process_frame(frame)

            if entry is not None:
                frame_with_boxes = processor.simple_draw_bounding_boxes(frame, entry)
                # Debug: Check if frame has content
                if frame_count < 3:  # Only log first few frames
                    print(f"Frame {frame_count}: Original shape: {frame.shape}, Processed shape: {frame_with_boxes.shape}")
                    print(f"Frame {frame_count}: Entry keys: {list(entry.keys()) if entry else 'None'}")
                    if "boxes" in entry and entry["boxes"] is not None:
                        print(f"Frame {frame_count}: Found {len(entry['boxes'])} boxes")
            else:
                frame_with_boxes = frame
                if frame_count < 3:
                    print(f"Frame {frame_count}: No entry data, using original frame")

            # Write frame (ignore write_success on Windows as it often returns False incorrectly)
            write_success = out.write(frame_with_boxes)
            # Note: On Windows, out.write() often returns False even when successful
            # We'll validate the final video file instead
            frame_count += 1

        cap.release()
        out.release()

        # Validate the created video
        if validate_video_file(output_path):
            file_size = os.path.getsize(output_path)
            print(f"Bounding box video created successfully: {output_path}, size: {file_size} bytes")
            return True, output_path
        else:
            print(f"Bounding box video file created but validation failed: {output_path}")
            return False, ""

    except Exception as e:
        print(f"Error creating bounding box video: {e}")
        import traceback
        traceback.print_exc()
        return False, ""


def create_processed_video_with_scene_graph(
    video_path: str, model_path: str, max_frames: int = 30
) -> tuple[bool, str]:
    """Create a new video file with scene graph overlay drawn on frames using simplified approach

    :param video_path: Path to input video file
    :type video_path: str
    :param model_path: Path to model checkpoint
    :type model_path: str
    :param max_frames: Maximum number of frames to process, defaults to 30
    :type max_frames: int, optional
    :return: Tuple of (success, output_path) where success is bool and output_path is str
    :rtype: tuple[bool, str]
    """
    try:
        import cv2
        import tempfile
        import os
        
        # Create project-local temp directory
        temp_dir = os.path.join("data", "temp_vid")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create temporary file in project temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=temp_dir) as tmp_file:
            output_path = tmp_file.name
        
        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, ""
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video properties: {fps} FPS, {width}x{height}, {total_frames} total frames")
        
        # Try different codecs in order of web compatibility
        codecs_to_try = ["avc1", "mp4v", "XVID", "MJPG", "H264"]
        
        out = None
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    break
                else:
                    out.release()
                    out = None
            except Exception as e:
                if out:
                    out.release()
                    out = None

        if not out or not out.isOpened():
            cap.release()
            return False, ""
        
        frame_count = 0
        processed_frames = 0
        
        # Create StreamlitVideoProcessor instance for proper scene graph generation
        processor = StreamlitVideoProcessor(model_path)
        
        # Process frames with scene graph overlay
        while frame_count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with model to get predictions
            frame_results, entry, pred = processor.process_frame(frame)
            
            if entry is not None and pred is not None:
                # Use the proper scene graph drawing method
                frame_with_sg = processor.create_scene_graph_frame(frame, entry, pred)
            else:
                # Fallback to simple visualization if no model data
                frame_with_sg = frame.copy()
                h, w = frame_with_sg.shape[:2]
                
                # Add simple visual elements to represent scene graph
                cv2.putText(frame_with_sg, f"Scene Graph Frame {frame_count} (No Model Data)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw some simple shapes to represent objects and relationships
                cv2.rectangle(frame_with_sg, (50, 50), (150, 150), (255, 0, 0), 2)
                cv2.putText(frame_with_sg, "Object 1", (55, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                cv2.rectangle(frame_with_sg, (w-150, 50), (w-50, 150), (0, 0, 255), 2)
                cv2.putText(frame_with_sg, "Object 2", (w-145, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                cv2.line(frame_with_sg, (150, 100), (w-150, 100), (0, 255, 255), 2)
                cv2.putText(frame_with_sg, "relationship", (w//2-50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                cv2.circle(frame_with_sg, (w//2, h-100), 30, (255, 255, 0), 2)
                cv2.putText(frame_with_sg, "Node", (w//2-20, h-95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Write processed frame
            out.write(frame_with_sg)
            processed_frames += 1
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"✅ Video processing completed: {processed_frames} frames processed")
        
        # Verify output file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Output file created: {file_size:,} bytes")
            
            # Test if output video is readable
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                ret, test_frame = test_cap.read()
                test_cap.release()
                if ret:
                    print(f"✅ Output video is readable: shape={test_frame.shape}")
                    return True, output_path
                else:
                    print(f"❌ Output video created but not readable")
                    return False, ""
            else:
                print(f"❌ Output video created but cannot be opened")
                return False, ""
        else:
            print(f"❌ Output file was not created at: {output_path}")
            print(f"❌ Current working directory: {os.getcwd()}")
            print(f"❌ Directory contents: {os.listdir('.')}")
            return False, ""
            
    except Exception as e:
        print(f"❌ Error processing scene graph video: {e}")
        import traceback
        traceback.print_exc()
        return False, ""


def find_available_checkpoints() -> Dict[str, str]:
    """Find available model checkpoints"""
    checkpoints = {}
    default_checkpoint = Path(
        "data/checkpoints/action_genome/sgdet_test/model_best.tar"
    )

    if default_checkpoint.exists():
        checkpoints["action_genome/sgdet_test (default)"] = str(default_checkpoint)

    output_dir = Path("output")
    if output_dir.exists():
        checkpoints.update(
            {
                f"{dataset_dir.name}/{model_dir.name}": str(run_dir / "model_best.tar")
                for dataset_dir in output_dir.iterdir()
                if dataset_dir.is_dir()
                for model_dir in dataset_dir.iterdir()
                if model_dir.is_dir()
                for run_dir in model_dir.iterdir()
                if run_dir.is_dir()
                if (run_dir / "model_best.tar").exists()
            }
        )

    return checkpoints


def process_video_with_sgg(
    video_path: str, model_path: str, max_frames: int = 30
) -> Dict[str, Any]:
    """Process video using scene graph generation"""
    try:
        # Initialize video processor
        processor = StreamlitVideoProcessor(model_path)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        results = {
            "total_frames": total_frames,
            "fps": fps,
            "processed_frames": 0,
            "detections": [],
            "relationships": [],
            "confidences": [],
            "frame_times": [],
            "errors": [],
            "frame_objects": [],  # Store object info for each frame
            "frame_relationships": [],  # Store relationship info for each frame
        }

        frame_count = 0

        while frame_count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with SGG
            frame_results, entry, pred = processor.process_frame(frame)

            results["detections"].append(frame_results.get("objects", 0))
            results["relationships"].append(frame_results.get("relationships", 0))
            results["confidences"].append(frame_results.get("confidence", 0.0))
            results["frame_times"].append(frame_count / fps)

            # Extract bbox info for each frame
            if entry is not None:
                bbox_info = processor.extract_bbox_info(entry, confidence_threshold=0.1)
                results["frame_objects"].append(bbox_info)

                # Extract relationship info for each frame
                relationship_info = processor.extract_relationships(entry, pred)
                results["frame_relationships"].append(relationship_info)

                # Store first frame bbox info in session state for backward compatibility
                if frame_count == 0:
                    st.session_state["bbox_info"] = bbox_info
                    st.session_state["relationship_info"] = relationship_info
            else:
                results["frame_objects"].append([])
                results["frame_relationships"].append([])

            if "error" in frame_results:
                results["errors"].append(
                    f"Frame {frame_count}: {frame_results['error']}"
                )

            frame_count += 1

        cap.release()
        results["processed_frames"] = frame_count

        return results

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None


def main():
    #--------------------------------
    # Header
    st.markdown(
        '<h1 class="main-header"> M3SGG</h1>',
        unsafe_allow_html=True,
    )
    # st.markdown("Video Scene Graph Generation with Deep Learning Models")
    st.markdown("---")

    #--------------------------------
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("Model Configuration")
        uploaded_file = st.file_uploader(
            "Upload Model Checkpoint",
            type=['tar', 'pth', 'pt'],
            help="Drag and drop a model checkpoint file (.tar, .pth, or .pt). Large files (>200MB) are supported.",
            key="checkpoint_uploader",
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # TODO: Store better
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            model_path = str(temp_path)
            st.session_state["model_path"] = model_path
            
            # Display model info
            if MODEL_DETECTOR_AVAILABLE:
                try:
                    model_info = get_model_info_from_checkpoint(model_path)
                    
                    st.success("Checkpoint uploaded successfully!")
                    st.write(f"**File:** {uploaded_file.name}")
                    st.write(f"**Model Type:** {model_info['model_type'] or 'Unknown'}")
                    st.write(f"**Dataset:** {model_info['dataset'] or 'Unknown'}")
                    st.write(f"**Model Class:** {model_info['model_class'] or 'Unknown'}")
                    
                except Exception as e:
                    st.error(f"Error analyzing checkpoint: {e}")
                    model_path = None
            else:
                st.success("Checkpoint uploaded successfully!")
                st.write(f"**File:** {uploaded_file.name}")
                st.warning("Model analysis unavailable - model_detector module not found")
                st.info("The checkpoint will still work, but automatic model detection is disabled.")
        else:
            # TODO: Test with different checkpoints
            # TODO: Use better default path
            checkpoints = find_available_checkpoints()
            if checkpoints:
                selected_model = st.selectbox(
                    "Or Select Existing Checkpoint",
                    list(checkpoints.keys()),
                    help="Available trained models",
                )
                model_path = checkpoints[selected_model]
                st.session_state["model_path"] = model_path
                if "default" in selected_model.lower():
                    st.success(" Default checkpoint loaded")
            else:
                st.warning("No trained models found in expected locations")
                st.info(
                    "Upload a checkpoint file above or place model at `data/checkpoints/action_genome/sgdet_test/model_best.tar`"
                )
                model_path = st.text_input(
                    "Or Enter Model Path",
                    value="data/checkpoints/action_genome/sgdet_test/model_best.tar",
                    placeholder="Path to model checkpoint (.tar or .pth)",
                    help="Provide path to a trained model checkpoint",
                )
                if model_path:
                    st.session_state["model_path"] = model_path

        st.subheader("Processing Settings")
        max_slider_value = 1000
        total_frames_info = ""

        if "video_total_frames" in st.session_state:
            total_frames = st.session_state["video_total_frames"]
            max_slider_value = min(total_frames, 1000)
            total_frames_info = f" (Video has {total_frames} total frames)"

        max_frames = st.slider(
            "Max Frames to Process",
            min_value=1,
            max_value=max_slider_value,
            value=min(30, max_slider_value),
            help=f"Maximum number of frames to process{total_frames_info}",
        )

        # TODO: Add back in
        # confidence_threshold = st.slider(
        #     "Confidence Threshold",
        #     0.0,
        #     1.0,
        #     0.5,
        #     help="Minimum confidence for object detection",
        # )

        # if st.button("Reset Settings"):
        #     st.rerun()

        st.markdown("---")
        st.subheader("Export")
        export_format = st.selectbox("Export Format", ["JSON", "CSV", "XML"])

        # Download Section
        if st.button("Download Results"):

            if "results" in st.session_state:
                results = st.session_state["results"]
                export_data = {
                    "video_metadata": {
                        "total_frames": results["total_frames"],
                        "fps": results["fps"],
                        "duration_seconds": results["total_frames"] / results["fps"]
                        if results["fps"] > 0
                        else 0,
                        "processed_frames": results["processed_frames"],
                    },
                    "statistics": {
                        "avg_objects_per_frame": np.mean(results["detections"])
                        if results["detections"]
                        else 0,
                        "avg_relationships_per_frame": np.mean(results["relationships"])
                        if results["relationships"]
                        else 0,
                        "avg_confidence": np.mean(results["confidences"])
                        if results["confidences"]
                        else 0,
                        "error_rate_percent": (
                            len(results.get("errors", []))
                            / results["processed_frames"]
                            * 100
                        )
                        if results["processed_frames"] > 0
                        else 0,
                        "total_processing_time": sum(results["frame_times"])
                        if results["frame_times"]
                        else 0,
                    },
                    "frame_details": [],
                }

                for i in range(len(results["detections"])):
                    frame_data = {
                        "frame_number": i + 1,
                        "objects_detected": results["detections"][i]
                        if i < len(results["detections"])
                        else 0,
                        "relationships_found": results["relationships"][i]
                        if i < len(results["relationships"])
                        else 0,
                        "confidence_score": results["confidences"][i]
                        if i < len(results["confidences"])
                        else 0,
                        "processing_time_ms": results["frame_times"][i] * 1000
                        if i < len(results["frame_times"])
                        else 0,
                    }
                    export_data["frame_details"].append(frame_data)

                if results.get("errors"):
                    export_data["errors"] = results["errors"]

                # Generate export based on format
                if export_format == "JSON":
                    # TODO: Modularize
                    import json
                    json_data = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"scene_graph_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

                elif export_format == "CSV":
                    # TODO: Modularize
                    summary_df = pd.DataFrame([export_data["video_metadata"]])
                    stats_df = pd.DataFrame([export_data["statistics"]])
                    frames_df = pd.DataFrame(export_data["frame_details"])
                    csv_buffer = []
                    csv_buffer.append("# Video Metadata")
                    csv_buffer.append(summary_df.to_csv(index=False))
                    csv_buffer.append("\n# Statistics Summary")
                    csv_buffer.append(stats_df.to_csv(index=False))
                    csv_buffer.append("\n# Frame-by-Frame Results")
                    csv_buffer.append(frames_df.to_csv(index=False))
                    if export_data.get("errors"):
                        errors_df = pd.DataFrame({"errors": export_data["errors"]})
                        csv_buffer.append("\n# Processing Errors")
                        csv_buffer.append(errors_df.to_csv(index=False))
                    csv_data = "\n".join(csv_buffer)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"scene_graph_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

                elif export_format == "XML":
                    # TODO: Modularize
                    import xml.etree.ElementTree as ET
                    root = ET.Element("scene_graph_results")
                    root.set("export_date", datetime.now().isoformat())
                    # Video metadata
                    metadata_elem = ET.SubElement(root, "video_metadata")
                    for key, value in export_data["video_metadata"].items():
                        elem = ET.SubElement(metadata_elem, key)
                        elem.text = str(value)
                    # Statistics
                    stats_elem = ET.SubElement(root, "statistics")
                    for key, value in export_data["statistics"].items():
                        elem = ET.SubElement(stats_elem, key)
                        elem.text = str(value)
                    # Frame details
                    frames_elem = ET.SubElement(root, "frame_details")
                    for frame in export_data["frame_details"]:
                        frame_elem = ET.SubElement(frames_elem, "frame")
                        for key, value in frame.items():
                            elem = ET.SubElement(frame_elem, key)
                            elem.text = str(value)
                    # Errors if any
                    if export_data.get("errors"):
                        errors_elem = ET.SubElement(root, "errors")
                        for error in export_data["errors"]:
                            error_elem = ET.SubElement(errors_elem, "error")
                            error_elem.text = str(error)
                    # Pretty format
                    import xml.dom.minidom
                    xml_data = ET.tostring(root, encoding="unicode", method="xml")
                    dom = xml.dom.minidom.parseString(xml_data)
                    pretty_xml = dom.toprettyxml(indent="  ")
                    st.download_button(
                        label="Download XML",
                        data=pretty_xml,
                        file_name=f"scene_graph_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml",
                        mime="application/xml",
                    )

                st.success(f"{export_format} export ready for download!")
            else:
                st.warning("No results to export")

        # Dark mode toggle at bottom of sidebar
        st.markdown("---")
        st.subheader("Theme")
        dark_mode = st.button("🌙 Toggle Dark Mode", key="dark_mode_toggle", help="Switch between light and dark themes")
        if dark_mode:
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    #--------------------------------
    # Video Analysis
    st.header("Video Analysis")
    if "uploaded_video_file" not in st.session_state:
        st.session_state.uploaded_video_file = None
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Please upload a video file for scene graph generation and NLP analysis",
    )
    if uploaded_file is not None:
        st.session_state.uploaded_video_file = uploaded_file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            cap = cv2.VideoCapture(tmp_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                st.session_state["video_total_frames"] = total_frames
                cap.release()
                
                # Display video info below upload component
                st.markdown("---")
                st.subheader("Video Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Frames", f"{total_frames:,}")
                with col2:
                    st.metric("FPS", f"{fps:.1f}")
                with col3:
                    st.metric("Duration", f"{duration:.1f}s")
            os.unlink(tmp_path)
        except Exception:
            pass

    #--------------------------------
    # Graph Processing Button
    if uploaded_file is not None and st.button("Generate Scene Graph", type="primary"):
        if not model_path:
            st.error(
                " No model checkpoint specified. Please select or provide a model path in the sidebar."
            )
        elif not os.path.exists(model_path):
            st.error(f" Model checkpoint not found at: `{model_path}`")

        else:
            import time
            start_time = time.time()
            progress_container = st.container()

            with progress_container:
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("### Processing Progress")
                    timer_display = st.empty()
                    timer_display.markdown(
                        "<div style='text-align: center; font-size: 24px; font-weight: bold; color: #1f77b4; margin: 10px 0;'>0.0s</div>",
                        unsafe_allow_html=True,
                    )
                progress_bar = st.progress(0)
                status_text = st.empty()
                st.markdown("#### Processing Log")
                log_display = st.empty()

            log_entries = []

            def update_progress(step, total_steps, message, log_message=None):
                """Update progress bar and log display with sub-ticks for smoother updates"""
                # TODO: Adapt test function call for this
                progress = step / total_steps
                
                # Add sub-ticks for smoother progress bar updates
                # Update progress bar with sub-tick increments
                for sub_tick in range(5):  # 5 sub-ticks per main step
                    sub_progress = progress + (sub_tick * 0.2 / total_steps)
                    if sub_progress <= 1.0:
                        progress_bar.progress(min(sub_progress, 1.0))
                        time.sleep(0.05)  # Small delay for visual effect
                
                # Final progress update
                progress_bar.progress(min(progress, 1.0))
                status_text.text(message)

                # Update timer
                current_time = time.time()
                elapsed = current_time - start_time
                timer_display.markdown(
                    f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: #1f77b4; margin: 10px 0;'>{elapsed:.1f}s</div>",
                    unsafe_allow_html=True,
                )

                if log_message:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    log_entries.append(f"[{timestamp}] {log_message}")
                    # Display log as simple text to avoid key conflicts
                    log_text = "\n".join(log_entries[-15:])  # Show last 15 entries
                    log_display.text(log_text)

            def update_timer():
                """Update the timer display"""
                current_time = time.time()
                elapsed = current_time - start_time
                timer_display.markdown(
                    f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: #1f77b4; margin: 10px 0;'>{elapsed:.1f}s</div>",
                    unsafe_allow_html=True,
                )
                return elapsed

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # Step 1: Process video with SGG
                update_progress(
                    1,
                    5,
                    "Processing video with scene graph generation...",
                    "Starting video processing with scene graph generation",
                )

                results = process_video_with_sgg(tmp_path, model_path, max_frames)

                if results:
                    update_progress(
                        2,
                        5,
                        "Video processing completed",
                        f"Video processed successfully! Analyzed {results['processed_frames']} frames",
                    )


                    # Step 2: Create bounding box video
                    update_progress(
                        3,
                        5,
                        "Creating video with bounding boxes...",
                        "Creating bounding box video...",
                    )

                    bbox_success, bbox_video_path = create_processed_video_with_bboxes(
                        tmp_path, model_path, max_frames
                    )

                    if bbox_success and bbox_video_path and os.path.exists(bbox_video_path):
                        print(f"Using original avc1 video: {bbox_video_path}")
                        file_size = os.path.getsize(bbox_video_path)
                        st.session_state["bbox_video_path"] = bbox_video_path
                        update_progress(
                            4,
                            5,
                            "Bounding box video created",
                            f"Bounding box video created successfully! Size: {file_size} bytes",
                        )
                        
                        # Debug: Store video path for display
                        st.session_state["debug_bbox_path"] = bbox_video_path

                        # Verify bbox video
                        try:
                            import cv2
                            test_cap = cv2.VideoCapture(bbox_video_path)
                            ret, test_frame = test_cap.read()
                            if ret:
                                log_entries.append(
                                    f"[{time.strftime('%H:%M:%S')}] Bounding box video file is readable by OpenCV"
                                )
                                log_text = "\n".join(log_entries[-15:])
                                log_display.text(log_text)
                            else:
                                log_entries.append(
                                    f"[{time.strftime('%H:%M:%S')}] ERROR: Bounding box video file created but not readable by OpenCV"
                                )
                                log_text = "\n".join(log_entries[-15:])
                                log_display.text(log_text)
                            test_cap.release()
                        except Exception as e:
                            log_entries.append(
                                f"[{time.strftime('%H:%M:%S')}] ERROR: Error verifying bbox video: {e}"
                            )
                            log_text = "\n".join(log_entries[-15:])
                            log_display.text(log_text)
                    else:
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] WARNING: Failed to create bounding box video"
                        )
                        log_text = "\n".join(log_entries[-15:])
                        log_display.text(log_text)

                    # Step 3: Create scene graph video
                    update_progress(
                        5,
                        5,
                        "Creating video with scene graph overlay...",
                        "Creating scene graph video...",
                    )
                    sg_success, scene_graph_video_path = create_processed_video_with_scene_graph(
                        tmp_path, model_path, max_frames
                    )
                    
                    # Debug: Check what happened
                    log_entries.append(
                        f"[{time.strftime('%H:%M:%S')}] Scene graph creation result: success={sg_success}, path_exists={os.path.exists(scene_graph_video_path) if scene_graph_video_path else False}"
                    )
                    if sg_success and scene_graph_video_path and os.path.exists(scene_graph_video_path):
                        file_size = os.path.getsize(scene_graph_video_path)
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] Scene graph file exists: {scene_graph_video_path}, size: {file_size} bytes"
                        )
                    else:
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] Scene graph file missing: {scene_graph_video_path}"
                        )
                    log_text = "\n".join(log_entries[-15:])
                    log_display.text(log_text)

                    if sg_success and os.path.exists(scene_graph_video_path):
                        file_size = os.path.getsize(scene_graph_video_path)
                        st.session_state["scene_graph_video_path"] = scene_graph_video_path
                        
                        # Debug: Store video path for display
                        st.session_state["debug_sg_path"] = scene_graph_video_path

                        # Final progress update
                        progress_bar.progress(1.0)
                        status_text.text("Processing completed successfully!")
                        final_elapsed = update_timer()

                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] Scene graph video created successfully! Size: {file_size} bytes"
                        )
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] All processing steps completed in {final_elapsed:.1f} seconds"
                        )
                        log_text = "\n".join(log_entries[-15:])
                        log_display.text(log_text)

                        # Verify scene graph video
                        try:
                            import cv2
                            test_cap = cv2.VideoCapture(scene_graph_video_path)
                            ret, test_frame = test_cap.read()
                            if ret:
                                log_entries.append(
                                    f"[{time.strftime('%H:%M:%S')}] Scene graph video file is readable by OpenCV"
                                )
                                log_text = "\n".join(log_entries[-15:])
                                log_display.text(log_text)
                            else:
                                log_entries.append(
                                    f"[{time.strftime('%H:%M:%S')}] ERROR: Scene graph video file created but not readable by OpenCV"
                                )
                                log_text = "\n".join(log_entries[-15:])
                                log_display.text(log_text)
                            test_cap.release()
                        except Exception as e:
                            log_entries.append(
                                f"[{time.strftime('%H:%M:%S')}] ERROR: Error verifying scene graph video: {e}"
                            )
                            log_text = "\n".join(log_entries[-15:])
                            log_display.text(log_text)
                    else:
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] WARNING: Failed to create scene graph video"
                        )
                        log_text = "\n".join(log_entries[-15:])
                        log_display.text(log_text)

                    # Show processing warnings if any
                    if results.get("errors"):
                        log_entries.append(
                            f"[{time.strftime('%H:%M:%S')}] WARNING: {len(results['errors'])} processing warnings occurred"
                        )
                        log_text = "\n".join(log_entries[-15:])
                        log_display.text(log_text)
                        with st.expander("Processing Warnings", expanded=False):
                            for error in results["errors"][:5]:  # Show first 5 errors
                                st.warning(error)

                    st.session_state["results"] = results
            finally:
                # Clean up temporary input file immediately after processing
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    log_entries.append(
                        f"[{time.strftime('%H:%M:%S')}] Cleaned up temporary input file"
                    )
                    log_text = "\n".join(log_entries[-15:])
                    log_display.text(log_text)

    #--------------------------------
    # Result View Tabs
    main_tab1, main_tab2 = st.tabs(["SGG View", "Advanced SGG View"])

    # SGG View Tab
    with main_tab1:

        st.header(" Video Players")
        if st.session_state.uploaded_video_file is not None:
            vid_col1, vid_col2, vid_col3 = st.columns(3)
            # Unprocessed Video
            with vid_col1:
                st.subheader("Original Video")
                st.video(st.session_state.uploaded_video_file)
            # Bounding Box Video
            with vid_col2:
                st.subheader("Object Detection")
                
                # Use the processed bbox video from session state (avc1 encoded)
                bbox_video_path = st.session_state.get("bbox_video_path") or st.session_state.get("debug_bbox_path")
                
                if bbox_video_path and os.path.exists(bbox_video_path):
                    bbox_size = os.path.getsize(bbox_video_path)
                    import cv2
                    cap = cv2.VideoCapture(bbox_video_path)
                    bbox_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
                    cap.release()
                    
                    # Validate and display video
                    if validate_video_file(bbox_video_path):
                        try:
                            # Try direct path first
                            st.video(bbox_video_path)
                        except Exception as e:
                            st.warning(f"Direct video display failed: {e}")
                            try:
                                # Fallback to bytes
                                with open(bbox_video_path, "rb") as video_file:
                                    video_bytes = video_file.read()
                                    st.video(video_bytes)
                            except Exception as e2:
                                st.error(f"Error loading bbox video: {e2}")
                                st.video(st.session_state.uploaded_video_file)
                    else:
                        st.error("Bounding box video file is corrupted or unreadable")
                        st.video(st.session_state.uploaded_video_file)
                else:
                    st.video(st.session_state.uploaded_video_file)
                    st.warning("No bounding box video available - showing original")
                    st.info(f"Processed bbox video not found. Available paths: {list(st.session_state.keys())}")

                # Add bbox table if we have detection results
                if "results" in st.session_state and "bbox_info" in st.session_state:
                    st.markdown("---")
                    st.subheader("Detected Objects")
                    bbox_info = st.session_state["bbox_info"]
                    if bbox_info:
                        # Create DataFrame for the table
                        bbox_df = pd.DataFrame(
                            [
                                {
                                    "Object": bbox["object_name"],
                                    "Confidence": f"{bbox['confidence']:.3f}",
                                    "BBox": f"[{bbox['bbox'][0]:.0f}, {bbox['bbox'][1]:.0f}, {bbox['bbox'][2]:.0f}, {bbox['bbox'][3]:.0f}]",
                                }
                                for bbox in bbox_info
                            ]
                        )
                        st.dataframe(bbox_df, width="stretch", hide_index=True)
                    else:
                        st.info("No objects detected above confidence threshold")
                else:
                    st.caption("Video with bounding box overlays")
            # Scene Graph Video
            with vid_col3:
                st.subheader("Scene Graph Analysis")
                
                # Use the processed scene graph video from session state (avc1 encoded)
                sg_video_path = st.session_state.get("scene_graph_video_path") or st.session_state.get("debug_sg_path")
                
                if sg_video_path and os.path.exists(sg_video_path):
                    # Debug information
                    sg_size = os.path.getsize(sg_video_path)
                    import cv2
                    cap = cv2.VideoCapture(sg_video_path)
                    sg_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
                    cap.release()
                    
                    # Validate and display video
                    if validate_video_file(sg_video_path):
                        try:
                            # Try direct path first
                            st.video(sg_video_path)
                        except Exception as e:
                            st.warning(f"Direct video display failed: {e}")
                            try:
                                # Fallback to bytes
                                with open(sg_video_path, "rb") as video_file:
                                    video_bytes = video_file.read()
                                    st.video(video_bytes)
                            except Exception as e2:
                                st.error(f"Error loading scene graph video: {e2}")
                                st.video(st.session_state.uploaded_video_file)
                                st.caption(" Scene graph overlay failed to load")
                    else:
                        st.error("Scene graph video file is corrupted or unreadable")
                        st.video(st.session_state.uploaded_video_file)
                else:
                    st.video(st.session_state.uploaded_video_file)
                    st.warning("No scene graph video available - showing original")
                    st.info(f"Processed scene graph video not found. Available paths: {list(st.session_state.keys())}")

                # Add relationship table if we have relationship results
                if "results" in st.session_state and "relationship_info" in st.session_state:
                    st.markdown("---")
                    st.subheader("Scene Graph Relationships")
                    relationship_info = st.session_state["relationship_info"]
                    if relationship_info:
                        # Create a temporary processor to get object/relationship names
                        temp_processor = None
                        if "model_path" in st.session_state:
                            try:
                                temp_processor = StreamlitVideoProcessor(st.session_state["model_path"])
                            except Exception:
                                pass
                        
                        relationship_data = []
                        for rel in relationship_info:
                            # Get subject and object names
                            subject_name = "person1"  # Default as requested
                            if "subject_class" in rel and temp_processor:
                                subject_name = temp_processor.get_object_name(rel["subject_class"])
                            
                            object_name = "object"
                            if "object_class" in rel and temp_processor:
                                object_name = temp_processor.get_object_name(rel["object_class"])
                            
                            # Get relationship name
                            relationship_name = "interacts_with"  # Default
                            if temp_processor:
                                if "attention_type" in rel and "attention_confidence" in rel:
                                    if rel["attention_confidence"] > 0.1:
                                        relationship_name = temp_processor.get_relationship_name(
                                            rel["attention_type"], "attention"
                                        )
                                elif "spatial_type" in rel and "spatial_confidence" in rel:
                                    if rel["spatial_confidence"] > 0.1:
                                        relationship_name = temp_processor.get_relationship_name(
                                            rel["spatial_type"], "spatial"
                                        )
                            
                            relationship_data.append({
                                "Subject": subject_name,
                                "Relation": relationship_name,
                                "Object": object_name,
                                "Confidence": f"{rel['confidence']:.3f}"
                            })
                        
                        if relationship_data:
                            relationship_df = pd.DataFrame(relationship_data)
                            st.dataframe(relationship_df, width="stretch", hide_index=True)
                        else:
                            st.info("No relationships detected above confidence threshold")
                    else:
                        st.info("No relationships detected above confidence threshold")
                else:
                    st.caption("Scene graph relationships will appear here")

            # Chat
            st.markdown("---")
            st.header("Chat Assistant")
            
            if CHAT_INTERFACE_AVAILABLE:
                # Use the new LLM-based chat interface
                if "chat_interface" not in st.session_state:
                    st.session_state.chat_interface = SceneGraphChatInterface(
                        model_name="google/gemma-3-270m",
                        model_type="gemma"
                    )
                
                # Set scene graph context if results are available
                if "results" in st.session_state:
                    st.session_state.chat_interface.set_scene_graph_context(
                        st.session_state["results"]
                    )
                
                # Render the chat interface
                st.session_state.chat_interface.render_chat_interface()
                
            else:
                # Fallback to simple chat interface
                st.warning("Advanced chat interface not available. Using basic chat.")
                if "chat_messages" not in st.session_state:
                    st.session_state.chat_messages = []
                    st.session_state.chat_intro_started = False
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                if not st.session_state.get("chat_intro_started", False):
                    st.session_state.chat_intro_started = True
                    intro_messages = [
                        """Hello there! Welcome to VidSgg... 
                        I'm your personal AI assistant for video scene graph analysis. 
                        I can help you discover hidden relationships and objects in your videos!
                        Just upload a video above to start.""",
                    ]
                    st.session_state.chat_messages = [
                        {"message": intro_messages[0], "is_user": False}
                    ]
                    intro_container = st.empty()
                    with intro_container.container():
                        message(
                            st.session_state.chat_messages[0]["message"],
                            is_user=False,
                            key=f"intro_0_{uuid.uuid4().hex[:8]}",
                            allow_html=True,
                        )
                    intro_container.empty()

                def handle_chat_input():
                    user_input = st.session_state.chat_input
                    if user_input.strip():
                        # Add user message
                        st.session_state.chat_messages.append(
                            {"message": user_input, "is_user": True}
                        )

                        # Generate bot response (placeholder logic)
                        bot_response = generate_bot_response(user_input)
                        st.session_state.chat_messages.append(
                            {"message": bot_response, "is_user": False}
                        )

                        # Clear input
                        st.session_state.chat_input = ""

                def generate_bot_response(user_input: str) -> str:
                    """Generate bot response based on user input"""
                    user_input_lower = user_input.lower()

                    if "hello" in user_input_lower or "hi" in user_input_lower:
                        return "Hello! I'm your VidSgg assistant. I can help you understand scene graph generation results and answer questions about the analysis."
                    elif "help" in user_input_lower:
                        return "I can help you with:\n• Understanding scene graph results\n• Explaining object detections\n• Interpreting relationship data\n• Model configuration questions\n• Export options"
                    elif "object" in user_input_lower and "results" in st.session_state:
                        results = st.session_state["results"]
                        avg_objects = (
                            np.mean(results["detections"]) if results["detections"] else 0
                        )
                        return f"In your video analysis, I detected an average of {avg_objects:.1f} objects per frame across {results['processed_frames']} processed frames."
                    elif (
                        "relationship" in user_input_lower and "results" in st.session_state
                    ):
                        results = st.session_state["results"]
                        avg_relationships = (
                            np.mean(results["relationships"])
                            if results["relationships"]
                            else 0
                        )
                        return f"The analysis found an average of {avg_relationships:.1f} relationships per frame in your video."
                    elif "confidence" in user_input_lower and "results" in st.session_state:
                        results = st.session_state["results"]
                        avg_confidence = (
                            np.mean(results["confidences"]) if results["confidences"] else 0
                        )
                        return f"The average confidence score across all detections was {avg_confidence:.2f}."
                    elif "model" in user_input_lower:
                        return "The VidSgg model uses STTran (Spatial-Temporal Transformer) for scene graph generation. It processes video frames to detect objects and their relationships over time."
                    elif "export" in user_input_lower:
                        return "You can export your results in JSON, CSV, or XML format using the export options in the sidebar. The exported data will include all detection and relationship information."
                    else:
                        return "I'm here to help with your scene graph analysis! Ask me about objects, relationships, confidence scores, or how the model works."

                chat_container = st.container() # Display chat messages
                with chat_container:
                    for i, msg in enumerate(st.session_state.chat_messages):
                        message(
                            msg["message"],
                            is_user=msg["is_user"],
                            key=f"chat_msg_{i}",
                            allow_html=True,
                        )
                st.text_input( # Chat input
                    "Ask me about your scene graph analysis:",
                    key="chat_input",
                    on_change=handle_chat_input,
                    placeholder="Type your question here...",
                )
                if st.button("Clear Chat"): # Clear chat button
                    st.session_state.chat_messages = []
                    st.rerun()

            # Sub-tabs for Frame view and Temporal view
            st.markdown("---")
            sgg_tab1, sgg_tab2 = st.tabs(["Temporal View", "NLP View"])

            with sgg_tab1:
                st.header("Temporal Scene Graph Analysis")

                # Results visualization if available
                if "results" in st.session_state:
                    results = st.session_state["results"]

                    if results["detections"]:
                        df = pd.DataFrame(
                            {
                                "Frame": range(len(results["detections"])),
                                "Objects_Detected": results["detections"],
                                "Relationships": results["relationships"],
                                "Confidence": results["confidences"],
                                "Time_Seconds": results["frame_times"],
                            }
                        )

                        # Vertical Object Node Timeline Visualization
                        st.subheader("Object Node Timeline")

                        # Create a vertical timeline showing object presence across frames
                        if results.get("frame_objects") and any(
                            results["frame_objects"]
                        ):
                            # Collect all unique objects across all frames
                            all_objects = set()
                            for frame_objects in results["frame_objects"]:
                                for obj in frame_objects:
                                    all_objects.add(obj["object_name"])

                            if all_objects:
                                all_objects = sorted(list(all_objects))

                                # Create vertical timeline visualization with connected spheres
                                fig_timeline = go.Figure()

                                # Define positions: person at top (y=0), objects below (y=1,2,3...)
                                person_y = 0
                                object_y_positions = list(
                                    range(1, len(all_objects) + 1)
                                )

                                # Add person node (always present) - RED
                                fig_timeline.add_trace(
                                    go.Scatter(
                                        x=[0, len(results["frame_objects"]) - 1],
                                        y=[person_y, person_y],
                                        mode="lines",
                                        name="Person",
                                        line=dict(width=8, color="red"),
                                        showlegend=True,
                                    )
                                )

                                # Add person spheres for each frame
                                for frame_idx in range(len(results["frame_objects"])):
                                    fig_timeline.add_trace(
                                        go.Scatter(
                                            x=[frame_idx],
                                            y=[person_y],
                                            mode="markers",
                                            marker=dict(
                                                size=20,
                                                color="red",
                                                line=dict(width=2, color="white"),
                                            ),
                                            name=f"Person (Frame {frame_idx})",
                                            hovertemplate="<b>Person</b><br>"
                                            + f"Frame: {frame_idx}<br>"
                                            + "<extra></extra>",
                                            showlegend=False,
                                        )
                                    )

                                # Add object nodes and their spheres with vertical connections
                                # Define distinct colors for better visibility
                                distinct_colors = [
                                    "rgb(0, 100, 200)",      # Dark blue
                                    "rgb(200, 50, 50)",      # Red
                                    "rgb(50, 150, 50)",      # Green
                                    "rgb(150, 50, 150)",     # Purple
                                    "rgb(200, 100, 0)",      # Orange
                                    "rgb(0, 150, 150)",      # Teal
                                    "rgb(150, 100, 50)",     # Brown
                                    "rgb(100, 0, 200)",      # Violet
                                    "rgb(200, 150, 0)",      # Gold
                                    "rgb(50, 100, 150)",     # Steel blue
                                    "rgb(150, 50, 100)",     # Magenta
                                ]
                                
                                for i, obj_name in enumerate(all_objects):
                                    obj_y = object_y_positions[i]

                                    # Use distinct colors from predefined palette
                                    color_index = i % len(distinct_colors)
                                    object_color = distinct_colors[color_index]

                                    # Add object horizontal line
                                    fig_timeline.add_trace(
                                        go.Scatter(
                                            x=[0, len(results["frame_objects"]) - 1],
                                            y=[obj_y, obj_y],
                                            mode="lines",
                                            name=obj_name,
                                            line=dict(width=6, color=object_color),
                                            showlegend=True,
                                        )
                                    )

                                    # Collect frames where this object appears
                                    object_frames = []
                                    object_confidences = []

                                    for frame_idx, frame_objects in enumerate(
                                        results["frame_objects"]
                                    ):
                                        # Check if object is present in this frame
                                        obj_in_frame = None
                                        for obj in frame_objects:
                                            if obj["object_name"] == obj_name:
                                                obj_in_frame = obj
                                                break

                                        if obj_in_frame:
                                            object_frames.append(frame_idx)
                                            object_confidences.append(
                                                obj_in_frame["confidence"]
                                            )

                                    # Add spheres for this object
                                    if object_frames:
                                        fig_timeline.add_trace(
                                            go.Scatter(
                                                x=object_frames,
                                                y=[obj_y] * len(object_frames),
                                                mode="markers",
                                                marker=dict(
                                                    size=20,
                                                    color=object_color,
                                                    line=dict(width=2, color="white"),
                                                ),
                                                name=f"{obj_name} spheres",
                                                hovertemplate=f"<b>{obj_name}</b><br>"
                                                + "Frame: %{x}<br>"
                                                + "Confidence: %{customdata:.3f}<br>"
                                                + "<extra></extra>",
                                                customdata=object_confidences,
                                                showlegend=False,
                                            )
                                        )

                                        # Add vertical connecting lines between spheres
                                        if len(object_frames) > 1:
                                            fig_timeline.add_trace(
                                                go.Scatter(
                                                    x=object_frames,
                                                    y=[obj_y] * len(object_frames),
                                                    mode="lines",
                                                    line=dict(
                                                        width=3,
                                                        color=object_color,
                                                        dash="solid",
                                                    ),
                                                    name=f"{obj_name} connections",
                                                    showlegend=False,
                                                    hoverinfo="skip",
                                                )
                                            )

                                    # Add edges from person to object spheres (vertical lines)
                                    for frame_idx in object_frames:
                                        fig_timeline.add_trace(
                                            go.Scatter(
                                                x=[frame_idx, frame_idx],
                                                y=[person_y, obj_y],
                                                mode="lines",
                                                line=dict(
                                                    width=2, color="gray", dash="dot"
                                                ),
                                                name=f"Edge {frame_idx}",
                                                hovertemplate=f"<b>Person → {obj_name}</b><br>"
                                                + f"Frame: {frame_idx}<br>"
                                                + "<extra></extra>",
                                                showlegend=False,
                                            )
                                        )

                                fig_timeline.update_layout(
                                    title="Object Nodes Across Frames with Person-Object Relationships",
                                    xaxis=dict(
                                        title="Frame Number",
                                        tickmode="linear",
                                        tick0=0,
                                        dtick=1,
                                        showgrid=True,
                                        gridcolor="lightgray",
                                    ),
                                    yaxis=dict(
                                        title="Nodes",
                                        tickmode="array",
                                        tickvals=[person_y] + object_y_positions,
                                        ticktext=["Person"] + all_objects,
                                        side="left",
                                    ),
                                    height=400 + (len(all_objects) + 1) * 30,
                                    showlegend=True,
                                    legend=dict(
                                        orientation="v",
                                        yanchor="top",
                                        y=1,
                                        xanchor="left",
                                        x=1.02,
                                    ),
                                )
                                st.plotly_chart(fig_timeline, width="stretch")

                                # Object statistics
                                st.subheader("Object Statistics")
                                obj_stats = []
                                for obj_name in all_objects:
                                    presence_count = sum(
                                        1
                                        for frame_objects in results["frame_objects"]
                                        if any(
                                            obj["object_name"] == obj_name
                                            for obj in frame_objects
                                        )
                                    )
                                    total_frames = len(results["frame_objects"])
                                    presence_percentage = (
                                        (presence_count / total_frames) * 100
                                        if total_frames > 0
                                        else 0
                                    )

                                    # Calculate average confidence for this object
                                    confidences = []
                                    for frame_objects in results["frame_objects"]:
                                        for obj in frame_objects:
                                            if obj["object_name"] == obj_name:
                                                confidences.append(obj["confidence"])
                                    avg_confidence = (
                                        np.mean(confidences) if confidences else 0.0
                                    )

                                    obj_stats.append(
                                        {
                                            "Object": obj_name,
                                            "Frames Present": presence_count,
                                            "Total Frames": total_frames,
                                            "Presence %": f"{presence_percentage:.1f}%",
                                            "Avg Confidence": f"{avg_confidence:.3f}",
                                        }
                                    )

                                stats_df = pd.DataFrame(obj_stats)
                                st.dataframe(
                                    stats_df, width="stretch", hide_index=True
                                )
                            else:
                                st.info("No objects detected in any frame")
                        else:
                            st.info(
                                "No detailed object information available for timeline visualization"
                            )

                        # Multi-line chart for detections and relationships
                        st.subheader("Scene Graph Metrics Over Time")
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=df["Time_Seconds"],
                                y=df["Objects_Detected"],
                                mode="lines+markers",
                                name="Objects Detected",
                                line=dict(color="blue"),
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=df["Time_Seconds"],
                                y=df["Relationships"],
                                mode="lines+markers",
                                name="Relationships",
                                line=dict(color="red"),
                                yaxis="y2",
                            )
                        )

                        fig.update_layout(
                            title="Scene Graph Analysis Over Time",
                            xaxis_title="Time (seconds)",
                            yaxis=dict(title="Objects Detected", side="left"),
                            yaxis2=dict(
                                title="Relationships", side="right", overlaying="y"
                            ),
                            legend=dict(x=0.02, y=0.98),
                        )
                        st.plotly_chart(fig, width="stretch")

                        # Confidence chart
                        fig3 = px.line(
                            df,
                            x="Time_Seconds",
                            y="Confidence",
                            title="Average Confidence Over Time",
                            labels={
                                "Time_Seconds": "Time (seconds)",
                                "Confidence": "Avg Confidence",
                            },
                        )
                        st.plotly_chart(fig3, width="stretch")

                        # Data table
                        st.subheader("Detection Details")
                        st.dataframe(df, width="stretch")
        
                # NLP View Tab Implementation
            
            with sgg_tab2:
                st.header(" NLP Analysis")

                # Single video player spanning full width
                if st.session_state.uploaded_video_file is not None:
                    st.subheader(" Video Analysis")
                    st.video(st.session_state.uploaded_video_file)

                    # NLP Analysis Results
                    st.markdown("---")
                    st.header("NLP Module Results")

                    # Create columns for different NLP outputs
                    nlp_col1, nlp_col2 = st.columns(2)

                    with nlp_col1:
                        st.subheader("Video Summarization")
                        summarization_text = """
                        **Automatically Generated Summary:**
                        
                        This video contains multiple scenes with various objects and activities. 
                        The analysis detected people interacting with objects in different spatial 
                        configurations. Key activities include movement patterns, object 
                        manipulations, and social interactions between detected entities.
                        
                        **Key Findings:**
                        • Multiple human subjects identified
                        • Various object interactions detected
                        • Temporal activity patterns observed
                        • Scene transitions and context changes noted
                        """
                        st.text_area("Summary", summarization_text, height=200, disabled=True)

                        st.subheader(" Video Semantic Search")
                        search_results = """
                        🔎 **Semantic Search Results:**
                        
                        **Query:** "People walking"
                        **Timestamps:** 0:12-0:18, 0:45-0:52, 1:23-1:30
                        
                        **Query:** "Object interaction"
                        **Timestamps:** 0:25-0:35, 1:05-1:15, 1:40-1:50
                        
                        **Query:** "Group activity"
                        **Timestamps:** 0:30-0:55, 1:10-1:35
                        """
                        st.text_area(
                            "Search Results", search_results, height=150, disabled=True
                        )

                    with nlp_col2:
                        st.subheader("Video Captioning")
                        captioning_text = """
                        **Frame-by-Frame Captions:**
                        
                        **00:05** - A person stands near a table with objects
                        **00:12** - Multiple people enter the scene from the left
                        **00:18** - Someone picks up an object from the surface
                        **00:25** - Two people engage in conversation
                        **00:32** - Group activity begins with shared focus
                        **00:40** - Objects are rearranged on the table
                        **00:48** - People move toward the background
                        **00:55** - Scene transitions to new activity
                        **01:02** - New objects appear in the frame
                        **01:10** - Final interactions before scene ends
                        """
                        st.text_area("Captions", captioning_text, height=200, disabled=True)

                        st.subheader("🔮 Action Anticipation")
                        anticipation_text = """
                        **Predicted Future Actions:**
                        
                        **Next 5 seconds:**
                        • Person likely to move towards door (85% confidence)
                        • Object manipulation probability: 72%
                        • Group dispersal expected: 68%
                        
                        **Next 10 seconds:**
                        • Scene change probability: 91%
                        • New person entry likelihood: 45%
                        • Activity continuation: 23%
                        
                        **Temporal Patterns:**
                        • Regular 15-second activity cycles detected
                        • Spatial movement patterns suggest routine behavior
                        """
                        st.text_area(
                            "Predictions", anticipation_text, height=150, disabled=True
                        )

                    # Additional NLP Features
                    st.markdown("---")
                    st.header("Advanced NLP Features")
                    feature_col1, feature_col2, feature_col3 = st.columns(3)

                    with feature_col1:
                        st.subheader("Emotion Analysis")
                        st.info("Detected emotions: Neutral (45%), Happy (30%), Focused (25%)")
                        emotion_data = pd.DataFrame(
                            {
                                "Emotion": ["Neutral", "Happy", "Focused", "Surprised"],
                                "Percentage": [45, 30, 25, 15],
                            }
                        )
                        fig_emotion = px.pie(
                            emotion_data,
                            values="Percentage",
                            names="Emotion",
                            title="Emotion Distribution",
                        )
                        st.plotly_chart(fig_emotion, width="stretch")

                    with feature_col2:
                        st.subheader("Scene Classification")
                        st.info("Scene type: Indoor Office Environment")
                        scene_data = pd.DataFrame(
                            {
                                "Scene Type": [
                                    "Office",
                                    "Meeting Room",
                                    "Kitchen",
                                    "Living Room",
                                ],
                                "Confidence": [0.89, 0.65, 0.23, 0.18],
                            }
                        )
                        fig_scene = px.bar(
                            scene_data,
                            x="Scene Type",
                            y="Confidence",
                            title="Scene Classification Confidence",
                        )
                        st.plotly_chart(fig_scene, width="stretch")

                    with feature_col3:
                        st.subheader("Activity Recognition")
                        st.info("Primary activity: Collaborative Work")
                        activity_data = pd.DataFrame(
                            {
                                "Activity": ["Meeting", "Discussion", "Presentation", "Break"],
                                "Duration": [45, 25, 20, 10],
                            }
                        )
                        fig_activity = px.bar(
                            activity_data,
                            x="Activity",
                            y="Duration",
                            title="Activity Duration (seconds)",
                        )
                        st.plotly_chart(fig_activity, width="stretch")

                else:
                    st.info("Please upload a video file above to see NLP analysis results.")
                    st.markdown("---")
                    st.subheader("Available NLP Features")
                    placeholder_features = [
                        "**Video Summarization** - Generate comprehensive summaries of video content",
                        "**Video Captioning** - Frame-by-frame natural language descriptions",
                        "**Semantic Search** - Find specific content using natural language queries",
                        "**Action Anticipation** - Predict future actions and activities",
                        "**Emotion Analysis** - Detect and analyze emotional states",
                        "**Scene Classification** - Identify and categorize different scene types",
                        "**Activity Recognition** - Recognize and track various activities",
                    ]
                    for feature in placeholder_features:
                        st.markdown(feature)

        else:
            st.info("Please upload a video file first to see the analysis results.")

    #--------------------------------
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def cleanup_temp_videos():
    """Clean up temporary video files from data/temp_vid/ directory on exit"""
    import os
    import shutil
    
    # Clean up the project temp directory (output videos)
    temp_dir = os.path.join("data", "temp_vid")
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary video directory: {temp_dir}")
        except Exception as e:
            print(f"Failed to clean up temp directory {temp_dir}: {e}")
    
    # Clean up any remaining session state video files
    cleanup_paths = []
    if "bbox_video_path" in st.session_state:
        cleanup_paths.append(st.session_state["bbox_video_path"])
    if "scene_graph_video_path" in st.session_state:
        cleanup_paths.append(st.session_state["scene_graph_video_path"])
    
    for path in cleanup_paths:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
                print(f"Cleaned up temporary file: {path}")
        except Exception as e:
            print(f"Failed to clean up {path}: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, cleaning up...")
        cleanup_temp_videos()
    except Exception as e:
        print(f"Error occurred: {e}")
        cleanup_temp_videos()
        raise
    finally:
        # This will run on normal exit, Ctrl+C, or any other exit
        cleanup_temp_videos()


    #--------------------------------
    # Video Display Debugging
    # # Debug Component - Simple Video Processing Test
    # st.markdown("### 🔧 Debug: Simple Video Processing Test")
    # with st.expander("Test Simple Video Processing with Generated Videos", expanded=True):
    #     st.markdown("**Testing simple video processing with pre-generated videos**")
    #     st.markdown("""
    #     **What this tests:**
    #     - **Basic Video Processing**: Just copies frames without any modifications
    #     - **Model Video Processing**: Runs model detection but saves original frames (no drawing)
    #     - **Video Display**: Tests if the processed videos display properly in Streamlit
        
    #     **Expected Results:**
    #     - Both videos should be identical to the original 0MK2C.mp4
    #     - Videos should display properly in all display methods
    #     - File sizes should be similar to original
    #     """)
        
    #     # Show current video paths being used
    #     st.markdown("**Current Video Paths:**")
    #     st.code("""
    #         Debug Component Videos:
    #         - Original: 0MK2C.mp4
    #         - Basic: simple_basic_full_0MK2C.mp4
    #         - Model: simple_model_full_0MK2C.mp4
    #         Main Display Videos:
    #         - Bbox Video: simple_model_full_0MK2C.mp4
    #         - Scene Graph Video: simple_basic_full_0MK2C.mp4
    #     """)
        
    #     # Use the generated simple videos (full length)
    #     original_video = "0MK2C.mp4"
    #     basic_video = "debug_bbox_c3aec751.mp4"
    #     model_video = "simple_model_full_0MK2C.mp4"
        

    #     # Check if videos exist
    #     original_exists = os.path.exists(original_video)
    #     basic_exists = os.path.exists(basic_video)
    #     model_exists = os.path.exists(model_video)
        
    #     # Show which videos are being used with detailed info
    #     if original_exists:
    #         original_size = os.path.getsize(original_video)
    #         # Get frame count
    #         import cv2
    #         cap = cv2.VideoCapture(original_video)
    #         original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
    #         cap.release()
    #         st.success(f"🎯 **Original Video**: `{original_video}` ({original_size:,} bytes, {original_frames} frames)")
    #     else:
    #         st.error(f"❌ **Original Video**: `{original_video}` not found")
            
    #     if basic_exists:
    #         basic_size = os.path.getsize(basic_video)
    #         # Get frame count
    #         cap = cv2.VideoCapture(basic_video)
    #         basic_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
    #         cap.release()
    #         st.success(f"🎯 **Basic Video**: `{basic_video}` ({basic_size:,} bytes, {basic_frames} frames) - Simple frame copy")
    #     else:
    #         st.error(f"❌ **Basic Video**: `{basic_video}` not found")
            
    #     if model_exists:
    #         model_size = os.path.getsize(model_video)
    #         # Get frame count
    #         cap = cv2.VideoCapture(model_video)
    #         model_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
    #         cap.release()
    #         st.success(f"🎯 **Model Video**: `{model_video}` ({model_size:,} bytes, {model_frames} frames) - Model processing without drawing")
    #     else:
    #         st.error(f"❌ **Model Video**: `{model_video}` not found")
        
    #     # Additional debug info
    #     st.markdown("---")
    #     st.markdown("**Debug Information:**")
    #     st.code(f"""
    #     Video Paths in Debug Component:
    #     - original_video = "{original_video}"
    #     - basic_video = "{basic_video}"  
    #     - model_video = "{model_video}"

    #     File Existence Check:
    #     - Original exists: {original_exists}
    #     - Basic exists: {basic_exists}
    #     - Model exists: {model_exists}
    #             """)
        
    #     # Add basic processing function
    #     st.markdown("---")
    #     st.markdown("**Generate New Basic Processing Video**")
        
    #     # Import the simple processing functions
    #     try:
    #         from simple_drawing_methods import simple_process_video_basic, simple_process_video_with_bboxes
            
    #         # Create buttons for different processing methods
    #         col_btn1, col_btn2 = st.columns(2)
            
    #         with col_btn1:
    #             # Button to generate new basic video
    #             if st.button("🔄 Generate Basic Video", help="Create a new basic video by copying frames from original"):
    #                 if original_exists:
    #                     # Generate new basic video with timestamp
    #                     import time
    #                     timestamp = int(time.time())
    #                     new_basic_video = f"debug_basic_{timestamp}.mp4"
                        
    #                     with st.spinner("Generating new basic video..."):
    #                         success = simple_process_video_basic(original_video, new_basic_video, max_frames=30)
                            
    #                         if success:
    #                             st.success(f"✅ New basic video generated: {new_basic_video}")
    #                             # Store in session state for central display
    #                             st.session_state["current_basic_video"] = new_basic_video
    #                             st.rerun()
    #                         else:
    #                             st.error("❌ Failed to generate new basic video")
    #                 else:
    #                     st.error("❌ Original video not found - cannot generate basic video")
            
    #         with col_btn2:
    #             # Button to generate bounding box video
    #             if st.button("📦 Generate Bbox Video", help="Create a new video with bounding boxes drawn"):
    #                 if original_exists:
    #                     # Generate new bbox video with timestamp
    #                     import time
    #                     timestamp = int(time.time())
    #                     new_bbox_video = f"debug_bbox_{timestamp}.mp4"
                        
    #                     with st.spinner("Generating bounding box video..."):
    #                         success = simple_process_video_with_bboxes(original_video, new_bbox_video, max_frames=30)
                            
    #                         if success:
    #                             st.success(f"✅ New bbox video generated: {new_bbox_video}")
    #                             # Store in session state for central display
    #                             st.session_state["current_bbox_video"] = new_bbox_video
    #                             st.rerun()
    #                         else:
    #                             st.error("❌ Failed to generate bounding box video")
    #                 else:
    #                     st.error("❌ Original video not found - cannot generate bbox video")
                        
    #     except ImportError as e:
    #         st.error(f"❌ Could not import processing functions: {e}")
        
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         st.markdown("**Original Video (0MK2C.mp4)**")
    #         if original_exists:
    #             st.success("✅ Original video found")
    #             st.write(f"File size: {original_size:,} bytes")
                
    #             # Method 1: Direct st.video
    #             st.markdown("**Method 1: Direct st.video**")
    #             try:
    #                 st.video(original_video)
    #                 st.success("✅ Original video works!")
    #             except Exception as e:
    #                 st.error(f"❌ Original video failed: {e}")
                
    #             # Method 2: st.video with bytes
    #             st.markdown("**Method 2: st.video with bytes**")
    #             try:
    #                 with open(original_video, "rb") as f:
    #                     video_bytes = f.read()
    #                 st.video(video_bytes)
    #                 st.success("✅ Original video with bytes works!")
    #             except Exception as e:
    #                 st.error(f"❌ Original video with bytes failed: {e}")
    #         else:
    #             st.error("❌ Original video not found")
        
    #     with col2:
    #         st.markdown("**Central Display - All Results**")
    #         st.caption("This column shows the most recently generated videos from the buttons above")
            
    #         # Display current basic video if available
    #         current_basic = st.session_state.get("current_basic_video")
    #         if current_basic and os.path.exists(current_basic):
    #             st.markdown("**🔄 Basic Video (Frame Copy)**")
    #             st.success(f"✅ Basic video: {current_basic}")
    #             basic_size = os.path.getsize(current_basic)
    #             cap = cv2.VideoCapture(current_basic)
    #             basic_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
    #             cap.release()
    #             st.write(f"File size: {basic_size:,} bytes, Frames: {basic_frames}")
                
    #             # Display the video
    #             try:
    #                 st.video(current_basic)
    #                 st.success("✅ Basic video displays correctly!")
    #             except Exception as e:
    #                 st.error(f"❌ Basic video display failed: {e}")
            
    #         # Display current bbox video if available
    #         current_bbox = st.session_state.get("current_bbox_video")
    #         if current_bbox and os.path.exists(current_bbox):
    #             st.markdown("**📦 Bounding Box Video**")
    #             st.success(f"✅ Bbox video: {current_bbox}")
    #             bbox_size = os.path.getsize(current_bbox)
    #             cap = cv2.VideoCapture(current_bbox)
    #             bbox_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else "unknown"
    #             cap.release()
    #             st.write(f"File size: {bbox_size:,} bytes, Frames: {bbox_frames}")
                
    #             # Display the video
    #             try:
    #                 st.video(current_bbox)
    #                 st.success("✅ Bbox video displays correctly!")
    #             except Exception as e:
    #                 st.error(f"❌ Bbox video display failed: {e}")
            
    #         # If no current videos, show default
    #         if not current_basic and not current_bbox:
    #             st.info("Click the buttons above to generate videos and see them here")
    #             if basic_exists:
    #                 st.markdown("**Default Basic Video**")
    #                 st.write(f"File size: {basic_size:,} bytes")
    #                 try:
    #                     st.video(basic_video)
    #                 except Exception as e:
    #                     st.error(f"❌ Video display failed: {e}")
                
    #             # Method 1: Direct st.video
    #             st.markdown("**Method 1: Direct st.video**")
    #             try:
    #                 st.video(basic_video)
    #                 st.success("✅ Direct st.video works!")
    #             except Exception as e:
    #                 st.error(f"❌ Direct st.video failed: {e}")
                
    #             # Method 2: st.video with bytes
    #             st.markdown("**Method 2: st.video with bytes**")
    #             try:
    #                 with open(basic_video, "rb") as f:
    #                     video_bytes = f.read()
    #                 st.video(video_bytes)
    #                 st.success("✅ st.video with bytes works!")
    #             except Exception as e:
    #                 st.error(f"❌ st.video with bytes failed: {e}")
                
    #             # Method 3: HTML5 video
    #             st.markdown("**Method 3: HTML5 video**")
    #             try:
    #                 import base64
    #                 video_html = f"""
    #                 <div style="text-align: center;">
    #                     <video width="100%" height="200" controls autoplay muted loop>
    #                         <source src="data:video/mp4;base64,{base64.b64encode(open(basic_video, 'rb').read()).decode()}" type="video/mp4">
    #                         Your browser does not support the video tag.
    #                     </video>
    #                 </div>
    #                 """
    #                 st.markdown(video_html, unsafe_allow_html=True)
    #                 st.success("✅ HTML5 video works!")
    #             except Exception as e:
    #                 st.error(f"❌ HTML5 video failed: {e}")
                
    #             # Download button
    #             with open(basic_video, "rb") as f:
    #                 st.download_button(
    #                     label="Download Basic Video",
    #                     data=f.read(),
    #                     file_name=basic_video,
    #                     mime="video/mp4"
    #                 )
    #         else:
    #             st.error(f"❌ Basic video not found: {basic_video}")
        
    #     with col3:
    #         st.markdown("**Model Processing Video**")
    #         if model_exists:
    #             st.success(f"✅ Model video exists: {model_video}")
    #             st.write(f"File size: {model_size:,} bytes")
                
    #             # Method 1: Direct st.video
    #             st.markdown("**Method 1: Direct st.video**")
    #             try:
    #                 st.video(model_video)
    #                 st.success("✅ Direct st.video works!")
    #             except Exception as e:
    #                 st.error(f"❌ Direct st.video failed: {e}")
                
    #             # Method 2: st.video with bytes
    #             st.markdown("**Method 2: st.video with bytes**")
    #             try:
    #                 with open(model_video, "rb") as f:
    #                     video_bytes = f.read()
    #                 st.video(video_bytes)
    #                 st.success("✅ st.video with bytes works!")
    #             except Exception as e:
    #                 st.error(f"❌ st.video with bytes failed: {e}")
                
    #             # Method 3: HTML5 video
    #             st.markdown("**Method 3: HTML5 video**")
    #             try:
    #                 import base64
    #                 video_html = f"""
    #                 <div style="text-align: center;">
    #                     <video width="100%" height="200" controls autoplay muted loop>
    #                         <source src="data:video/mp4;base64,{base64.b64encode(open(model_video, 'rb').read()).decode()}" type="video/mp4">
    #                         Your browser does not support the video tag.
    #                     </video>
    #                 </div>
    #                 """
    #                 st.markdown(video_html, unsafe_allow_html=True)
    #                 st.success("✅ HTML5 video works!")
    #             except Exception as e:
    #                 st.error(f"❌ HTML5 video failed: {e}")
                
    #             # Download button
    #             with open(model_video, "rb") as f:
    #                 st.download_button(
    #                     label="Download Model Video",
    #                     data=f.read(),
    #                     file_name=model_video,
    #                     mime="video/mp4"
    #                 )
    #         else:
    #             st.error(f"❌ Model video not found: {model_video}")
        
    #     # Method 4: streamlit-player (full width)
    #     st.markdown("**Method 4: streamlit-player**")
    #     if STREAMLIT_PLAYER_AVAILABLE:
    #         col4, col5 = st.columns(2)
    #         with col4:
    #             st.markdown("**Basic Video (streamlit-player)**")
    #             if basic_exists:
    #                 try:
    #                     file_url = f"file:///{os.path.abspath(basic_video).replace(os.sep, '/')}"
    #                     st_player(file_url)
    #                     st.success("✅ streamlit-player works!")
    #                 except Exception as e:
    #                     st.error(f"❌ streamlit-player failed: {e}")
    #         with col5:
    #             st.markdown("**Model Video (streamlit-player)**")
    #             if model_exists:
    #                 try:
    #                     file_url = f"file:///{os.path.abspath(model_video).replace(os.sep, '/')}"
    #                     st_player(file_url)
    #                     st.success("✅ streamlit-player works!")
    #                 except Exception as e:
    #                     st.error(f"❌ streamlit-player failed: {e}")
    #     else:
    #         st.info("ℹ️ streamlit-player not available - install with: pip install streamlit-player")

    # # Debug Component - Video Display
    # st.markdown("### 🔧 Debug: Video Files")
    # with st.expander("Debug Video Display", expanded=False):
    #     st.markdown("**Debugging video display issues - showing debug files from project root:**")
        
    #     # Look for debug video files in project root (including browser-friendly versions)
    #     debug_files = []
    #     for file in os.listdir("."):
    #         if file.startswith("debug_") and file.endswith(".mp4"):
    #             debug_files.append(file)
        
    #     # Prioritize browser-friendly versions
    #     browser_friendly_files = [f for f in debug_files if "browser_friendly" in f]
    #     if browser_friendly_files:
    #         debug_files = browser_friendly_files + [f for f in debug_files if "browser_friendly" not in f]
        
    #     if debug_files:
    #         st.info(f"Found {len(debug_files)} debug video files in project root")
            
    #         # Group files by type (prioritize browser-friendly versions)
    #         bbox_files = [f for f in debug_files if f.startswith("debug_bbox_")]
    #         sg_files = [f for f in debug_files if f.startswith("debug_scene_graph_")]
            
    #         # Sort to prioritize browser-friendly versions
    #         bbox_files.sort(key=lambda x: ("browser_friendly" not in x, x))
    #         sg_files.sort(key=lambda x: ("browser_friendly" not in x, x))
            
    #         debug_col1, debug_col2 = st.columns(2)
            
    #         with debug_col1:
    #             st.subheader("Debug: Bounding Box Videos")
    #             if bbox_files:
    #                 # Show the most recent bbox file
    #                 latest_bbox = max(bbox_files, key=lambda x: os.path.getctime(x))
    #                 st.info(f"Latest bbox video: {latest_bbox}")
                    
    #                 if os.path.exists(latest_bbox):
    #                     st.success("✅ Bbox video file exists!")
                        
    #                     # Use HTML video element for better control
    #                     video_html = f"""
    #                     <div style="text-align: center;">
    #                         <video width="100%" height="300" controls autoplay muted loop>
    #                             <source src="data:video/mp4;base64,{base64.b64encode(open(latest_bbox, 'rb').read()).decode()}" type="video/mp4">
    #                             Your browser does not support the video tag.
    #                         </video>
    #                     </div>
    #                     """
    #                     st.markdown(video_html, unsafe_allow_html=True)
                        
    #                     # Alternative: File download button
    #                     with open(latest_bbox, "rb") as f:
    #                         video_bytes = f.read()
    #                     st.download_button(
    #                         label="📥 Download Bbox Video",
    #                         data=video_bytes,
    #                         file_name=latest_bbox,
    #                         mime="video/mp4"
    #                     )
                        
    #                     # Video info
    #                     st.info(f"📁 File: {latest_bbox}")
    #                     st.info(f"📏 Size: {os.path.getsize(latest_bbox):,} bytes")
                        
    #                     # Alternative: Use st.video with bytes for comparison
    #                     st.markdown("**Alternative Display (st.video with bytes):**")
    #                     try:
    #                         st.video(video_bytes)
    #                         st.success("✅ st.video with bytes works!")
    #                     except Exception as e:
    #                         st.error(f"❌ st.video with bytes failed: {e}")
                        
    #                     # Third approach: Use streamlit-player if available
    #                     if STREAMLIT_PLAYER_AVAILABLE:
    #                         st.markdown("**Third Display (streamlit-player):**")
    #                         try:
    #                             # Convert to file:// URL for local file
    #                             file_url = f"file:///{os.path.abspath(latest_bbox).replace(os.sep, '/')}"
    #                             st_player(file_url)
    #                             st.success("✅ streamlit-player works!")
    #                         except Exception as e:
    #                             st.error(f"❌ streamlit-player failed: {e}")
    #                     else:
    #                         st.info("ℹ️ streamlit-player not available - install with: pip install streamlit-player")
    #                 else:
    #                     st.error("❌ Bbox video file does not exist!")
                    
    #                 # Show all bbox files
    #                 if len(bbox_files) > 1:
    #                     st.markdown("**All bbox files:**")
    #                     for file in sorted(bbox_files):
    #                         file_size = os.path.getsize(file) if os.path.exists(file) else 0
    #                         st.text(f"  • {file} ({file_size} bytes)")
    #             else:
    #                 st.info("No bbox video files found")
            
    #         with debug_col2:
    #             st.subheader("Debug: Scene Graph Videos")
    #             if sg_files:
    #                 # Show the most recent scene graph file
    #                 latest_sg = max(sg_files, key=lambda x: os.path.getctime(x))
    #                 st.info(f"Latest scene graph video: {latest_sg}")
                    
    #                 if os.path.exists(latest_sg):
    #                     st.success("✅ Scene graph video file exists!")
                        
    #                     # Use HTML video element for better control
    #                     video_html = f"""
    #                     <div style="text-align: center;">
    #                         <video width="100%" height="300" controls autoplay muted loop>
    #                             <source src="data:video/mp4;base64,{base64.b64encode(open(latest_sg, 'rb').read()).decode()}" type="video/mp4">
    #                             Your browser does not support the video tag.
    #                         </video>
    #                     </div>
    #                     """
    #                     st.markdown(video_html, unsafe_allow_html=True)
                        
    #                     # Alternative: File download button
    #                     with open(latest_sg, "rb") as f:
    #                         video_bytes = f.read()
    #                     st.download_button(
    #                         label="📥 Download Scene Graph Video",
    #                         data=video_bytes,
    #                         file_name=latest_sg,
    #                         mime="video/mp4"
    #                     )
                        
    #                     # Video info
    #                     st.info(f"📁 File: {latest_sg}")
    #                     st.info(f"📏 Size: {os.path.getsize(latest_sg):,} bytes")
                        
    #                     # Alternative: Use st.video with bytes for comparison
    #                     st.markdown("**Alternative Display (st.video with bytes):**")
    #                     try:
    #                         st.video(video_bytes)
    #                         st.success("✅ st.video with bytes works!")
    #                     except Exception as e:
    #                         st.error(f"❌ st.video with bytes failed: {e}")
                        
    #                     # Third approach: Use streamlit-player if available
    #                     if STREAMLIT_PLAYER_AVAILABLE:
    #                         st.markdown("**Third Display (streamlit-player):**")
    #                         try:
    #                             # Convert to file:// URL for local file
    #                             file_url = f"file:///{os.path.abspath(latest_sg).replace(os.sep, '/')}"
    #                             st_player(file_url)
    #                             st.success("✅ streamlit-player works!")
    #                         except Exception as e:
    #                             st.error(f"❌ streamlit-player failed: {e}")
    #                     else:
    #                         st.info("ℹ️ streamlit-player not available - install with: pip install streamlit-player")
    #                 else:
    #                     st.error("❌ Scene graph video file does not exist!")
                    
    #                 # Show all scene graph files
    #                 if len(sg_files) > 1:
    #                     st.markdown("**All scene graph files:**")
    #                     for file in sorted(sg_files):
    #                         file_size = os.path.getsize(file) if os.path.exists(file) else 0
    #                         st.text(f"  • {file} ({file_size} bytes)")
    #             else:
    #                 st.info("No scene graph video files found")
    #     else:
    #         st.info("No debug video files found in project root")
    #         st.markdown("**Expected files:** `debug_bbox_*.mp4` and `debug_scene_graph_*.mp4`")
        
    #     # Installation instructions for streamlit-player
    #     if not STREAMLIT_PLAYER_AVAILABLE:
    #         st.markdown("---")
    #         st.subheader("🔧 Enhanced Video Display")
    #         st.info("For better video debugging, install streamlit-player:")
    #         st.code("pip install streamlit-player", language="bash")
    #         st.markdown("This will enable additional video display methods in the debug component.")
        
    #     # Additional debug info
    #     st.markdown("---")
    #     st.subheader("Debug: File System Info")
        
    #     # Show temp file info
    #     if "debug_temp_path" in st.session_state:
    #         temp_path = st.session_state["debug_temp_path"]
    #         st.markdown("**Original Temp File:**")
    #         st.text(f"Path: {temp_path}")
    #         st.text(f"Exists: {os.path.exists(temp_path)}")
    #         if os.path.exists(temp_path):
    #             st.text(f"Size: {os.path.getsize(temp_path)} bytes")
    #             st.text(f"Readable: {os.access(temp_path, os.R_OK)}")
        
    #     if "bbox_video_path" in st.session_state and "scene_graph_video_path" in st.session_state:
    #         bbox_path = st.session_state["bbox_video_path"]
    #         sg_path = st.session_state["scene_graph_video_path"]
            
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             st.markdown("**Bbox Video:**")
    #             st.text(f"Path: {bbox_path}")
    #             st.text(f"Exists: {os.path.exists(bbox_path)}")
    #             if os.path.exists(bbox_path):
    #                 st.text(f"Size: {os.path.getsize(bbox_path)} bytes")
    #                 st.text(f"Readable: {os.access(bbox_path, os.R_OK)}")
            
    #         with col2:
    #             st.markdown("**Scene Graph Video:**")
    #             st.text(f"Path: {sg_path}")
    #             st.text(f"Exists: {os.path.exists(sg_path)}")
    #             if os.path.exists(sg_path):
    #                 st.text(f"Size: {os.path.getsize(sg_path)} bytes")
    #                 st.text(f"Readable: {os.access(sg_path, os.R_OK)}")
        
    #     # Add cleanup button for debugging
    #     st.markdown("---")
    #     st.subheader("Debug: Cleanup")
    #     if st.button("🗑️ Clean Up Debug Files", help="Remove all debug files and reset session state"):
    #         cleanup_paths = []
            
    #         # Add temp files from session state
    #         if "debug_temp_path" in st.session_state:
    #             cleanup_paths.append(st.session_state["debug_temp_path"])
    #         if "bbox_video_path" in st.session_state:
    #             cleanup_paths.append(st.session_state["bbox_video_path"])
    #         if "scene_graph_video_path" in st.session_state:
    #             cleanup_paths.append(st.session_state["scene_graph_video_path"])
            
    #         # Add debug files from project root
    #         for file in os.listdir("."):
    #             if file.startswith("debug_") and file.endswith(".mp4"):
    #                 cleanup_paths.append(file)
            
    #         cleaned_count = 0
    #         for path in cleanup_paths:
    #             try:
    #                 if os.path.exists(path):
    #                     os.unlink(path)
    #                     cleaned_count += 1
    #                     st.info(f"Deleted: {path}")
    #             except Exception as e:
    #                 st.error(f"Failed to delete {path}: {e}")
            
    #         # Clear session state
    #         keys_to_clear = ["debug_temp_path", "bbox_video_path", "scene_graph_video_path", "results"]
    #         for key in keys_to_clear:
    #             if key in st.session_state:
    #                 del st.session_state[key]
            
    #         st.success(f"Cleaned up {cleaned_count} debug files and reset session state")
    #         st.rerun()
    # st.markdown("---")