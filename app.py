import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cv2
import torch
import os
import sys
from pathlib import Path
import tempfile
import time
from typing import Dict, Any, List, Tuple
from streamlit_chat import message
import uuid

"""
TODO:
- add sphinx doc string for each function
- modularize in ~./app directory
"""

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts", "core"))

try:
    from lib.track import get_sequence
except ImportError:
    print("Warning: Could not import get_sequence")

st.set_page_config(
    page_title="VidSgg - Scene Graph Generation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.setup_models()

    def setup_models(self):
        """Initialize models for video processing"""
        try:
            from lib.config import Config
            from lib.object_detector import detector
            from lib.sttran import STTran
            from lib.matcher import HungarianMatcher
            from dataloader.action_genome import AG

            self.conf = Config()
            self.conf.mode = "sgdet"  # Use sgdet mode for custom videos
            self.conf.data_path = "data/action_genome"

            # Dataset
            self.AG_dataset = AG(
                mode="test",
                datasize=self.conf.datasize,
                data_path=self.conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=True,
            )

            # Object Detector
            self.object_detector = detector(
                train=False,
                object_classes=self.AG_dataset.object_classes,
                use_SUPPLY=True,
                mode=self.conf.mode,
            ).to(device=self.device)
            self.object_detector.eval()

            # SGG
            self.model = STTran(
                mode=self.conf.mode,
                attention_class_num=len(self.AG_dataset.attention_relationships),
                spatial_class_num=len(self.AG_dataset.spatial_relationships),
                contact_class_num=len(self.AG_dataset.contacting_relationships),
                obj_classes=self.AG_dataset.object_classes,
                enc_layer_num=self.conf.enc_layer,
                dec_layer_num=self.conf.dec_layer,
            ).to(device=self.device)

            # Checkpoint
            if os.path.exists(self.model_path):
                ckpt = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(ckpt["state_dict"], strict=False)
                print(f"Loaded model from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model file {self.model_path} not found!")

            self.model.eval()
            self.matcher = HungarianMatcher(0.5, 1, 1, 0.5).to(device=self.device)
            self.matcher.eval()

        except Exception as e:
            st.error(f"Failed to setup models: {e}")
            raise

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
            # Preprocess frame
            im_data = self.preprocess_frame(frame)
            im_info = torch.tensor([[600, 600, 1.0]], dtype=torch.float32).to(
                self.device
            )
            gt_boxes = torch.zeros([1, 1, 5]).to(self.device)
            num_boxes = torch.zeros([1], dtype=torch.int64).to(self.device)

            with torch.no_grad():
                # Object detection
                empty_annotation = []
                entry = self.object_detector(
                    im_data, im_info, gt_boxes, num_boxes, empty_annotation, im_all=None
                )

                # Debug: Check raw detection outputs
                if "boxes" in entry and entry["boxes"] is not None:
                    print(f"Raw detections: {entry['boxes'].shape[0]} boxes")
                    if "pred_scores" in entry:
                        raw_scores = entry["pred_scores"].cpu().numpy()
                        print(
                            f"Raw scores range: {raw_scores.min():.3f} - {raw_scores.max():.3f}"
                        )
                        print(f"Scores above 0.1: {(raw_scores > 0.1).sum()}")
                        print(f"Scores above 0.3: {(raw_scores > 0.3).sum()}")

                # Process sequence for sgdet mode
                if self.conf.mode == "sgdet":
                    # Store original format
                    original_scores = (
                        entry["scores"].clone() if "scores" in entry else None
                    )
                    original_boxes = entry["boxes"].clone()

                    # Prepare entry for get_sequence (like test_bbox_prediction.py does)
                    if "distribution" in entry:
                        entry["scores"] = entry["distribution"]  # Use 2D distribution

                    # Fix boxes format: remove batch index column for get_sequence
                    if len(entry["boxes"].shape) > 1 and entry["boxes"].shape[1] > 4:
                        entry["boxes"] = entry["boxes"][
                            :, 1:
                        ]  # Remove first column (batch index)

                    # Get sequence information (like test_bbox_prediction.py does)
                    get_sequence(
                        entry,
                        [],  # empty annotation for sgdet mode
                        self.matcher,  # CRITICAL: This was missing!
                        torch.tensor([600, 600]).to(self.device),  # Image dimensions
                        "sgdet",
                    )

                    # Restore original format after get_sequence
                    if original_scores is not None:
                        entry["scores"] = original_scores
                    entry["boxes"] = original_boxes

                    print(f"After get_sequence: boxes shape = {entry['boxes'].shape}")

                # Scene graph generation
                pred = self.model(entry)

            return self.extract_results(entry, pred), entry, pred

        except Exception as e:
            print(f"Frame processing error: {e}")
            return {"objects": 0, "relationships": 0, "error": str(e)}, None, None

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
        print(f"DRAW_BBOX: Entry fields: {list(entry.keys())}")
        print(
            f"DRAW_BBOX: Boxes shape: {boxes.shape if hasattr(boxes, 'shape') else len(boxes)}"
        )
        if labels is not None:
            print(
                f"DRAW_BBOX: Labels shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}"
            )
            # Add detailed object list logging
            if hasattr(self, "AG_dataset") and labels is not None:
                print("DRAW_BBOX: === DETECTED OBJECTS LIST ===")
                for i, label_idx in enumerate(labels):
                    if label_idx < len(self.AG_dataset.object_classes):
                        object_name = self.AG_dataset.object_classes[label_idx]
                        score_str = (
                            f" (score: {scores[i]:.3f})"
                            if scores is not None and i < len(scores)
                            else ""
                        )
                        print(f"  {i+1}. {object_name}{score_str}")
                    else:
                        print(f"  {i+1}. unknown_class_{label_idx}")
                print("DRAW_BBOX: === END OBJECT LIST ===")
        if scores is not None:
            print(
                f"DRAW_BBOX: Scores shape: {scores.shape if hasattr(scores, 'shape') else len(scores)}"
            )
            print(f"DRAW_BBOX: All scores: {scores}")
            print(
                f"DRAW_BBOX: Score statistics - Min: {scores.min():.3f}, Max: {scores.max():.3f}, Mean: {scores.mean():.3f}"
            )

        # Check if we have distribution instead
        if "distribution" in entry:
            distribution = entry["distribution"]
            if isinstance(distribution, torch.Tensor):
                print(f"Distribution shape: {distribution.shape}")
                # Use distribution to get scores
                if scores is None:
                    scores = torch.max(distribution, dim=1)[0].cpu().numpy()
                    print(f"Generated scores from distribution: {scores}")

        # If we still don't have scores, create dummy ones
        if scores is None and boxes is not None:
            scores = np.ones(len(boxes))
            print(f"Using dummy scores for {len(boxes)} boxes")

        # If we don't have labels, create dummy ones
        if labels is None and boxes is not None:
            labels = np.ones(len(boxes), dtype=int)
            print(f"Using dummy labels for {len(boxes)} boxes")

        # Debug: Show all detections before filtering
        if scores is not None and len(scores) > 0:
            print(f"Total detections before filtering: {len(scores)}")

            # Filter detections with lower confidence threshold
            high_conf_mask = scores > confidence_threshold
            print(
                f"Detections after confidence filtering (>{confidence_threshold}): {high_conf_mask.sum()}"
            )

            if boxes is not None:
                boxes = boxes[high_conf_mask]
            if labels is not None:
                labels = labels[high_conf_mask]
            scores = scores[high_conf_mask]

            print(f"Final boxes to draw: {len(boxes) if boxes is not None else 0}")

        # Draw boxes
        print(
            f"DRAW_BBOX: About to draw {len(boxes) if boxes is not None else 0} boxes"
        )
        for i, box in enumerate(boxes):
            print(f"DRAW_BBOX: Processing box {i+1}/{len(boxes)}: {box}")
            if len(box) >= 4:
                # Handle batch dimension if present
                if len(box) == 5:
                    x1, y1, x2, y2 = box[1:5].astype(int)
                    print(
                        f"DRAW_BBOX: Box {i+1} (with batch): ({x1}, {y1}, {x2}, {y2})"
                    )
                else:
                    x1, y1, x2, y2 = box[:4].astype(int)
                    print(f"DRAW_BBOX: Box {i+1} (no batch): ({x1}, {y1}, {x2}, {y2})")

                # Scale to frame size (assuming model uses 600x600)
                h, w = frame.shape[:2]
                x1 = int(x1 * w / 600)
                y1 = int(y1 * h / 600)
                x2 = int(x2 * w / 600)
                y2 = int(y2 * h / 600)

                print(
                    f"DRAW_BBOX: Box {i+1} scaled to frame ({w}x{h}): ({x1}, {y1}, {x2}, {y2})"
                )

                # Ensure coordinates are valid
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                print(f"DRAW_BBOX: Box {i+1} after clipping: ({x1}, {y1}, {x2}, {y2})")

                if x2 > x1 and y2 > y1:
                    print(f"DRAW_BBOX:  Drawing box {i+1} - valid coordinates")
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
                        if hasattr(self, "AG_dataset") and labels[i] < len(
                            self.AG_dataset.object_classes
                        ):
                            label_text = f"{self.AG_dataset.object_classes[labels[i]]}"
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

                    print(f"DRAW_BBOX:  Successfully drew box {i+1}")
                else:
                    print(
                        f"DRAW_BBOX:  Skipped box {i+1} - invalid coordinates (x2 <= x1 or y2 <= y1)"
                    )
            else:
                print(
                    f"DRAW_BBOX:  Skipped box {i+1} - insufficient coordinates (need at least 4)"
                )

        print(
            f"DRAW_BBOX: === Finished drawing {len(boxes) if boxes is not None else 0} boxes ==="
        )
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
            print(f"Result extraction error: {e}")
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

                # Create simple relationship label
                rel_label = f"rel_{rel.get('attention_type', 0)}"
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
            if hasattr(self, "AG_dataset") and labels is not None and i < len(labels):
                if labels[i] < len(self.AG_dataset.object_classes):
                    class_name = self.AG_dataset.object_classes[labels[i]]
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


def create_processed_video_with_bboxes(
    video_path: str, model_path: str, output_path: str, max_frames: int = 30
) -> bool:
    """Create a new video file with bounding boxes drawn on frames"""
    try:
        processor = StreamlitVideoProcessor(model_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open input video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"H264")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            st.warning("H264 codec not available, trying alternative...")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            st.warning("XVID codec not available, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise ValueError("Could not initialize video writer with any codec")

        frame_count = 0
        progress_placeholder = st.empty()
        st.info(
            f"Creating video with codec {fourcc} at {fps} FPS, size {width}x{height}"
        )

        while frame_count < max_frames and cap.isOpened():
            if not (ret := cap.read())[0]: 
                break
            frame = ret[1]

            progress_placeholder.text(
                f"Processing frame {frame_count + 1}/{max_frames} for bbox video"
            )

            # Process frame with SGG
            frame_results, entry, pred = processor.process_frame(frame)

            # TODO: remove the st.write's later
            if entry is not None:
                frame_with_boxes = processor.draw_bounding_boxes(frame, entry)
                if "boxes" in entry:
                    num_boxes = len(entry["boxes"]) if entry["boxes"] is not None else 0
                    if frame_count == 0:  # Log for first frame only
                        st.write(f"Frame {frame_count}: Detected {num_boxes} boxes")
            else:
                frame_with_boxes = frame
                if frame_count == 0:  # Log for first frame only
                    st.write(f"Frame {frame_count}: No detection entry")

            out.write(frame_with_boxes)
            frame_count += 1

        cap.release()
        out.release()
        progress_placeholder.empty()

        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)

            if file_size > 1000:
                st.success(f"Video written successfully: {file_size} bytes")
                return True
            else:
                st.error(f"Video file too small: {file_size} bytes")
                return False
        else:
            st.error("Output video file was not created")
            return False
        
    except Exception as e:
        st.error(f"Error creating processed video: {e}")
        return False


def create_processed_video_with_scene_graph(
    video_path: str, model_path: str, output_path: str, max_frames: int = 30
) -> bool:
    """Create a new video file with scene graph overlay drawn on frames

    :param video_path: Path to input video file
    :type video_path: str
    :param model_path: Path to model checkpoint
    :type model_path: str
    :param output_path: Path for output video file
    :type output_path: str
    :param max_frames: Maximum number of frames to process, defaults to 30
    :type max_frames: int, optional
    :return: True if video creation was successful, False otherwise
    :rtype: bool
    """
    try:
        # Initialize video processor
        processor = StreamlitVideoProcessor(model_path)

        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open input video file")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer with better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*"H264")  # Better browser compatibility
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Check if video writer was opened successfully
        if not out.isOpened():
            st.warning("H264 codec not available, trying alternative...")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            st.warning("XVID codec not available, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise ValueError("Could not initialize video writer with any codec")

        frame_count = 0
        progress_placeholder = st.empty()

        st.info(
            f"Creating scene graph video with codec {fourcc} at {fps} FPS, size {width}x{height}"
        )

        while frame_count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            progress_placeholder.text(
                f"Processing frame {frame_count + 1}/{max_frames} for scene graph video"
            )

            # Process frame with SGG
            frame_results, entry, pred = processor.process_frame(frame)

            # Create scene graph overlay if detection was successful
            if entry is not None and pred is not None:
                frame_with_sg = processor.create_scene_graph_frame(frame, entry, pred)
                # Debug: Check if any relationships were detected
                relationships = processor.extract_relationships(entry, pred)
                if frame_count == 0:  # Log for first frame only
                    st.write(
                        f"Frame {frame_count}: {len(relationships)} relationships detected"
                    )
            else:
                frame_with_sg = frame
                if frame_count == 0:  # Log for first frame only
                    st.write(f"Frame {frame_count}: No scene graph data available")

            # Write frame to output video
            out.write(frame_with_sg)
            frame_count += 1

        # Release resources
        cap.release()
        out.release()
        progress_placeholder.empty()

        # Verify the output file was created successfully
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 1000:  # At least 1KB
                st.success(f"Scene graph video written successfully: {file_size} bytes")
                return True
            else:
                st.error(f"Scene graph video file too small: {file_size} bytes")
                return False
        else:
            st.error("Output scene graph video file was not created")
            return False

    except Exception as e:
        st.error(f"Error creating scene graph video: {e}")
        return False


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
        checkpoints.update({
            f"{dataset_dir.name}/{model_dir.name}": str(run_dir / "model_best.tar")
            for dataset_dir in output_dir.iterdir() if dataset_dir.is_dir()
            for model_dir in dataset_dir.iterdir() if model_dir.is_dir()  
            for run_dir in model_dir.iterdir() if run_dir.is_dir()
            if (run_dir / "model_best.tar").exists()
        })

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
        }

        frame_count = 0
        progress_bar = st.progress(0)
        status_text = st.empty()

        while frame_count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            status_text.text(
                f"Processing frame {frame_count + 1}/{min(max_frames, total_frames)}"
            )

            # Process frame with SGG
            frame_results, entry, pred = processor.process_frame(frame)

            results["detections"].append(frame_results.get("objects", 0))
            results["relationships"].append(frame_results.get("relationships", 0))
            results["confidences"].append(frame_results.get("confidence", 0.0))
            results["frame_times"].append(frame_count / fps)

            if "error" in frame_results:
                results["errors"].append(
                    f"Frame {frame_count}: {frame_results['error']}"
                )

            frame_count += 1
            progress_bar.progress(frame_count / min(max_frames, total_frames))

        cap.release()
        results["processed_frames"] = frame_count

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        return results

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None


def main():
    # Header
    st.markdown(
        '<h1 class="main-header"> VidSgg - Scene Graph Generation</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("Video Scene Graph Generation with Deep Learning Models")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Model Configuration")
        checkpoints = find_available_checkpoints()

        if checkpoints:
            selected_model = st.selectbox(
                "Select Model Checkpoint",
                list(checkpoints.keys()),
                help="Available trained models",
            )
            model_path = checkpoints[selected_model]
            if "default" in selected_model.lower():
                st.success(" Default checkpoint loaded")
        else:
            st.warning("No trained models found in expected locations")
            st.info(
                "For development: Place model at `data/checkpoints/action_genome/sgdet_test/model_best.tar`"
            )
            model_path = st.text_input(
                "Model Path",
                value="data/checkpoints/action_genome/sgdet_test/model_best.tar",
                placeholder="Path to model checkpoint (.tar or .pth)",
                help="Provide path to a trained model checkpoint",
            )

        st.markdown("---")
        st.header("Processing Settings")

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

        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.5,
            help="Minimum confidence for object detection"
        )

        # TODO: Use these parameters in processing
        # processing_mode = st.radio(
        #     "Processing Mode",
        #     ["Scene Graph Detection (SGDET)", "Scene Graph Classification (SGCLS)", "Predicate Classification (PREDCLS)"],
        #     help="SGG processing mode"
        # )

        if st.button("🔄 Reset Settings"):
            st.rerun()

        # Statistics section moved from col2
        st.markdown("---")
        st.header("Statistics")

        # Metrics
        if "results" in st.session_state:
            results = st.session_state["results"]

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Frames", results["total_frames"])
            with col_b:
                st.metric("Processed", results["processed_frames"])

            col_c, col_d = st.columns(2)
            with col_c:
                avg_objects = (
                    np.mean(results["detections"]) if results["detections"] else 0
                )
                st.metric("Avg Objects", f"{avg_objects:.1f}")
            with col_d:
                avg_relationships = (
                    np.mean(results["relationships"]) if results["relationships"] else 0
                )
                st.metric("Avg Relations", f"{avg_relationships:.1f}")

            # Additional metrics
            col_e, col_f = st.columns(2)
            with col_e:
                avg_confidence = (
                    np.mean(results["confidences"]) if results["confidences"] else 0
                )
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            with col_f:
                error_rate = (
                    len(results.get("errors", [])) / results["processed_frames"] * 100
                    if results["processed_frames"] > 0
                    else 0
                )
                st.metric("Error Rate", f"{error_rate:.1f}%")

            # Video info
            st.markdown("---")
            st.header("Video Info")
            st.write(f"**FPS:** {results['fps']:.1f}")
            st.write(f"**Duration:** {results['total_frames'] / results['fps']:.1f}s")
        else:
            # Placeholder metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Frames", "-")
            with col_b:
                st.metric("Processed", "-")

            col_c, col_d = st.columns(2)
            with col_c:
                st.metric("Avg Objects", "-")
            with col_d:
                st.metric("Avg Relations", "-")

        # Model info moved from col2
        st.markdown("---")
        st.header("Model Info")

        if model_path and os.path.exists(model_path):
            st.success(" Model loaded")
            st.write(f"**Path:** {os.path.basename(model_path)}")
        else:
            st.warning(" No model loaded")

        # Export options moved from col2
        st.markdown("---")
        st.header("Export")

        # TODO: Implement export functionality
        # export_format = st.selectbox(
        #     "Export Format",
        #     ["JSON", "CSV", "XML"]
        # )

        if st.button("Download Results"):
            if "results" in st.session_state:
                st.info("Export functionality will be available soon")
                # TODO: Add export functionality
            else:
                st.warning("No results to export")

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
                st.session_state["video_total_frames"] = total_frames
                cap.release()
            os.unlink(tmp_path)
        except Exception:
            pass

    # Graph Processing Button
    if uploaded_file is not None and st.button("Generate Scene Graph", type="primary"):

        # TODO: Add a progress bar
        # TODO: Time the main button

        if not model_path:
            st.error(" No model checkpoint specified. Please select or provide a model path in the sidebar.")
        elif not os.path.exists(model_path):
            st.error(f" Model checkpoint not found at: `{model_path}`")
            st.info("Make sure the checkpoint file exists or select a different model from the sidebar.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner("Processing video..."):
                    results = process_video_with_sgg(tmp_path, model_path, max_frames)

                if results:
                    st.success(f"Video processed successfully! Analyzed {results['processed_frames']} frames.")
                    # Create processed video with bounding boxes
                    # Use a properly named temporary file
                    bbox_video_path = tmp_path.replace(".mp4", "_with_bboxes.mp4")
                    scene_graph_video_path = tmp_path.replace(
                        ".mp4", "_with_scene_graph.mp4"
                    )

                    st.info(f"Creating bounding box video at: {bbox_video_path}")

                    with st.spinner("Creating video with bounding boxes..."):
                        bbox_success = create_processed_video_with_bboxes(
                            tmp_path, model_path, bbox_video_path, max_frames
                        )

                    if bbox_success and os.path.exists(bbox_video_path):
                        # Check if the video file has content
                        file_size = os.path.getsize(bbox_video_path)
                        st.session_state["bbox_video_path"] = bbox_video_path
                        st.success(
                            f" Bounding box video created successfully! Size: {file_size} bytes"
                        )

                        # Also create a sample frame for debugging
                        try:
                            test_cap = cv2.VideoCapture(bbox_video_path)
                            ret, test_frame = test_cap.read()
                            if ret:
                                st.write(
                                    " Bounding box video file is readable by OpenCV"
                                )
                                # Show the first frame as an image for verification
                                with st.expander(
                                    "First frame with bboxes (verification)"
                                ):
                                    st.image(
                                        cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB),
                                        caption="First frame from bbox video",
                                    )
                            else:
                                st.error(
                                    " Bounding box video file created but not readable by OpenCV"
                                )
                            test_cap.release()
                        except Exception as e:
                            st.error(f"Error verifying bbox video: {e}")
                    else:
                        st.warning(
                            f" Failed to create bounding box video. File exists: {os.path.exists(bbox_video_path) if 'bbox_video_path' in locals() else 'N/A'}"
                        )

                    # Create scene graph video
                    st.info(f"Creating scene graph video at: {scene_graph_video_path}")

                    with st.spinner("Creating video with scene graph overlay..."):
                        sg_success = create_processed_video_with_scene_graph(
                            tmp_path, model_path, scene_graph_video_path, max_frames
                        )

                    if sg_success and os.path.exists(scene_graph_video_path):
                        # Check if the video file has content
                        file_size = os.path.getsize(scene_graph_video_path)
                        st.session_state["scene_graph_video_path"] = (
                            scene_graph_video_path
                        )
                        st.success(
                            f"Scene graph video created successfully! Size: {file_size} bytes"
                        )

                        # Also create a sample frame for debugging
                        try:
                            test_cap = cv2.VideoCapture(scene_graph_video_path)
                            ret, test_frame = test_cap.read()
                            if ret:
                                st.write(
                                    "Scene graph video file is readable by OpenCV"
                                )
                                # Show the first frame as an image for verification
                                with st.expander(
                                    "First frame with scene graph (verification)"
                                ):
                                    st.image(
                                        cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB),
                                        caption="First frame from scene graph video",
                                    )
                            else:
                                st.error(
                                    "Scene graph video file created but not readable by OpenCV"
                                )
                            test_cap.release()
                        except Exception as e:
                            st.error(f"Error verifying scene graph video: {e}")
                    else:
                        st.warning(
                            f" Failed to create scene graph video. File exists: {os.path.exists(scene_graph_video_path) if 'scene_graph_video_path' in locals() else 'N/A'}"
                        )

                    if results.get("errors"):
                        with st.expander("Processing Warnings", expanded=False):
                            for error in results["errors"][:5]:  # Show first 5 errors
                                st.warning(error)

                    st.session_state["results"] = results

            finally:
                os.unlink(tmp_path) # Clean up temporary file

    main_tab1, main_tab2 = st.tabs([" SGG View", " NLP View"])
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
                if "bbox_video_path" in st.session_state and os.path.exists(
                    st.session_state["bbox_video_path"]
                ):
                    try:
                        with open(
                            st.session_state["bbox_video_path"], "rb"
                        ) as video_file:
                            video_bytes = video_file.read()
                            if len(video_bytes) > 0:
                                st.video(video_bytes)
                                st.caption("Video with bounding box overlays")
                            else:
                                st.error("Video file is empty")

                    except Exception as e:
                        st.error(f"Error loading bbox video: {e}")
                        st.video(st.session_state.uploaded_video_file)
                else:
                    st.video(st.session_state.uploaded_video_file)
                    bbox_path = st.session_state.get("bbox_video_path", "None")
                    bbox_exists = (
                        os.path.exists(bbox_path) if bbox_path != "None" else False
                    )

            # Scene Graph Video
            with vid_col3:
                st.subheader(" Scene Graph Analysis")
                if "scene_graph_video_path" in st.session_state and os.path.exists(
                    st.session_state["scene_graph_video_path"]
                ):
                    try:
                        with open(
                            st.session_state["scene_graph_video_path"], "rb"
                        ) as video_file:
                            video_bytes = video_file.read()
                            if len(video_bytes) > 0:
                                st.video(video_bytes)
                                st.caption("Video with 2D scene graph overlays")
                            else:
                                st.error("Scene graph video file is empty")

                    except Exception as e:
                        st.error(f"Error loading scene graph video: {e}")
                        st.video(st.session_state.uploaded_video_file)
                        st.caption(" Scene graph overlay failed to load")
                else:
                    st.video(st.session_state.uploaded_video_file)
                    sg_path = st.session_state.get("scene_graph_video_path", "None")
                    sg_exists = os.path.exists(sg_path) if sg_path != "None" else False

            # Chat
            st.markdown("---")
            st.header("Chat Assistant")
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
                elif "relationship" in user_input_lower and "results" in st.session_state:
                    results = st.session_state["results"]
                    avg_relationships = (
                        np.mean(results["relationships"]) if results["relationships"] else 0
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

            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for i, msg in enumerate(st.session_state.chat_messages):
                    message(
                        msg["message"],
                        is_user=msg["is_user"],
                        key=f"chat_msg_{i}",
                        allow_html=True,
                    )

            # Chat input
            st.text_input(
                "Ask me about your scene graph analysis:",
                key="chat_input",
                on_change=handle_chat_input,
                placeholder="Type your question here...",
            )

            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.chat_messages = []
                st.rerun()

            # Sub-tabs for Frame view and Temporal view
            st.markdown("---")
            sgg_tab1, sgg_tab2 = st.tabs([" Frame View", " Temporal View"])

            with sgg_tab1:
                st.header("Frame-based Scene Graph Analysis")
                st.info(" Frame view implementation will be added here")

                # Placeholder for frame-based scene graph visualization
                if "results" in st.session_state:
                    results = st.session_state["results"]
                    if results["detections"]:
                        st.subheader(" Detection Statistics")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            avg_objects = np.mean(results["detections"])
                            st.metric("Avg Objects/Frame", f"{avg_objects:.1f}")
                        with col_b:
                            avg_relationships = np.mean(results["relationships"])
                            st.metric("Avg Relations/Frame", f"{avg_relationships:.1f}")
                        with col_c:
                            avg_confidence = np.mean(results["confidences"])
                            st.metric("Avg Confidence", f"{avg_confidence:.2f}")

            with sgg_tab2:
                st.header("Temporal Scene Graph Analysis")
                st.info(" Temporal view implementation will be added here")

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

                        # Multi-line chart for detections and relationships
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
                        st.plotly_chart(fig, use_container_width=True)

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
                        st.plotly_chart(fig3, use_container_width=True)

                        # Data table
                        st.subheader("📋 Detection Details")
                        st.dataframe(df, use_container_width=True)
        else:
            st.info("Please upload a video file first to see the analysis results.")

    # NLP View Tab Implementation
    with main_tab2:
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
                st.subheader("📖 Video Summarization")
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
                st.subheader("💬 Video Captioning")
                captioning_text = """
                🎯 **Frame-by-Frame Captions:**
                
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
                🚀 **Predicted Future Actions:**
                
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
                st.plotly_chart(fig_emotion, use_container_width=True)

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
                st.plotly_chart(fig_scene, use_container_width=True)

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
                st.plotly_chart(fig_activity, use_container_width=True)

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

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>Built with ❤️ using Streamlit | VidSgg Scene Graph Generation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
