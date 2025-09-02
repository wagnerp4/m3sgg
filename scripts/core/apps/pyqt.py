import copy
import glob
import json
import os
import platform
import random
import sys
import tempfile
import time
from datetime import datetime

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Import the necessary modules from the codebase
from datasets.action_genome import AG, cuda_collate_fn
from datasets.easg import EASG
from datasets.easg import cuda_collate_fn as easg_cuda_collate_fn
from lib.config import Config
from lib.easg.object_detector_EASG import detector as detector_EASG
from lib.easg.sttran_EASG import STTran as STTran_EASG
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.matcher import HungarianMatcher
from lib.nlp.summarization_wrapper import (
    PegasusCustomConfig,
    PegasusSeparateLoader,
    PegasusSummarizationWrapper,
    T5SummarizationWrapper,
)
from lib.object_detector import detector
from lib.sttran import STKET, STTran
from lib.tempura.tempura import TEMPURA
from lib.track import get_sequence


def find_latest_checkpoint(dataset_type, model_type, mode):
    base_path = os.path.join("data", "checkpoints", dataset_type, model_type, mode)
    if not os.path.exists(base_path):
        return None
    checkpoint_files = ["model_best.tar", "model_best_Mrecall.tar"]

    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(base_path, checkpoint_file)
        if os.path.exists(checkpoint_path):
            return checkpoint_path


def set_application_icon(app, icon_path):
    try:
        ico_path = icon_path.replace(".png", ".ico")
        ico_path_alt = "assets/tum-logo.ico"

        def set_windows_icon():
            try:
                import ctypes

                myappid = "dlhm.vidsgg.gui.1.0"  # arbitrary string
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
                try:
                    from PyQt5.QtWinExtras import QtWin

                    QtWin.setCurrentProcessExplicitAppUserModelID(myappid)
                except ImportError:
                    print("PyQt5.QtWinExtras not available")
                except Exception as e:
                    print(f"QtWin method failed: {str(e)}")
            except Exception as e:
                print(f"Windows taskbar icon setting failed: {str(e)}")

        # Use ICO
        if os.path.exists(ico_path):
            icon = QIcon(ico_path)
            app.setWindowIcon(icon)
            if platform.system() == "Windows":
                set_windows_icon()
            return True
        elif os.path.exists(ico_path_alt):
            icon = QIcon(ico_path_alt)
            app.setWindowIcon(icon)
            if platform.system() == "Windows":
                set_windows_icon()
            return True

        # Fallback to PNG
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            sizes = [16, 32, 48, 64, 128]
            for size in sizes:
                icon.addFile(icon_path, QSize(size, size))

            app.setWindowIcon(icon)
            if platform.system() == "Windows":
                set_windows_icon()
            return True
        return False
    except Exception as e:
        print(f"Error setting application icon: {str(e)}")
        return False


class VideoProcessor(QThread):
    frame_processed = pyqtSignal(
        object, object, object, object
    )  # entry, pred, original_frame, im_info
    error_occurred = pyqtSignal(str)
    video_info_updated = pyqtSignal(str)
    frame_info_updated = pyqtSignal(str)  # Signal for frame information updates

    def __init__(
        self, model_manager, model_type, dataset_type, mode, checkpoint_path=None
    ):
        super().__init__()
        self.model_manager = model_manager
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        self.running = False
        self.paused = False
        self.cap = None
        self.dataset = None
        self.current_video_index = 0
        self.current_frame_index = 0
        self.target_fps = 5  # Default FPS - reduced for better demonstration

        # Video cycling parameters
        self.available_video_indices = []  # List of available video indices
        self.video_cycle_index = 0  # Current position in video cycle
        self.frames_per_video = (
            30  # Number of frames to show per video before switching
        )
        self.frame_count_in_current_video = 0  # Counter for frames in current video

        # Video selection parameters
        self.use_expanded_video_selection = (
            True  # Whether to use expanded video selection (include non-GT videos)
        )
        self.gt_video_selection_probability = 0.7  # Probability of selecting from GT videos when expanded selection is enabled

        # Navigation parameters
        self.manual_frame_navigation = (
            False  # Whether manual frame navigation is active
        )
        self.navigation_signal = None  # Signal to emit when navigation is requested

        # Initialize models and dataset like demo.py
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.object_detector = None
        self.model = None
        self.matcher = None

    def setup_models(self):
        """Initialize models following demo.py approach"""
        try:
            # Load dataset
            if self.dataset_type == "action_genome":
                self.dataset = AG(
                    mode="test",
                    datasize="large",
                    data_path="data/action_genome",
                    filter_nonperson_box_frame=True,
                    filter_small_box=False if self.mode == "predcls" else True,
                )
            elif self.dataset_type == "EASG":
                self.dataset = EASG(
                    split="val",
                    datasize="large",
                    data_path="data/EASG",
                )
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_type}")

            # Load object detector
            if self.dataset_type == "action_genome":
                self.object_detector = detector(
                    train=False,
                    object_classes=self.dataset.object_classes,
                    use_SUPPLY=True,
                    mode=self.mode,
                ).to(device=self.device)
            elif self.dataset_type == "EASG":
                self.object_detector = detector_EASG(
                    train=False,
                    object_classes=self.dataset.obj_classes,
                    use_SUPPLY=True,
                    mode=self.mode,
                ).to(device=self.device)

            self.object_detector.eval()

            # Load scene graph model
            if self.model_type == "sttran":
                if self.dataset_type == "action_genome":
                    self.model = STTran(
                        mode=self.mode,
                        attention_class_num=len(self.dataset.attention_relationships),
                        spatial_class_num=len(self.dataset.spatial_relationships),
                        contact_class_num=len(self.dataset.contacting_relationships),
                        obj_classes=self.dataset.object_classes,
                        enc_layer_num=1,
                        dec_layer_num=3,
                    ).to(device=self.device)
                elif self.dataset_type == "EASG":
                    self.model = STTran_EASG(
                        mode=self.mode,
                        obj_classes=self.dataset.obj_classes,
                        verb_classes=self.dataset.verb_classes,
                        edge_class_num=len(self.dataset.edge_classes),
                        enc_layer_num=1,
                        dec_layer_num=3,
                    ).to(device=self.device)

            elif self.model_type == "stket":
                try:
                    trainPrior = json.load(open("data/TrainPrior.json", "r"))
                except FileNotFoundError:
                    print("Warning: TrainPrior.json not found, using empty prior")
                    trainPrior = {}

                try:
                    self.model = STKET(
                        mode=self.mode,
                        attention_class_num=len(self.dataset.attention_relationships),
                        spatial_class_num=len(self.dataset.spatial_relationships),
                        contact_class_num=len(self.dataset.contacting_relationships),
                        obj_classes=self.dataset.object_classes,
                        N_layer_num=1,
                        enc_layer_num=1,
                        dec_layer_num=3,
                        pred_contact_threshold=0.5,
                        window_size=3,
                        trainPrior=trainPrior,
                        use_spatial_prior=False,
                        use_temporal_prior=False,
                    ).to(device=self.device)
                except Exception as e:
                    print(f"Warning: STKET model initialization failed: {str(e)}")
                    print("Falling back to STTran model...")

                    # Fallback to STTran model
                    self.model = STTran(
                        mode=self.mode,
                        attention_class_num=len(self.dataset.attention_relationships),
                        spatial_class_num=len(self.dataset.spatial_relationships),
                        contact_class_num=len(self.dataset.contacting_relationships),
                        obj_classes=self.dataset.object_classes,
                        enc_layer_num=1,
                        dec_layer_num=3,
                    ).to(device=self.device)

            elif self.model_type == "tempura":
                self.model = TEMPURA(
                    mode=self.mode,
                    attention_class_num=len(self.dataset.attention_relationships),
                    spatial_class_num=len(self.dataset.spatial_relationships),
                    contact_class_num=len(self.dataset.contacting_relationships),
                    obj_classes=self.dataset.object_classes,
                    enc_layer_num=1,
                    dec_layer_num=3,
                    obj_mem_compute=False,
                    rel_mem_compute=None,
                    take_obj_mem_feat=False,
                    mem_fusion="early",
                    selection="manual",
                    selection_lambda=None,
                    obj_head="gmm",
                    rel_head="gmm",
                    K=4,
                ).to(device=self.device)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # Load model weights
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                try:
                    ckpt = torch.load(self.checkpoint_path, map_location=self.device)

                    # Handle size mismatches for embedding layers
                    if self.model_type == "stket":
                        state_dict = ckpt["state_dict"]

                        # Check for embedding size mismatches
                        if "object_classifier.obj_embed.weight" in state_dict:
                            checkpoint_size = state_dict[
                                "object_classifier.obj_embed.weight"
                            ].shape[0]
                            current_size = (
                                self.model.object_classifier.obj_embed.weight.shape[0]
                            )

                            if checkpoint_size != current_size:
                                print("Warning: Embedding size mismatch detected!")
                                print(f"Checkpoint: {checkpoint_size} classes")
                                print(f"Current model: {current_size} classes")
                                print("Skipping incompatible embedding layers...")
                                keys_to_remove = [
                                    "object_classifier.obj_embed.weight",
                                    "obj_embed.weight",
                                    "obj_embed2.weight",
                                ]
                                for key in keys_to_remove:
                                    if key in state_dict:
                                        del state_dict[key]
                                        print(f"  Removed {key} from loading")

                    # Load state dict with strict=False to ignore missing keys
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        ckpt["state_dict"], strict=False
                    )

                    if missing_keys:
                        print(
                            f"Warning: Missing keys when loading model: {missing_keys}"
                        )
                    if unexpected_keys:
                        print(
                            f"Warning: Unexpected keys when loading model: {unexpected_keys}"
                        )

                    print(f"Loaded model from {self.checkpoint_path}")

                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    print("Continuing with randomly initialized model...")
            else:
                print(f"Warning: Model file {self.checkpoint_path} not found!")
                print("Using randomly initialized model...")

            self.model.eval()

            # Create matcher
            self.matcher = HungarianMatcher(0.5, 1, 1, 0.5).to(device=self.device)
            self.matcher.eval()

            return True

        except Exception as e:
            print(f"Error setting up models: {str(e)}")
            return False

    def setup_camera(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            self.error_occurred.emit("Failed to open camera")
            return False
        return True

    def setup_video_file(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.error_occurred.emit("Failed to open video file")
            return False
        return True

    def setup_dataset_video(self, video_index=None):
        """Setup dataset video for predcls/sgcls modes"""
        if video_index is None:
            # Choose a random video from available videos
            if not hasattr(self, "dataset") or self.dataset is None:
                print("Error: Dataset not loaded yet")
                return False

            if (
                not hasattr(self.dataset, "video_list")
                or len(self.dataset.video_list) == 0
            ):
                print("Error: No videos available in dataset")
                return False

            print(f"Total videos in dataset: {len(self.dataset.video_list)}")
            print(f"Total GT annotations: {len(self.dataset.gt_annotations)}")

            # Find videos with ground truth bounding boxes (primary selection)
            available_videos_with_gt = []
            # Also include all videos from the dataset for expanded selection
            all_available_videos = []

            for i, video_frames in enumerate(self.dataset.video_list):
                # Add all videos to the expanded list
                all_available_videos.append(i)

                if i < len(self.dataset.gt_annotations):
                    # Check if this video has ground truth annotations
                    video_annotations = self.dataset.gt_annotations[i]
                    if video_annotations and len(video_annotations) > 0:
                        # Check if any frame has bounding boxes
                        has_bboxes = False
                        bbox_count = 0
                        for frame_annots in video_annotations:
                            if frame_annots and len(frame_annots) > 0:
                                for obj_annot in frame_annots:
                                    if (
                                        "bbox" in obj_annot
                                        and obj_annot["bbox"] is not None
                                    ):
                                        # Check if bbox has valid coordinates (not all zeros)
                                        bbox = obj_annot["bbox"]
                                        if hasattr(bbox, "any"):  # numpy array
                                            if (
                                                bbox.any()
                                            ):  # Check if any element is non-zero
                                                has_bboxes = True
                                                bbox_count += 1
                                                break
                                        else:  # regular value
                                            if bbox:
                                                has_bboxes = True
                                                bbox_count += 1
                                                break
                            if has_bboxes:
                                break
                        if has_bboxes:
                            available_videos_with_gt.append(i)
                            print(
                                f"Video {i}: {len(video_frames)} frames, {bbox_count} bboxes (with GT)"
                            )

            print(f"Found {len(available_videos_with_gt)} videos with GT bboxes")
            print(f"Total videos in dataset: {len(all_available_videos)}")

            # Use expanded selection: prefer videos with GT bboxes, but also include others
            if available_videos_with_gt:
                if self.use_expanded_video_selection:
                    # Use configurable probability to select from videos with GT bboxes
                    if (
                        random.random() < self.gt_video_selection_probability
                        and len(available_videos_with_gt) > 0
                    ):
                        available_videos = available_videos_with_gt
                        print(
                            f"Using {len(available_videos)} videos with GT bboxes for selection"
                        )
                    else:
                        available_videos = all_available_videos
                        print(
                            f"Using {len(available_videos)} total videos for expanded selection"
                        )
                else:
                    # Use only videos with GT bboxes (original behavior)
                    available_videos = available_videos_with_gt
                    print(
                        f"Using {len(available_videos)} videos with GT bboxes for selection (expanded selection disabled)"
                    )
            else:
                # Fallback to all videos if no GT bboxes found
                available_videos = all_available_videos
                print(
                    f"Using {len(available_videos)} total videos (no GT bboxes found)"
                )

            if not available_videos:
                print("Error: No videos available in dataset")
                return False

            # Select a random video from the available set
            video_index = random.choice(available_videos)
            video_type = (
                "with GT bboxes"
                if video_index in available_videos_with_gt
                else "without GT bboxes"
            )
            print(
                f"Selected random video {video_index} ({video_type}) from {len(available_videos)} available videos"
            )

        self.current_video_index = video_index
        self.current_frame_index = 0

        video_frames = self.dataset.video_list[video_index]
        print(f"Loaded dataset video {video_index} with {len(video_frames)} frames")

        # Check if this video has ground truth annotations
        if video_index < len(self.dataset.gt_annotations):
            video_annotations = self.dataset.gt_annotations[video_index]
            frames_with_bboxes = 0
            total_bboxes = 0
            for frame_annots in video_annotations:
                if frame_annots and len(frame_annots) > 0:
                    frames_with_bboxes += 1
                    for obj_annot in frame_annots:
                        if "bbox" in obj_annot and obj_annot["bbox"] is not None:
                            bbox = obj_annot["bbox"]
                            if hasattr(bbox, "any"):  # numpy array
                                if bbox.any():  # Check if any element is non-zero
                                    total_bboxes += 1
                            else:  # regular value
                                if bbox:
                                    total_bboxes += 1

            print(
                f"Video {video_index} has {frames_with_bboxes} frames with GT bboxes, total {total_bboxes} bboxes"
            )

        # Emit a signal to update the UI with video info
        self.video_info_updated.emit(
            f"Dataset video {video_index} ({len(video_frames)} frames) - Will cycle every {self.frames_per_video} frames"
        )

        # Initialize video cycling if this is the first setup
        if not self.available_video_indices:
            self._initialize_video_cycling()

        return True

    def _initialize_video_cycling(self):
        """Initialize the list of available video indices for cycling"""
        try:
            if not hasattr(self, "dataset") or self.dataset is None:
                print("Error: Dataset not loaded yet")
                return

            if (
                not hasattr(self.dataset, "video_list")
                or len(self.dataset.video_list) == 0
            ):
                print("Error: No videos available in dataset")
                return

            print(f"Total videos in dataset: {len(self.dataset.video_list)}")
            print(f"Total GT annotations: {len(self.dataset.gt_annotations)}")

            # Find videos with ground truth bounding boxes (primary selection)
            available_videos_with_gt = []
            # Also include all videos from the dataset for expanded selection
            all_available_videos = []

            for i, video_frames in enumerate(self.dataset.video_list):
                # Add all videos to the expanded list
                all_available_videos.append(i)

                if i < len(self.dataset.gt_annotations):
                    # Check if this video has ground truth annotations
                    video_annotations = self.dataset.gt_annotations[i]
                    if video_annotations and len(video_annotations) > 0:
                        # Check if any frame has bounding boxes
                        has_bboxes = False
                        bbox_count = 0
                        for frame_annots in video_annotations:
                            if frame_annots and len(frame_annots) > 0:
                                for obj_annot in frame_annots:
                                    if (
                                        "bbox" in obj_annot
                                        and obj_annot["bbox"] is not None
                                    ):
                                        # Check if bbox has valid coordinates (not all zeros)
                                        bbox = obj_annot["bbox"]
                                        if hasattr(bbox, "any"):  # numpy array
                                            if (
                                                bbox.any()
                                            ):  # Check if any element is non-zero
                                                has_bboxes = True
                                                bbox_count += 1
                                                break
                                        else:  # regular value
                                            if bbox:
                                                has_bboxes = True
                                                bbox_count += 1
                                                break
                            if has_bboxes:
                                break
                        if has_bboxes:
                            available_videos_with_gt.append(i)
                            print(
                                f"Video {i}: {len(video_frames)} frames, {bbox_count} bboxes (with GT)"
                            )

            print(f"Found {len(available_videos_with_gt)} videos with GT bboxes")
            print(f"Total videos in dataset: {len(all_available_videos)}")

            # Use expanded selection for cycling: include both GT and non-GT videos
            if available_videos_with_gt:
                if self.use_expanded_video_selection:
                    # For cycling, use a mix of videos with and without GT bboxes
                    # Start with videos that have GT bboxes, then add others
                    available_videos = available_videos_with_gt.copy()

                    # Add some videos without GT bboxes to expand the cycling set
                    videos_without_gt = [
                        v
                        for v in all_available_videos
                        if v not in available_videos_with_gt
                    ]
                    if videos_without_gt:
                        # Add up to 50% more videos from the non-GT set
                        max_additional = min(
                            len(available_videos_with_gt) // 2, len(videos_without_gt)
                        )
                        if max_additional > 0:
                            additional_videos = random.sample(
                                videos_without_gt, max_additional
                            )
                            available_videos.extend(additional_videos)
                            print(
                                f"Added {len(additional_videos)} additional videos without GT bboxes to cycling set"
                            )

                    print(
                        f"Using {len(available_videos)} videos for cycling (mix of GT and non-GT)"
                    )
                else:
                    # Use only videos with GT bboxes for cycling (original behavior)
                    available_videos = available_videos_with_gt.copy()
                    print(
                        f"Using {len(available_videos)} videos with GT bboxes for cycling (expanded selection disabled)"
                    )
            else:
                # Fallback to all videos if no GT bboxes found
                available_videos = all_available_videos
                print(
                    f"Using {len(available_videos)} total videos for cycling (no GT bboxes found)"
                )

            if not available_videos:
                print("Error: No videos available in dataset")
                return

            # Store available video indices for cycling
            self.available_video_indices = available_videos
            self.video_cycle_index = 0
            self.frame_count_in_current_video = 0

            print(
                f"Initialized video cycling with {len(self.available_video_indices)} videos"
            )

        except Exception as e:
            print(f"Error initializing video cycling: {str(e)}")

    def _switch_to_next_video(self):
        """Switch to the next video in the cycle"""
        if not self.available_video_indices:
            return False

        # Move to next video in cycle
        self.video_cycle_index = (self.video_cycle_index + 1) % len(
            self.available_video_indices
        )
        self.current_video_index = self.available_video_indices[self.video_cycle_index]
        self.current_frame_index = 0
        self.frame_count_in_current_video = 0

        # Update video info
        video_frames = self.dataset.video_list[self.current_video_index]
        self.video_info_updated.emit(
            f"Dataset video {self.current_video_index} ({len(video_frames)} frames) - Video {self.video_cycle_index + 1}/{len(self.available_video_indices)} - Cycling every {self.frames_per_video} frames"
        )

        print(
            f"Switched to video {self.current_video_index} (cycle {self.video_cycle_index + 1}/{len(self.available_video_indices)})"
        )
        return True

    def tensor_to_uint8(self, img_t: torch.Tensor) -> np.ndarray:
        """
        Convert a CHW float32 tensor that is normalised with (img-mean)/std
        back to an HWC uint8 image **in RGB order** ready for drawing.
        """
        img = img_t.clone().cpu().float().numpy()
        # Add BGR means back
        bgr_means = np.array([102.9801, 115.9465, 122.7717], dtype=np.float32)
        img = img + bgr_means[:, None, None]
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = img.transpose(1, 2, 0)  # HWC, BGR
        img = img[..., ::-1]  # BGR to RGB
        return img

    def process_frame_demo_style(self, data):
        """Process a single frame following demo.py approach exactly"""
        im_data = copy.deepcopy(data[0]).to(device=self.device)
        im_info = copy.deepcopy(data[1]).to(device=self.device)
        gt_boxes = copy.deepcopy(data[2]).to(device=self.device)
        num_boxes = copy.deepcopy(data[3]).to(device=self.device)
        gt_annotation = self.dataset.gt_annotations[data[4]]

        with torch.no_grad():
            # Object detection
            entry = self.object_detector(
                im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None
            )

            # Scene graph generation
            if self.dataset_type == "action_genome":
                get_sequence(
                    entry,
                    gt_annotation,
                    self.matcher,
                    (im_info[0][:2] / im_info[0, 2]),
                    self.mode,
                )
            pred = self.model(entry)

        return entry, pred

    def run(self):
        self.running = True

        # Setup models first
        if not self.setup_models():
            self.error_occurred.emit("Failed to setup models")
            return

        # Frame rate control
        frame_interval = 1.0 / self.target_fps
        last_frame_time = time.time()

        while self.running:
            # Check if paused
            if self.paused:
                time.sleep(0.1)
                continue

            # Frame rate limiting
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                continue

            try:
                # Handle different video sources based on mode
                if self.mode in ["predcls", "sgcls"]:
                    # Use dataset video following demo.py approach
                    if self.dataset is None:
                        time.sleep(0.1)
                        continue

                    # Get video data like demo.py
                    data = self.dataset.__getitem__(self.current_video_index)
                    data = [
                        d.to(self.device) if isinstance(d, torch.Tensor) else d
                        for d in data
                    ]

                    # Process only the current frame
                    if self.current_frame_index >= len(
                        self.dataset.video_list[self.current_video_index]
                    ):
                        # Check if manual navigation is active
                        if self.manual_frame_navigation:
                            # In manual mode, loop back to start of current video
                            self.current_frame_index = 0
                        else:
                            # Check if we should switch to next video or loop back to start of current video
                            if (
                                self.frame_count_in_current_video
                                >= self.frames_per_video
                                and len(self.available_video_indices) > 1
                            ):
                                # Switch to next video in cycle
                                if self._switch_to_next_video():
                                    # Get data for the new video
                                    data = self.dataset.__getitem__(
                                        self.current_video_index
                                    )
                                    data = [
                                        d.to(self.device)
                                        if isinstance(d, torch.Tensor)
                                        else d
                                        for d in data
                                    ]
                                else:
                                    # Fallback: loop back to start of current video
                                    self.current_frame_index = 0
                                    self.frame_count_in_current_video = 0
                            else:
                                # Loop back to start of current video
                                self.current_frame_index = 0
                                self.frame_count_in_current_video = 0

                    im_data = data[0][
                        self.current_frame_index : self.current_frame_index + 1
                    ]
                    im_info = data[1][
                        self.current_frame_index : self.current_frame_index + 1
                    ]
                    gt_boxes = data[2][
                        self.current_frame_index : self.current_frame_index + 1
                    ]
                    num_boxes = data[3][
                        self.current_frame_index : self.current_frame_index + 1
                    ]
                    frame_data = [im_data, im_info, gt_boxes, num_boxes, data[4]]

                    # Process frame following demo.py approach
                    entry, pred = self.process_frame_demo_style(frame_data)

                    # Convert tensor to image like demo.py
                    im_tensor = frame_data[0][0]  # CHW, float32, normalised
                    frame = self.tensor_to_uint8(im_tensor)  # HWC, uint8, RGB

                    # Only increment frame count if not in manual navigation mode
                    if not self.manual_frame_navigation:
                        self.current_frame_index += 1
                        self.frame_count_in_current_video += 1
                    else:
                        # In manual mode, don't auto-increment frame index
                        # The frame index is controlled by navigation buttons
                        pass

                else:
                    # Use camera/video file (sgdet mode)
                    if self.cap is None:
                        time.sleep(0.1)
                        continue

                    ret, frame = self.cap.read()
                    if not ret:
                        print("End of video stream reached")
                        break

                    # Process frame for sgdet mode (simplified approach)
                    entry, pred = self.process_frame_sgdet(frame)

                # Update frame time
                last_frame_time = current_time

                # Emit frame with rate limiting
                # For predcls/sgcls modes, also pass the im_info for proper box scaling
                if self.mode in ["predcls", "sgcls"]:
                    im_info = frame_data[1][0]  # Get the im_info for this frame
                    self.frame_processed.emit(entry, pred, frame, im_info)

                    # Emit frame info update
                    frame_info = self.get_frame_info()
                    self.frame_info_updated.emit(frame_info)
                else:
                    # For sgdet mode, create a simple im_info
                    im_info = torch.tensor(
                        [[frame.shape[1], frame.shape[0], 1.0]], dtype=torch.float32
                    ).to(self.device)
                    self.frame_processed.emit(entry, pred, frame, im_info)

                # Small delay to prevent GUI freezing
                time.sleep(0.001)

            except Exception as e:
                print(f"Error in video processing loop: {str(e)}")
                # Don't emit error signal to avoid infinite dialogs
                # Just log the error and continue
                time.sleep(0.1)  # Brief pause before continuing
                continue

    def process_frame_sgdet(self, frame):
        """Process frame for sgdet mode (simplified approach)"""
        try:
            # Validate input frame
            if frame is None:
                raise ValueError("Frame is None")

            if not isinstance(frame, np.ndarray):
                raise ValueError(f"Frame is not numpy array, got {type(frame)}")

            # Convert frame to tensor format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize and normalize
            frame_resized = cv2.resize(frame_rgb, (600, 600))
            frame_tensor = (
                torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0)
            )

            # Normalize
            frame_tensor = frame_tensor / 255.0
            frame_tensor = (
                frame_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            ) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

            # Create input data structure
            im_data = frame_tensor.to(self.device)
            im_info = torch.tensor([[600, 600, 1.0]], dtype=torch.float32).to(
                self.device
            )
            gt_boxes = torch.zeros([1, 1, 5]).to(self.device)
            num_boxes = torch.zeros([1], dtype=torch.int64).to(self.device)

            with torch.no_grad():
                # Object detection for sgdet mode only
                empty_annotation = []
                entry = self.object_detector(
                    im_data, im_info, gt_boxes, num_boxes, empty_annotation, im_all=None
                )

                # Validate entry is a dictionary
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Object detector returned {type(entry)}, expected dict"
                    )

                # Scene graph generation for sgdet mode
                pred = self.model(entry)

                # Validate pred is a dictionary
                if not isinstance(pred, dict):
                    raise ValueError(f"Model returned {type(pred)}, expected dict")

            return entry, pred

        except Exception as e:
            # Create safe fallback data
            print(f"Frame processing error: {str(e)}")

            fallback_entry = {
                "boxes": torch.empty(0, 5).to(self.device),
                "features": torch.empty(0, 2048).to(self.device),
                "distribution": torch.empty(0, 37).to(self.device),
                "labels": torch.empty(0, dtype=torch.long).to(self.device),
                "pair_idx": torch.empty(0, 2).to(self.device),
                "im_idx": torch.empty(0, dtype=torch.long).to(self.device),
                "union_feat": torch.empty(0, 1024, 7, 7).to(self.device),
                "spatial_masks": torch.empty(0, 2, 224, 224).to(self.device),
            }

            fallback_pred = {
                "distribution": torch.empty(0, 37).to(self.device),
                "labels": torch.empty(0, dtype=torch.long).to(self.device),
                "attention_distribution": torch.empty(0, 3).to(self.device),
                "spatial_distribution": torch.empty(0, 6).to(self.device),
                "contact_distribution": torch.empty(0, 17).to(self.device),
            }

            return fallback_entry, fallback_pred

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def set_fps(self, fps):
        """Update the target FPS for frame processing"""
        self.target_fps = max(1, min(60, fps))  # Clamp between 1 and 60 FPS

    def skip_to_next_video(self):
        """Skip to the next video in the cycling sequence"""
        if self.available_video_indices and len(self.available_video_indices) > 1:
            self._switch_to_next_video()
            return True
        return False

    def navigate_frame(self, direction):
        """Navigate to the previous or next frame"""
        if not hasattr(self, "dataset") or self.dataset is None:
            return False

        if self.current_video_index >= len(self.dataset.video_list):
            return False

        video_frames = self.dataset.video_list[self.current_video_index]
        total_frames = len(video_frames)

        if direction == "next":
            if self.current_frame_index < total_frames - 1:
                self.current_frame_index += 1
                return True
        elif direction == "previous":
            if self.current_frame_index > 0:
                self.current_frame_index -= 1
                return True
        elif direction == "first":
            self.current_frame_index = 0
            return True
        elif direction == "last":
            self.current_frame_index = total_frames - 1
            return True

        return False

    def get_frame_info(self):
        """Get current frame information"""
        if not hasattr(self, "dataset") or self.dataset is None:
            return "Frame: 0/0"

        if self.current_video_index >= len(self.dataset.video_list):
            return "Frame: 0/0"

        video_frames = self.dataset.video_list[self.current_video_index]
        total_frames = len(video_frames)
        current_frame = self.current_frame_index + 1  # 1-based for display

        return f"Frame: {current_frame}/{total_frames}"

    def set_manual_navigation(self, enabled, navigation_signal=None):
        """Enable or disable manual frame navigation"""
        self.manual_frame_navigation = enabled
        self.navigation_signal = navigation_signal


class SceneGraphVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.dataset = None  # Will be set when model is loaded

    def setup_ui(self):
        layout = QVBoxLayout()

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def update_scene_graph(self, entry, pred, frame):
        try:
            # Validate inputs
            if entry is None or not isinstance(entry, dict):
                print("Warning: Invalid entry for scene graph")
                return

            if pred is None or not isinstance(pred, dict):
                print("Warning: Invalid pred for scene graph")
                return

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 6)
            ax.axis("off")
            self.figure.patch.set_facecolor("white")

            # Extract object labels - prioritize ground truth for better demonstration
            if self.dataset and hasattr(self.dataset, "object_classes"):
                # For predcls mode, use ground truth labels (better for demonstration)
                if (
                    "labels" in entry
                    and isinstance(entry["labels"], torch.Tensor)
                    and entry["labels"].shape[0] > 0
                ):
                    obj_labels = entry["labels"]
                    if isinstance(obj_labels, torch.Tensor):
                        obj_labels = obj_labels.cpu().numpy()
                    use_ground_truth = True
                else:
                    # For sgcls/sgdet modes, use predicted labels
                    pred_labels = entry.get(
                        "pred_labels", torch.empty(0, dtype=torch.long)
                    )
                    if isinstance(pred_labels, torch.Tensor):
                        obj_labels = pred_labels.cpu().numpy()
                    else:
                        obj_labels = np.array(pred_labels)
                    use_ground_truth = False
            else:
                # Fallback for when dataset is not available
                pred_labels = entry.get("pred_labels", torch.empty(0, dtype=torch.long))
                if isinstance(pred_labels, torch.Tensor):
                    obj_labels = pred_labels.cpu().numpy()
                else:
                    obj_labels = np.array(pred_labels)
                use_ground_truth = False

            # Get boxes with safe defaults
            if "boxes" in entry and isinstance(entry["boxes"], torch.Tensor):
                boxes = entry["boxes"]
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes[:, 1:].cpu().numpy()
                else:
                    boxes = (
                        boxes[:, 1:]
                        if len(boxes.shape) > 1
                        else np.array([[100, 100, 200, 200]])
                    )
            else:
                boxes = np.array([[100, 100, 200, 200]])

            # Apply NMS filtering
            if len(boxes) > 0 and len(obj_labels) > 0:
                from torchvision.ops import nms

                final_indices = []
                for class_idx in np.unique(obj_labels):
                    class_mask = obj_labels == class_idx
                    class_boxes = boxes[class_mask]
                    if len(class_boxes) > 0:
                        boxes_tensor = torch.tensor(class_boxes, dtype=torch.float32)
                        scores_tensor = torch.ones(
                            len(class_boxes), dtype=torch.float32
                        )
                        keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.0)
                        class_indices = np.where(class_mask)[0][keep.cpu().numpy()]
                        final_indices.extend(class_indices)

                # Filter detections
                if final_indices:
                    boxes = boxes[final_indices]
                    obj_labels = obj_labels[final_indices]

            # Extract relationships - prioritize ground truth relationships for demonstration
            pred_relations = []

            # If using ground truth, try to get ground truth relationships
            if use_ground_truth and "gt_relations" in entry:
                gt_relations = entry["gt_relations"]
                if isinstance(gt_relations, torch.Tensor):
                    gt_relations = gt_relations.cpu().numpy()

                # Convert ground truth relationships to our format
                for rel in gt_relations:
                    if len(rel) >= 3:  # subject, object, predicate
                        subj_idx, obj_idx, predicate = rel[0], rel[1], rel[2]
                        pred_relations.append((subj_idx, obj_idx, predicate, "gt", 1.0))

            # If no ground truth relationships or not using ground truth, use predicted relationships
            if (
                not pred_relations
                and "pair_idx" in entry
                and entry["pair_idx"].shape[0] > 0
            ):
                pair_indices = entry["pair_idx"]
                if isinstance(pair_indices, torch.Tensor):
                    pair_indices = pair_indices.cpu().numpy()

                # Get relationship distributions
                attention_dist = pred.get("attention_distribution", torch.zeros(1, 3))
                if isinstance(attention_dist, torch.Tensor):
                    attention_dist = attention_dist.cpu().numpy()
                else:
                    attention_dist = np.array(attention_dist)

                spatial_dist = pred.get("spatial_distribution", torch.zeros(1, 6))
                if isinstance(spatial_dist, torch.Tensor):
                    spatial_dist = spatial_dist.cpu().numpy()
                else:
                    spatial_dist = np.array(spatial_dist)

                contact_dist = pred.get("contact_distribution", torch.zeros(1, 17))
                if isinstance(contact_dist, torch.Tensor):
                    contact_dist = contact_dist.cpu().numpy()
                else:
                    contact_dist = np.array(contact_dist)

                for i, (subj_idx, obj_idx) in enumerate(pair_indices):
                    if i < len(attention_dist):
                        # Attention relationships
                        att_pred = np.argmax(attention_dist[i])
                        att_conf = attention_dist[i][att_pred]
                        if att_conf > 0.3:  # Lower threshold for more relationships
                            pred_relations.append(
                                (subj_idx, obj_idx, att_pred, "attention", att_conf)
                            )

                    if i < len(spatial_dist):
                        # Spatial relationships
                        spa_pred = np.argmax(spatial_dist[i])
                        spa_conf = spatial_dist[i][spa_pred]
                        if spa_conf > 0.3:
                            pred_relations.append(
                                (subj_idx, obj_idx, spa_pred, "spatial", spa_conf)
                            )

                    if i < len(contact_dist):
                        # Contact relationships
                        con_pred = np.argmax(contact_dist[i])
                        con_conf = contact_dist[i][con_pred]
                        if con_conf > 0.3:
                            pred_relations.append(
                                (subj_idx, obj_idx, con_pred, "contact", con_conf)
                            )

                # Sort by confidence and take top k
                pred_relations.sort(key=lambda x: x[4], reverse=True)
                pred_relations = pred_relations[:5]  # Show more relationships

            # Create scene graph visualization with improved layout
            self._draw_scene_graph(
                ax, pred_relations, obj_labels, boxes, use_ground_truth
            )

            self.canvas.draw()

        except Exception as e:
            print(f"Error updating scene graph: {str(e)}")

    def _draw_scene_graph(
        self, ax, pred_relations, object_labels, boxes, use_ground_truth=False
    ):
        try:
            # Add nodes (objects) involved in relationships
            unique_objects = set()
            for rel in pred_relations:
                subj_idx, obj_idx, predicate, rel_type, conf = rel
                if subj_idx < len(object_labels) and obj_idx < len(object_labels):
                    # Get object labels from dataset if available
                    if self.dataset and hasattr(self.dataset, "object_classes"):
                        subj_label = self.dataset.object_classes[
                            object_labels[subj_idx]
                        ]
                        obj_label = self.dataset.object_classes[object_labels[obj_idx]]
                    else:
                        subj_label = f"obj_{object_labels[subj_idx]}"
                        obj_label = f"obj_{object_labels[obj_idx]}"
                    unique_objects.add((subj_idx, subj_label))
                    unique_objects.add((obj_idx, obj_label))

            # If no relationships, add all objects
            if not unique_objects and len(object_labels) > 0:
                for i, label in enumerate(object_labels):
                    if self.dataset and hasattr(self.dataset, "object_classes"):
                        obj_label = self.dataset.object_classes[label]
                    else:
                        obj_label = f"obj_{label}"
                    unique_objects.add((i, obj_label))

            # Build mapping: index -> unique label (e.g., person 1, person 2)
            instance_counts = {}
            idx_to_unique_label = {}
            for idx, label in sorted(unique_objects):
                count = instance_counts.get(label, 0) + 1
                instance_counts[label] = count
                unique_label = (
                    f"{label} {count}"
                    if instance_counts[label] > 1
                    or list(instance_counts.values()).count(count) > 1
                    else label
                )
                idx_to_unique_label[idx] = unique_label

            # Improved positioning with overlap avoidance
            positions = {}
            if len(unique_objects) > 0:
                objects_list = list(unique_objects)

                # Sort objects by importance (person first, then others)
                def get_importance(label):
                    if "person" in label.lower():
                        return 0
                    elif "phone" in label.lower() or "laptop" in label.lower():
                        return 1
                    else:
                        return 2

                objects_list.sort(key=lambda x: get_importance(x[1]))

                if len(objects_list) == 1:
                    positions[objects_list[0][0]] = (5, 3)
                elif len(objects_list) == 2:
                    positions[objects_list[0][0]] = (3.5, 3)
                    positions[objects_list[1][0]] = (6.5, 3)
                else:
                    # Use force-directed layout for better positioning
                    positions = self._calculate_force_directed_layout(
                        objects_list, pred_relations
                    )

                # Draw object nodes with improved styling
                for idx, label in objects_list:
                    if idx in positions:
                        x, y = positions[idx]
                        unique_label = idx_to_unique_label[idx]

                        # Use different colors based on object type
                        if "person" in unique_label.lower():
                            node_color = "lightgreen"
                            edge_color = "darkgreen"
                        elif (
                            "phone" in unique_label.lower()
                            or "laptop" in unique_label.lower()
                        ):
                            node_color = "lightcoral"
                            edge_color = "darkred"
                        else:
                            node_color = "lightblue"
                            edge_color = "darkblue"

                        circle = plt.Circle(
                            (x, y),
                            0.5,  # Smaller radius to reduce overlaps
                            color=node_color,
                            alpha=0.9,
                            linewidth=2,
                            edgecolor=edge_color,
                        )
                        ax.add_patch(circle)

                        # Add text with better positioning
                        ax.text(
                            x,
                            y,
                            unique_label,
                            ha="center",
                            va="center",
                            fontsize=9,
                            weight="bold",
                            bbox=dict(
                                boxstyle="round,pad=0.2",
                                facecolor="white",
                                alpha=0.8,
                                edgecolor=edge_color,
                            ),
                        )

            # Draw relationships with improved visibility
            for rel in pred_relations:
                subj_idx, obj_idx, predicate, rel_type, conf = rel
                if (
                    subj_idx in positions
                    and obj_idx in positions
                    and subj_idx != obj_idx
                ):
                    x1, y1 = positions[subj_idx]
                    x2, y2 = positions[obj_idx]

                    # Determine relationship type and color
                    if rel_type == "gt":
                        # Ground truth relationships
                        if self.dataset and hasattr(
                            self.dataset, "relationship_classes"
                        ):
                            rel_name = self.dataset.relationship_classes[predicate]
                        else:
                            rel_name = f"gt_{predicate}"
                        color = "purple"
                        line_style = "-"
                    elif rel_type == "attention":
                        if self.dataset and hasattr(
                            self.dataset, "attention_relationships"
                        ):
                            rel_name = self.dataset.attention_relationships[predicate]
                        else:
                            rel_name = f"attention_{predicate}"
                        color = "red"
                        line_style = "--"
                    elif rel_type == "spatial":
                        if self.dataset and hasattr(
                            self.dataset, "spatial_relationships"
                        ):
                            rel_name = self.dataset.spatial_relationships[predicate]
                        else:
                            rel_name = f"spatial_{predicate}"
                        color = "blue"
                        line_style = "-"
                    else:  # contact
                        if self.dataset and hasattr(
                            self.dataset, "contacting_relationships"
                        ):
                            rel_name = self.dataset.contacting_relationships[predicate]
                        else:
                            rel_name = f"contact_{predicate}"
                        color = "green"
                        line_style = "-"

                    # Draw arrow with better visibility and avoid overlaps
                    ax.annotate(
                        "",
                        xy=(x2, y2),
                        xytext=(x1, y1),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=color,
                            lw=2,
                            alpha=0.8,
                            linestyle=line_style,
                        ),
                    )

                    # Add relationship label with offset to avoid overlaps
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    # Add small offset to avoid overlapping with nodes
                    offset_x = (y2 - y1) * 0.1
                    offset_y = -(x2 - x1) * 0.1

                    ax.text(
                        mid_x + offset_x,
                        mid_y + offset_y,
                        rel_name,
                        ha="center",
                        va="center",
                        fontsize=7,
                        weight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            alpha=0.9,
                            edgecolor=color,
                        ),
                    )

            # Add title indicating data source
            title = "Scene Graph" if use_ground_truth else "Scene Graph (Predicted)"
            ax.set_title(title, fontsize=12, weight="bold")

        except Exception as e:
            print(f"Error drawing scene graph: {str(e)}")

    def _calculate_force_directed_layout(self, objects_list, pred_relations):
        """Calculate positions using a simple force-directed layout to avoid overlaps"""
        positions = {}

        # Initialize positions in a circle
        n_objects = len(objects_list)
        for i, (idx, label) in enumerate(objects_list):
            angle = 2 * np.pi * i / n_objects
            x = 5 + 2.5 * np.cos(angle)
            y = 3 + 1.5 * np.sin(angle)
            positions[idx] = (x, y)

        # Simple force-directed adjustment
        for _ in range(10):  # 10 iterations
            for idx, label in objects_list:
                if idx in positions:
                    x, y = positions[idx]

                    # Repulsion from other nodes
                    fx, fy = 0, 0
                    for other_idx, other_label in objects_list:
                        if other_idx != idx and other_idx in positions:
                            ox, oy = positions[other_idx]
                            dx, dy = x - ox, y - oy
                            dist = max(0.1, np.sqrt(dx * dx + dy * dy))
                            if dist < 1.5:  # Repulsion radius
                                force = (1.5 - dist) / dist
                                fx += dx * force * 0.1
                                fy += dy * force * 0.1

                    # Attraction for related nodes
                    for rel in pred_relations:
                        subj_idx, obj_idx, predicate, rel_type, conf = rel
                        if subj_idx == idx and obj_idx in positions:
                            ox, oy = positions[obj_idx]
                            dx, dy = ox - x, oy - y
                            dist = max(0.1, np.sqrt(dx * dx + dy * dy))
                            if dist > 0.5:  # Attraction radius
                                force = (dist - 0.5) / dist
                                fx += dx * force * 0.05
                                fy += dy * force * 0.05

                    # Update position
                    new_x = x + fx
                    new_y = y + fy

                    # Keep within bounds
                    new_x = max(1, min(9, new_x))
                    new_y = max(1, min(5, new_y))

                    positions[idx] = (new_x, new_y)

        return positions


class TemporalGraphVisualizer(QWidget):
    """Visualizer for temporal sequence graphs showing object relationships across frames"""

    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.dataset = None
        self.temporal_data = []  # Store temporal sequence data
        self.max_temporal_frames = 5

    def setup_ui(self):
        layout = QVBoxLayout()

        # Create matplotlib figure for temporal graph
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def update_temporal_graph(self, temporal_entries, temporal_preds, temporal_frames):
        """Update the temporal graph with new sequence data"""
        try:
            # Filter out None entries before creating temporal data
            valid_data = []
            for entry, pred, frame in zip(
                temporal_entries, temporal_preds, temporal_frames
            ):
                if entry is not None and pred is not None and frame is not None:
                    valid_data.append((entry, pred, frame))

            # Keep only the last 5 frames to prevent memory issues and halting
            if len(valid_data) > 5:
                valid_data = valid_data[-5:]

            self.temporal_data = valid_data
            self._draw_temporal_graph()
        except Exception as e:
            print(f"Error updating temporal graph: {str(e)}")
            # Don't let temporal graph errors crash the application

    def _draw_temporal_graph(self):
        """Draw the temporal graph showing object relationships across frames"""
        try:
            # Limit to maximum 5 frames for performance
            if len(self.temporal_data) > 5:
                self.temporal_data = self.temporal_data[-5:]

            self.figure.clear()

            if len(self.temporal_data) < 1:
                # Not enough data for temporal graph
                ax = self.figure.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    "Need at least 1 frame for temporal graph",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.axis("off")
                self.canvas.draw()
                return

            # Filter out None entries
            valid_data = [
                (entry, pred, frame)
                for entry, pred, frame in self.temporal_data
                if entry is not None and pred is not None and frame is not None
            ]

            if len(valid_data) < 2:
                # Not enough valid data for temporal graph
                ax = self.figure.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    "Need at least 2 valid frames for temporal graph",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.axis("off")
                self.canvas.draw()
                return

            # Create subplot for temporal graph
            ax = self.figure.add_subplot(111)
            ax.set_xlim(0, len(valid_data) * 2 + 1)
            ax.set_ylim(0, 10)
            ax.axis("off")
            self.figure.patch.set_facecolor("white")

            # Extract object information from all frames
            all_objects = {}  # object_id -> (label, color, positions)
            object_counter = 0

            # Process each frame to extract objects
            for frame_idx, (entry, pred, frame) in enumerate(valid_data):
                if entry is None or pred is None:
                    continue

                # Extract object labels and boxes - prioritize ground truth for better demonstration
                if (
                    "labels" in entry
                    and isinstance(entry["labels"], torch.Tensor)
                    and entry["labels"].shape[0] > 0
                ):
                    obj_labels = entry["labels"]
                    if isinstance(obj_labels, torch.Tensor):
                        obj_labels = obj_labels.cpu().numpy()
                else:
                    pred_labels = entry.get(
                        "pred_labels", torch.empty(0, dtype=torch.long)
                    )
                    if isinstance(pred_labels, torch.Tensor):
                        obj_labels = pred_labels.cpu().numpy()
                    else:
                        obj_labels = np.array(pred_labels)

                if "boxes" in entry and isinstance(entry["boxes"], torch.Tensor):
                    boxes = entry["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes[:, 1:].cpu().numpy()
                    else:
                        boxes = boxes[:, 1:] if len(boxes.shape) > 1 else np.array([])
                else:
                    boxes = np.array([])

                # Apply NMS filtering to reduce duplicate objects
                if len(boxes) > 0 and len(obj_labels) > 0:
                    from torchvision.ops import nms

                    final_indices = []
                    for class_idx in np.unique(obj_labels):
                        class_mask = obj_labels == class_idx
                        class_boxes = boxes[class_mask]
                        if len(class_boxes) > 0:
                            boxes_tensor = torch.tensor(
                                class_boxes, dtype=torch.float32
                            )
                            scores_tensor = torch.ones(
                                len(class_boxes), dtype=torch.float32
                            )
                            keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.0)
                            class_indices = np.where(class_mask)[0][keep.cpu().numpy()]
                            final_indices.extend(class_indices)

                    # Filter detections
                    if final_indices:
                        boxes = boxes[final_indices]
                        obj_labels = obj_labels[final_indices]

                # Create object nodes for this frame
                for obj_idx, (label, box) in enumerate(zip(obj_labels, boxes)):
                    if (
                        self.dataset
                        and hasattr(self.dataset, "object_classes")
                        and label < len(self.dataset.object_classes)
                    ):
                        obj_label = self.dataset.object_classes[label]
                    else:
                        obj_label = f"obj_{label}"

                    # Create unique object identifier based on class and frame
                    obj_id = f"{obj_label}_{frame_idx}"

                    if obj_id not in all_objects:
                        all_objects[obj_id] = {
                            "label": obj_label,
                            "color": f"C{object_counter % 10}",
                            "positions": [],
                            "frame_indices": [],
                        }
                        object_counter += 1

                    # Add position for this frame
                    all_objects[obj_id]["positions"].append(
                        (frame_idx * 2 + 1, 8 - obj_idx)
                    )
                    all_objects[obj_id]["frame_indices"].append(frame_idx)

            # Draw object nodes for each frame (limit to prevent performance issues)
            object_count = 0
            max_objects = 20  # Limit objects to prevent performance issues

            for obj_id, obj_data in all_objects.items():
                if object_count >= max_objects:
                    break

                color = obj_data["color"]
                label = obj_data["label"]

                # Draw nodes for each frame where this object appears
                for pos_idx, (x, y) in enumerate(obj_data["positions"]):
                    if object_count >= max_objects:
                        break

                    frame_idx = obj_data["frame_indices"][pos_idx]

                    # Draw object node
                    circle = plt.Circle(
                        (x, y),
                        0.4,
                        color=color,
                        alpha=0.8,
                        linewidth=2,
                        edgecolor="black",
                    )
                    ax.add_patch(circle)

                    # Add object label
                    ax.text(
                        x,
                        y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                        weight="bold",
                    )

                    # Add frame number
                    ax.text(
                        x,
                        y - 0.8,
                        f"F{frame_idx + 1}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="gray",
                    )

                    object_count += 1

            # Draw relationships between consecutive frames (simplified to prevent halting)
            # For now, just show object stacks without complex relationships
            pass

            # Add frame labels at the bottom
            for frame_idx in range(len(valid_data)):
                ax.text(
                    frame_idx * 2 + 1,
                    0.5,
                    f"Frame {frame_idx + 1}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    weight="bold",
                )

            ax.set_title("Temporal Scene Graph", fontsize=14, weight="bold")
            self.canvas.draw()

        except Exception as e:
            print(f"Error drawing temporal graph: {str(e)}")

    def _extract_relationships(self, entry, pred):
        """Extract relationships from entry and prediction data"""
        relations = []

        if "pair_idx" in entry and entry["pair_idx"].shape[0] > 0:
            # Ensure tensor is on CPU and convert to numpy
            pair_indices = entry["pair_idx"]
            if isinstance(pair_indices, torch.Tensor):
                pair_indices = pair_indices.cpu().numpy()

            # Get relationship distributions with proper CPU conversion
            attention_dist = pred.get("attention_distribution", torch.zeros(1, 3))
            if isinstance(attention_dist, torch.Tensor):
                attention_dist = attention_dist.cpu().numpy()
            else:
                attention_dist = np.array(attention_dist)

            spatial_dist = pred.get("spatial_distribution", torch.zeros(1, 6))
            if isinstance(spatial_dist, torch.Tensor):
                spatial_dist = spatial_dist.cpu().numpy()
            else:
                spatial_dist = np.array(spatial_dist)

            contact_dist = pred.get("contact_distribution", torch.zeros(1, 17))
            if isinstance(contact_dist, torch.Tensor):
                contact_dist = contact_dist.cpu().numpy()
            else:
                contact_dist = np.array(contact_dist)

            for i, (subj_idx, obj_idx) in enumerate(pair_indices):
                if i < len(attention_dist):
                    # Attention relationships
                    att_pred = np.argmax(attention_dist[i])
                    att_conf = attention_dist[i][att_pred]
                    if att_conf > 0.5:
                        relations.append(
                            (subj_idx, obj_idx, att_pred, "attention", att_conf)
                        )

                if i < len(spatial_dist):
                    # Spatial relationships
                    spa_pred = np.argmax(spatial_dist[i])
                    spa_conf = spatial_dist[i][spa_pred]
                    if spa_conf > 0.5:
                        relations.append(
                            (subj_idx, obj_idx, spa_pred, "spatial", spa_conf)
                        )

                if i < len(contact_dist):
                    # Contact relationships
                    con_pred = np.argmax(contact_dist[i])
                    con_conf = contact_dist[i][con_pred]
                    if con_conf > 0.5:
                        relations.append(
                            (subj_idx, obj_idx, con_pred, "contact", con_conf)
                        )

        # Sort by confidence and take top k
        relations.sort(key=lambda x: x[4], reverse=True)
        return relations[:5]  # Top 5 relationships

    def _find_object_position(self, frame_idx, obj_idx, all_objects):
        """Find the position of an object in the temporal graph"""
        for obj_id, obj_data in all_objects.items():
            if frame_idx in obj_data["frame_indices"]:
                frame_pos_idx = obj_data["frame_indices"].index(frame_idx)
                return obj_data["positions"][frame_pos_idx]
        return None

    def clear_temporal_data(self):
        """Clear temporal data to free memory"""
        self.temporal_data.clear()
        self.figure.clear()
        if hasattr(self, "canvas"):
            self.canvas.draw()


class TextSummarizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.summary_history = []
        self.summarization_wrapper = None
        self.max_length = 150
        self.auto_summarize = True

    def setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Text Summarization")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)

        # Summary text area
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(200)
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def set_summarization_wrapper(self, wrapper):
        """Set the summarization wrapper to use"""
        self.summarization_wrapper = wrapper
        if wrapper:
            self.status_label.setText(f"Model: {type(wrapper).__name__}")
        else:
            self.status_label.setText("No model loaded")

    def update_parameters(self, max_length, auto_summarize):
        """Update summarization parameters"""
        self.max_length = max_length
        self.auto_summarize = auto_summarize

    def update_summary(self, entry, pred, frame):
        try:
            if not self.auto_summarize:
                return

            # Validate inputs
            if entry is None or not isinstance(entry, dict):
                print("Warning: Invalid entry for text summary")
                return

            if pred is None or not isinstance(pred, dict):
                print("Warning: Invalid pred for text summary")
                return

            # Generate summary based on predictions
            summary = self.generate_summary(entry, pred)

            # Add to history
            self.summary_history.append(summary)
            if len(self.summary_history) > 10:
                self.summary_history.pop(0)

            # Display current summary
            self.summary_text.setText(summary)

        except Exception as e:
            print(f"Error updating text summary: {str(e)}")

    def generate_summary(self, entry, pred):
        try:
            # If no summarization wrapper is available, fall back to simple summary
            if self.summarization_wrapper is None:
                return self.generate_simple_summary(entry, pred)

            # Generate text description from scene graph predictions
            description = self.generate_scene_description(entry, pred)

            if not description or description.strip() == "":
                return "No significant objects or relationships detected."

            # Use the summarization wrapper to generate summary
            try:
                summary = self.summarization_wrapper.summarize(
                    description,
                    max_length=self.max_length,
                    min_length=20,
                    num_beams=4,
                    early_stopping=True,
                )
                return summary
            except Exception as e:
                print(f"Error in summarization wrapper: {str(e)}")
                # Fall back to simple summary
                return self.generate_simple_summary(entry, pred)

        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return "Error generating summary."

    def generate_simple_summary(self, entry, pred):
        """Generate a simple summary without using the summarization wrapper"""
        try:
            summary_parts = []

            # Object detection summary
            if "distribution" in pred and isinstance(
                pred["distribution"], torch.Tensor
            ):
                obj_scores = pred["distribution"].cpu().numpy()
                obj_labels = (
                    pred.get("labels", pred.get("pred_labels", torch.tensor([1])))
                    .cpu()
                    .numpy()
                )

                objects = []
                for i, (score, label) in enumerate(zip(obj_scores, obj_labels)):
                    if score.max() > 0.5:
                        objects.append(f"object_{label}")

                if objects:
                    summary_parts.append(f"Detected objects: {', '.join(objects)}")

            # Relationship summary
            if "attention_distribution" in pred and isinstance(
                pred["attention_distribution"], torch.Tensor
            ):
                attention_scores = pred["attention_distribution"].cpu().numpy()
                spatial_scores = (
                    pred.get("spatial_distribution", torch.zeros(1, 6)).cpu().numpy()
                )
                contact_scores = (
                    pred.get("contact_distribution", torch.zeros(1, 17)).cpu().numpy()
                )

                relationships = []

                # Attention relationships
                for i, scores in enumerate(attention_scores):
                    if scores.max() > 0.5:
                        relationships.append("attention")

                # Spatial relationships
                for i, scores in enumerate(spatial_scores):
                    if scores.max() > 0.5:
                        relationships.append("spatial")

                # Contact relationships
                for i, scores in enumerate(contact_scores):
                    if scores.max() > 0.5:
                        relationships.append("contact")

                if relationships:
                    summary_parts.append(f"Relationships: {', '.join(relationships)}")

            # Generate final summary
            if summary_parts:
                summary = ". ".join(summary_parts)
            else:
                summary = "No significant objects or relationships detected."

            # Truncate to specified length
            if len(summary) > self.max_length:
                summary = summary[: self.max_length - 3] + "..."

            return summary

        except Exception as e:
            print(f"Error generating simple summary: {str(e)}")
            return "Error generating summary."

    def generate_scene_description(self, entry, pred):
        """Generate a detailed scene description for summarization"""
        try:
            description_parts = []

            # Object detection description
            if "distribution" in pred and isinstance(
                pred["distribution"], torch.Tensor
            ):
                obj_scores = pred["distribution"].cpu().numpy()
                obj_labels = (
                    pred.get("labels", pred.get("pred_labels", torch.tensor([1])))
                    .cpu()
                    .numpy()
                )

                objects = []
                for i, (score, label) in enumerate(zip(obj_scores, obj_labels)):
                    if (
                        score.max() > 0.3
                    ):  # Lower threshold for more comprehensive description
                        objects.append(f"object_{label}")

                if objects:
                    description_parts.append(
                        f"The scene contains {', '.join(objects)}."
                    )

            # Relationship description
            if "attention_distribution" in pred and isinstance(
                pred["attention_distribution"], torch.Tensor
            ):
                attention_scores = pred["attention_distribution"].cpu().numpy()
                spatial_scores = (
                    pred.get("spatial_distribution", torch.zeros(1, 6)).cpu().numpy()
                )
                contact_scores = (
                    pred.get("contact_distribution", torch.zeros(1, 17)).cpu().numpy()
                )

                relationships = []

                # Attention relationships
                for i, scores in enumerate(attention_scores):
                    if scores.max() > 0.3:
                        relationships.append("attention")

                # Spatial relationships
                for i, scores in enumerate(spatial_scores):
                    if scores.max() > 0.3:
                        relationships.append("spatial")

                # Contact relationships
                for i, scores in enumerate(contact_scores):
                    if scores.max() > 0.3:
                        relationships.append("contact")

                if relationships:
                    description_parts.append(
                        f"The objects have {', '.join(relationships)} relationships."
                    )

            # Combine into a comprehensive description
            if description_parts:
                description = " ".join(description_parts)
            else:
                description = "The scene shows various objects and their interactions."

            return description

        except Exception as e:
            print(f"Error generating scene description: {str(e)}")
            return "The scene contains objects and their relationships."


class VideoDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.dataset = None  # Will be set when model is loaded

        # Color mappings for visualization (from demo.py)
        self.colors = list(mcolors.TABLEAU_COLORS.values())

        # Default scaling mode
        self.scaling_mode = "fill"

        # Temporal sequence data structures
        self.temporal_frames = []  # Store last 5 frames
        self.temporal_entries = []  # Store last 5 entries
        self.temporal_preds = []  # Store last 5 predictions
        self.temporal_im_infos = []  # Store last 5 im_infos
        self.max_temporal_frames = 5
        self.temporal_carousel_index = 0  # Current position in carousel
        self.temporal_carousel_timer = None
        self.temporal_carousel_interval = 2000  # 2 seconds between carousel moves

    def setup_ui(self):
        layout = QVBoxLayout()

        # Create horizontal layout for 3 columns
        self.columns_layout = QHBoxLayout()

        # Create 3 video display labels
        self.video_labels = []
        self.info_labels = []

        for i in range(3):
            # Create container for each column
            column_widget = QWidget()
            column_layout = QVBoxLayout()

            # Video display label
            video_label = QLabel()
            video_label.setMinimumSize(320, 240)  # Smaller size for 3 columns
            video_label.setAlignment(Qt.AlignCenter)
            video_label.setStyleSheet("border: 2px solid gray;")
            column_layout.addWidget(video_label)

            # Info label
            info_label = QLabel(f"Column {i+1}: No video source")
            info_label.setAlignment(Qt.AlignCenter)
            column_layout.addWidget(info_label)

            column_widget.setLayout(column_layout)
            self.columns_layout.addWidget(column_widget)

            # Store references
            self.video_labels.append(video_label)
            self.info_labels.append(info_label)

        layout.addLayout(self.columns_layout)
        self.setLayout(layout)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def draw_bounding_boxes(self, image, boxes, labels, scores, relationships=None):
        img_copy = image.copy()
        h, w = img_copy.shape[:2]

        # Handle empty detections
        if len(boxes) == 0 or len(labels) == 0 or len(scores) == 0:
            return img_copy

        # Filter and sort by confidence, keep only top k
        top_k_bboxes = 10
        valid_detections = [
            (box, label, score, i)
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores))
            if score > 0.01
        ]
        valid_detections = sorted(valid_detections, key=lambda x: x[2], reverse=True)[
            :top_k_bboxes
        ]  # Top k only

        for box, label, score, original_idx in valid_detections:
            x1, y1, x2, y2 = box.astype(int)

            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Use consistent color for each object class (from demo.py)
            color_idx = (
                label % len(self.colors) if hasattr(self, "colors") else (label % 10)
            )
            if hasattr(self, "colors"):
                color = self.colors[color_idx]
                color_bgr = tuple(int(c * 255) for c in mcolors.to_rgb(color))
            else:
                colors = [
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (255, 255, 0),
                    (255, 0, 255),
                    (0, 255, 255),
                    (128, 0, 0),
                    (0, 128, 0),
                    (0, 0, 128),
                    (128, 128, 0),
                ]
                color_bgr = colors[color_idx]

            # Draw bounding box with thicker line
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color_bgr, 3)

            # Draw label with better visibility
            if (
                self.dataset
                and hasattr(self.dataset, "object_classes")
                and label < len(self.dataset.object_classes)
            ):
                label_text = f"{self.dataset.object_classes[label]} ({score:.2f})"
            else:
                label_text = f"obj_{label} ({score:.2f})"

            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )

            # Background for text
            cv2.rectangle(
                img_copy,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color_bgr,
                -1,
            )
            cv2.putText(
                img_copy,
                label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        return img_copy

    def update_frame(self, entry, pred, frame, column_index=0, im_info=None):
        """Update a specific column with frame data following demo.py approach"""
        try:
            # Validate inputs
            if frame is None or not isinstance(frame, np.ndarray):
                print("Warning: Invalid frame data")
                return

            if entry is None or not isinstance(entry, dict):
                print("Warning: Invalid entry data")
                return

            if pred is None or not isinstance(pred, dict):
                print("Warning: Invalid pred data")
                return

            # Ensure column index is valid
            if column_index >= len(self.video_labels):
                print(f"Warning: Invalid column index {column_index}")
                return

            # Get predictions following demo.py approach
            boxes = entry["boxes"][:, 1:].cpu().numpy()  # Remove batch dimension
            if self.dataset and hasattr(self.dataset, "object_classes"):
                # For predcls mode, use ground truth labels and scores
                if (
                    "labels" in entry
                    and isinstance(entry["labels"], torch.Tensor)
                    and entry["labels"].shape[0] > 0
                ):
                    labels = entry["labels"].cpu().numpy()
                    scores = (
                        entry.get("scores", torch.ones_like(entry["labels"]))
                        .cpu()
                        .numpy()
                    )
                else:
                    # For sgcls/sgdet modes, use predicted labels
                    labels = (
                        entry.get("pred_labels", torch.empty(0, dtype=torch.long))
                        .cpu()
                        .numpy()
                    )
                    scores = entry.get("pred_scores", torch.empty(0)).cpu().numpy()
            else:
                # Fallback for when dataset is not available
                labels = (
                    entry.get("pred_labels", torch.empty(0, dtype=torch.long))
                    .cpu()
                    .numpy()
                )
                scores = entry.get("pred_scores", torch.empty(0)).cpu().numpy()

            print("Number of boxes before NMS:", len(labels))

            # Apply NMS per class following demo.py approach
            if len(boxes) > 0 and len(labels) > 0:
                from torchvision.ops import nms

                final_indices = []
                for class_idx in np.unique(labels):
                    class_mask = labels == class_idx
                    class_boxes = boxes[class_mask]
                    class_scores = scores[class_mask]
                    if len(class_boxes) > 0:
                        boxes_tensor = torch.tensor(class_boxes, dtype=torch.float32)
                        scores_tensor = torch.tensor(class_scores, dtype=torch.float32)
                        keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.0)
                        class_indices = np.where(class_mask)[0][keep.cpu().numpy()]
                        final_indices.extend(class_indices)

                # Filter detections
                if final_indices:
                    boxes = boxes[final_indices]
                    labels = labels[final_indices]
                    scores = scores[final_indices]

            # Scale boxes to image size following demo.py approach
            if len(boxes) > 0:
                # Get the scale factor from im_info like demo.py does
                # The boxes need to be scaled by the im_info scale factor
                # This is the correct way to handle the coordinate transformation
                if im_info is not None:
                    # Use the passed im_info parameter
                    im_info_np = im_info.cpu().numpy()
                    scale = im_info_np[2]  # The scale factor
                    boxes = boxes * scale
                elif "im_info" in entry and entry["im_info"] is not None:
                    # Fallback to entry im_info
                    im_info_np = entry["im_info"][0].cpu().numpy()
                    scale = im_info_np[2]  # The scale factor
                    boxes = boxes * scale
                else:
                    # Fallback: check if boxes are normalized (0-1) or in pixel coordinates
                    if boxes.max() <= 1.0:
                        # Boxes are normalized, scale to image size
                        img_height, img_width = frame.shape[:2]
                        boxes[:, [0, 2]] *= img_width  # Scale x coordinates
                        boxes[:, [1, 3]] *= img_height  # Scale y coordinates

            # Draw bounding boxes on frame
            frame_with_boxes = self.draw_bounding_boxes(frame, boxes, labels, scores)

            # Convert frame to QPixmap
            height, width, channel = frame_with_boxes.shape
            bytes_per_line = 3 * width
            q_image = QImage(
                frame_with_boxes.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(q_image)

            # Get the target size for this column
            target_size = self.video_labels[column_index].size()

            # Scale based on user preference (default to fill entire space)
            scaling_mode = getattr(self, "scaling_mode", "fill")  # Default to 'fill'

            if scaling_mode == "fit":
                # Keep aspect ratio, fit within bounds
                scaled_pixmap = pixmap.scaled(
                    target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            elif scaling_mode == "fill":
                # Fill entire space, may crop
                scaled_pixmap = pixmap.scaled(
                    target_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )
            elif scaling_mode == "stretch":
                # Stretch to fill, may distort
                scaled_pixmap = pixmap.scaled(
                    target_size, Qt.IgnoreAspectRatio, Qt.FastTransformation
                )
            else:
                # Default to fill
                scaled_pixmap = pixmap.scaled(
                    target_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )

            # Set the pixmap
            self.video_labels[column_index].setPixmap(scaled_pixmap)

            # Update info for the specific column
            num_objects = len(boxes)
            self.info_labels[column_index].setText(
                f"Column {column_index+1}: {width}x{height} | Objects: {num_objects}"
            )

        except Exception as e:
            print(f"Error updating video display column {column_index}: {str(e)}")

    def set_scaling_mode(self, mode):
        """Set the scaling mode for video display"""
        self.scaling_mode = mode
        print(f"Video scaling mode set to: {mode}")

    def update_all_columns(self, entry, pred, frame, im_info=None):
        """Update all 3 columns with the same frame data"""
        # Validate frame before processing
        if frame is None:
            print("Warning: Frame is None in update_all_columns")
            return

        # Column 0: Video without bounding boxes (raw)
        self.update_frame_no_boxes(frame, 0)
        # Column 1: Video with bounding boxes
        self.update_frame(entry, pred, frame, 1, im_info)
        # Column 2: Video with bounding boxes (same as column 1)
        self.update_frame(entry, pred, frame, 2, im_info)

    def update_frame_no_boxes(self, frame, column_index=0):
        """Update a specific column with frame data without bounding boxes"""
        try:
            # Validate inputs
            if frame is None or not isinstance(frame, np.ndarray):
                print(
                    f"Warning: Invalid frame data for no-boxes display - frame: {type(frame)}"
                )
                return

            # Ensure column index is valid
            if column_index >= len(self.video_labels):
                print(f"Warning: Invalid column index {column_index}")
                return

            # Convert frame to QPixmap (no bounding boxes)
            height, width, channel = frame.shape

            # Ensure frame is contiguous in memory
            frame_contiguous = np.ascontiguousarray(frame)
            bytes_per_line = 3 * width
            q_image = QImage(
                frame_contiguous.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(q_image)

            # Get the target size for this column
            target_size = self.video_labels[column_index].size()

            # Scale based on user preference (default to fill entire space)
            scaling_mode = getattr(self, "scaling_mode", "fill")  # Default to 'fill'

            if scaling_mode == "fit":
                # Keep aspect ratio, fit within bounds
                scaled_pixmap = pixmap.scaled(
                    target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            elif scaling_mode == "fill":
                # Fill entire space, may crop
                scaled_pixmap = pixmap.scaled(
                    target_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )
            elif scaling_mode == "stretch":
                # Stretch to fill, may distort
                scaled_pixmap = pixmap.scaled(
                    target_size, Qt.IgnoreAspectRatio, Qt.FastTransformation
                )
            else:
                # Default to fill
                scaled_pixmap = pixmap.scaled(
                    target_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )

            # Set the pixmap
            self.video_labels[column_index].setPixmap(scaled_pixmap)

            # Update info for the specific column
            self.info_labels[column_index].setText(
                f"Column {column_index+1}: {width}x{height} | Raw Video"
            )

        except Exception as e:
            print(
                f"Error updating no-boxes video display column {column_index}: {str(e)}"
            )

    def update_column_with_different_data(
        self, column_index, entry, pred, frame, im_info=None
    ):
        """Update a specific column with different data"""
        self.update_frame(entry, pred, frame, column_index, im_info)

    def update_temporal_sequence(self, entry, pred, frame, im_info=None):
        try:
            # Add new frame data to temporal buffers
            # Handle different data types properly
            if frame is not None:
                if isinstance(frame, np.ndarray):
                    self.temporal_frames.append(frame.copy())
                else:
                    self.temporal_frames.append(frame)
            else:
                self.temporal_frames.append(None)

            if entry is not None:
                # Deep copy entry data, handling tensors properly
                entry_copy = {}
                for key, value in entry.items():
                    if isinstance(value, torch.Tensor):
                        # Move tensor to CPU and detach to prevent CUDA memory issues
                        entry_copy[key] = value.cpu().detach()
                    elif isinstance(value, dict):
                        entry_copy[key] = value.copy()
                    else:
                        entry_copy[key] = value
                self.temporal_entries.append(entry_copy)
            else:
                self.temporal_entries.append(None)

            if pred is not None:
                # Deep copy pred data, handling tensors properly
                pred_copy = {}
                for key, value in pred.items():
                    if isinstance(value, torch.Tensor):
                        # Move tensor to CPU and detach to prevent CUDA memory issues
                        pred_copy[key] = value.cpu().detach()
                    elif isinstance(value, dict):
                        pred_copy[key] = value.copy()
                    else:
                        pred_copy[key] = value
                self.temporal_preds.append(pred_copy)
            else:
                self.temporal_preds.append(None)

            if im_info is not None:
                self.temporal_im_infos.append(im_info.copy())
            else:
                self.temporal_im_infos.append(None)

            # Keep only the last max_temporal_frames to prevent memory issues
            while len(self.temporal_frames) > self.max_temporal_frames:
                self.temporal_frames.pop(0)
                self.temporal_entries.pop(0)
                self.temporal_preds.pop(0)
                self.temporal_im_infos.pop(0)

            # Update the temporal display
            self._update_temporal_display()

        except Exception as e:
            print(f"Error in temporal sequence update: {str(e)}")
            # Don't let temporal sequence errors crash the application

    def _update_temporal_display(self):
        """Update the temporal sequence display"""
        if len(self.temporal_frames) == 0:
            return

        # Show the 5 most recent frames in the top panel
        frames_to_show = (
            self.temporal_frames[-5:]
            if len(self.temporal_frames) >= 5
            else self.temporal_frames
        )
        entries_to_show = (
            self.temporal_entries[-len(frames_to_show) :]
            if len(self.temporal_entries) >= len(frames_to_show)
            else self.temporal_entries
        )
        preds_to_show = (
            self.temporal_preds[-len(frames_to_show) :]
            if len(self.temporal_preds) >= len(frames_to_show)
            else self.temporal_preds
        )
        im_infos_to_show = (
            self.temporal_im_infos[-len(frames_to_show) :]
            if len(self.temporal_im_infos) >= len(frames_to_show)
            else self.temporal_im_infos
        )

        # Update each column with a frame
        for i, (frame, entry, pred, im_info) in enumerate(
            zip(frames_to_show, entries_to_show, preds_to_show, im_infos_to_show)
        ):
            if i < len(self.video_labels):
                self.update_frame(entry, pred, frame, i, im_info)

        # Clear remaining columns if we have fewer than 5 frames
        for i in range(len(frames_to_show), len(self.video_labels)):
            self.video_labels[i].clear()
            self.info_labels[i].setText(f"Column {i+1}: No frame")

    def start_temporal_carousel(self):
        """Start the temporal carousel animation"""
        if self.temporal_carousel_timer is None:
            self.temporal_carousel_timer = QTimer()
            self.temporal_carousel_timer.timeout.connect(
                self._advance_temporal_carousel
            )
            self.temporal_carousel_timer.start(self.temporal_carousel_interval)

    def stop_temporal_carousel(self):
        """Stop the temporal carousel animation"""
        if self.temporal_carousel_timer is not None:
            self.temporal_carousel_timer.stop()
            self.temporal_carousel_timer = None

    def _advance_temporal_carousel(self):
        """Advance the temporal carousel to the next position"""
        if len(self.temporal_frames) >= 5:
            # For now, just update the display with the latest frames
            # In a more advanced implementation, we could show different sliding windows
            self._update_temporal_display()

    def set_temporal_carousel_interval(self, interval_ms):
        """Set the carousel interval in milliseconds"""
        self.temporal_carousel_interval = interval_ms
        if self.temporal_carousel_timer is not None:
            self.temporal_carousel_timer.setInterval(interval_ms)

    def clear_temporal_data(self):
        """Clear temporal data to free memory"""
        self.temporal_frames.clear()
        self.temporal_entries.clear()
        self.temporal_preds.clear()
        self.temporal_im_infos.clear()
        self.temporal_carousel_index = 0

    def hide_columns_2_and_3(self):
        """Hide columns 2 and 3 for temporal sequence mode"""
        if len(self.video_labels) >= 3:
            # Hide the video labels for columns 2 and 3
            self.video_labels[1].setVisible(False)
            self.video_labels[2].setVisible(False)
            # Hide the info labels for columns 2 and 3
            self.info_labels[1].setVisible(False)
            self.info_labels[2].setVisible(False)

    def show_all_columns(self):
        """Show all columns for normal modes"""
        if len(self.video_labels) >= 3:
            # Show all video labels
            self.video_labels[0].setVisible(True)
            self.video_labels[1].setVisible(True)
            self.video_labels[2].setVisible(True)
            # Show all info labels
            self.info_labels[0].setVisible(True)
            self.info_labels[1].setVisible(True)
            self.info_labels[2].setVisible(True)


class SGGGUI(QMainWindow):
    """
    Scene Graph Generation GUI

    Modes:
    - predcls: Uses dataset video with ground truth object labels (relationship prediction only)
    - sgcls: Uses dataset video with ground truth object labels (object classification + relationship prediction)
    - sgdet: Uses custom video/camera with full object detection + relationship prediction
    """

    def __init__(self):
        super().__init__()
        self.video_processor = None
        self.summarization_wrapper = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Scene Graph Generation GUI")
        self.setGeometry(100, 100, 1400, 900)

        # Set application icon
        icon_path = "assets/tum-logo.ico"
        if os.path.exists(icon_path):
            # Create icon with proper scaling
            icon = QIcon(icon_path)
            sizes = [16, 32, 48, 64, 128]
            for size in sizes:
                icon.addFile(icon_path, QSize(size, size))
            self.setWindowIcon(icon)
            print(f"Set window icon: {icon_path}")
        else:
            print(f"Icon file not found: {icon_path}")

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout()

        # Left panel - Controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(
            left_panel, 2
        )  # Increased from 1 to 2 for better visibility

        # Right panel - Display
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)

        central_widget.setLayout(main_layout)

        # Create status bar
        self.statusBar = self.statusBar()
        self.statusBar.showMessage(
            "Ready - Use Left/Right arrows for frame navigation, Space to pause, N to skip video"
        )

        # Initialize compatibility warning
        self.update_compatibility_warning()

        # Initialize model path display
        self.update_model_path_display()

        # Initialize video source controls
        self.update_video_source_controls()

        # Initialize video cycling controls
        self.update_video_cycling_controls()

        # Initialize column labels
        self.update_column_labels()

        # Initialize summarization model
        self.on_summarization_model_changed()

        # Auto-start processing for predcls mode
        self.auto_start_processing()

    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()

        # Model Selection Group
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        # Model type
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["sttran", "stket", "tempura"])  # "dsg-detr",
        model_layout.addWidget(QLabel("Model Type:"))
        model_layout.addWidget(self.model_type_combo)

        # Dataset type
        self.dataset_type_combo = QComboBox()
        self.dataset_type_combo.addItems(["action_genome", "EASG"])
        model_layout.addWidget(QLabel("Dataset:"))
        model_layout.addWidget(self.dataset_type_combo)

        # Compatibility warning
        self.compatibility_label = QLabel("")
        self.compatibility_label.setStyleSheet("color: orange; font-weight: bold;")
        model_layout.addWidget(self.compatibility_label)

        # Connect signals to update compatibility warning
        self.model_type_combo.currentTextChanged.connect(
            self.update_compatibility_warning
        )
        self.dataset_type_combo.currentTextChanged.connect(
            self.update_compatibility_warning
        )

        # Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["predcls", "sgcls", "sgdet"])
        model_layout.addWidget(QLabel("Mode:"))
        model_layout.addWidget(self.mode_combo)

        # Connect signals to update model path display (after mode_combo is created)
        self.model_type_combo.currentTextChanged.connect(self.update_model_path_display)
        self.dataset_type_combo.currentTextChanged.connect(
            self.update_model_path_display
        )
        self.mode_combo.currentTextChanged.connect(self.update_model_path_display)

        # Connect mode changes to update video source controls
        self.mode_combo.currentTextChanged.connect(self.update_video_source_controls)

        # Connect mode changes to auto-start processing for predcls mode
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)

        # Connect mode changes to update video cycling controls
        self.mode_combo.currentTextChanged.connect(self.update_video_cycling_controls)

        # Model path
        self.model_path_edit = QLabel("Checking for checkpoints...")
        model_layout.addWidget(QLabel("Model Path:"))
        model_layout.addWidget(self.model_path_edit)

        # Refresh button for checkpoints
        self.refresh_checkpoint_button = QPushButton("Refresh Checkpoints")
        self.refresh_checkpoint_button.clicked.connect(self.update_model_path_display)
        model_layout.addWidget(self.refresh_checkpoint_button)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # View Mode Group
        view_group = QGroupBox("View Mode")
        view_layout = QVBoxLayout()

        self.multiple_view_check = QCheckBox("Multiple View")
        self.egocentric_check = QCheckBox("Egocentric")
        self.multiple_view_check.setChecked(True)

        # Connect multiple view checkbox to update display mode
        self.multiple_view_check.toggled.connect(self.on_multiple_view_toggled)

        view_layout.addWidget(self.multiple_view_check)
        view_layout.addWidget(self.egocentric_check)

        # Video scaling mode
        scaling_layout = QHBoxLayout()
        scaling_layout.addWidget(QLabel("Video Scaling:"))
        self.scaling_mode_combo = QComboBox()
        self.scaling_mode_combo.addItems(["Fill", "Fit", "Stretch"])
        self.scaling_mode_combo.setCurrentText("Fill")
        self.scaling_mode_combo.currentTextChanged.connect(self.on_scaling_mode_changed)
        scaling_layout.addWidget(self.scaling_mode_combo)
        view_layout.addLayout(scaling_layout)

        # Column display controls
        self.column_controls_group = QGroupBox("Column Display")
        column_controls_layout = QVBoxLayout()

        # Column mode selection
        self.column_mode_combo = QComboBox()
        self.column_mode_combo.addItems(["Column Mode", "Temporal Sequence"])
        self.column_mode_combo.setCurrentText("Column Mode")
        column_controls_layout.addWidget(QLabel("Column Mode:"))
        column_controls_layout.addWidget(self.column_mode_combo)

        # Connect column mode changes
        self.column_mode_combo.currentTextChanged.connect(self.on_column_mode_changed)

        # Column labels
        self.column_labels = []
        for i in range(3):
            label_edit = QLabel(f"Column {i+1}: Main View")
            label_edit.setStyleSheet("font-size: 10px; color: gray;")
            self.column_labels.append(label_edit)
            column_controls_layout.addWidget(label_edit)

        self.column_controls_group.setLayout(column_controls_layout)
        view_layout.addWidget(self.column_controls_group)

        # Temporal carousel controls
        self.temporal_carousel_group = QGroupBox("Temporal Carousel")
        temporal_carousel_layout = QVBoxLayout()

        # Carousel speed slider
        self.carousel_speed_slider = QSlider(Qt.Horizontal)
        self.carousel_speed_slider.setRange(500, 5000)  # 0.5 to 5 seconds
        self.carousel_speed_slider.setValue(2000)  # Default 2 seconds
        self.carousel_speed_slider.setEnabled(False)  # Only enabled in temporal mode

        temporal_carousel_layout.addWidget(QLabel("Carousel Speed (ms):"))
        temporal_carousel_layout.addWidget(self.carousel_speed_slider)

        # Connect carousel speed slider
        self.carousel_speed_slider.valueChanged.connect(self.on_carousel_speed_changed)

        self.temporal_carousel_group.setLayout(temporal_carousel_layout)
        view_layout.addWidget(self.temporal_carousel_group)

        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # Video Source Group
        video_group = QGroupBox("Video Source")
        video_layout = QVBoxLayout()

        self.camera_radio = QCheckBox("Camera")
        self.file_radio = QCheckBox("Video File")
        self.file_radio.setChecked(True)

        self.camera_id_spin = QSpinBox()
        self.camera_id_spin.setRange(0, 10)
        self.camera_id_spin.setValue(0)

        self.file_path_label = QLabel("No file selected")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_video_file)

        # Dataset video info for predcls/sgcls modes
        self.dataset_video_label = QLabel("Using dataset video (random selection)")
        self.dataset_video_label.setStyleSheet("color: blue; font-weight: bold;")
        self.dataset_video_label.setVisible(False)

        video_layout.addWidget(self.camera_radio)
        video_layout.addWidget(QLabel("Camera ID:"))
        video_layout.addWidget(self.camera_id_spin)
        video_layout.addWidget(self.file_radio)
        video_layout.addWidget(self.file_path_label)
        video_layout.addWidget(self.browse_button)
        video_layout.addWidget(self.dataset_video_label)

        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # Control Group
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()

        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        control_layout.addWidget(self.start_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        control_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        # FPS slider
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(
            1, 60
        )  # Increased max FPS to 60 for more meaningful control
        self.fps_slider.setValue(
            5
        )  # Default to 5 FPS - reduced for better demonstration
        control_layout.addWidget(QLabel("FPS:"))
        control_layout.addWidget(self.fps_slider)

        # Connect FPS slider to video processor
        self.fps_slider.valueChanged.connect(self.on_fps_changed)

        # Video cycling controls (for predcls/sgcls modes)
        self.video_cycling_group = QGroupBox("Video Cycling")
        video_cycling_layout = QVBoxLayout()

        # Frames per video slider
        self.frames_per_video_slider = QSlider(Qt.Horizontal)
        self.frames_per_video_slider.setRange(10, 100)
        self.frames_per_video_slider.setValue(30)
        video_cycling_layout.addWidget(QLabel("Frames per Video:"))
        video_cycling_layout.addWidget(self.frames_per_video_slider)

        # Connect frames per video slider
        self.frames_per_video_slider.valueChanged.connect(
            self.on_frames_per_video_changed
        )

        # Video cycling info label
        self.video_cycling_info = QLabel("Video cycling: Disabled")
        self.video_cycling_info.setStyleSheet("color: gray; font-size: 10px;")
        video_cycling_layout.addWidget(self.video_cycling_info)

        # Video navigation controls
        video_nav_layout = QHBoxLayout()

        # Skip to next video button
        self.skip_video_button = QPushButton("Skip to Next Video")
        self.skip_video_button.setToolTip(
            "Skip to the next video in the cycling sequence"
        )
        self.skip_video_button.clicked.connect(self.skip_to_next_video)
        video_nav_layout.addWidget(self.skip_video_button)

        # Frame navigation controls
        frame_nav_layout = QHBoxLayout()

        # Previous frame button
        self.prev_frame_button = QPushButton("⏮ Previous Frame")
        self.prev_frame_button.setToolTip("Go to the previous frame (Left Arrow)")
        self.prev_frame_button.clicked.connect(self.previous_frame)
        frame_nav_layout.addWidget(self.prev_frame_button)

        # Next frame button
        self.next_frame_button = QPushButton("Next Frame ⏭")
        self.next_frame_button.setToolTip("Go to the next frame (Right Arrow)")
        self.next_frame_button.clicked.connect(self.next_frame)
        frame_nav_layout.addWidget(self.next_frame_button)

        # Frame info label
        self.frame_info_label = QLabel("Frame: 0/0")
        self.frame_info_label.setStyleSheet("color: blue; font-weight: bold;")
        frame_nav_layout.addWidget(self.frame_info_label)

        video_cycling_layout.addLayout(video_nav_layout)
        video_cycling_layout.addLayout(frame_nav_layout)

        self.video_cycling_group.setLayout(video_cycling_layout)
        control_layout.addWidget(self.video_cycling_group)

        # Video Selection Group
        video_selection_group = QGroupBox("Video Selection")
        video_selection_layout = QVBoxLayout()

        # Expanded video selection checkbox
        self.expanded_video_selection_check = QCheckBox("Use Expanded Video Selection")
        self.expanded_video_selection_check.setChecked(True)
        self.expanded_video_selection_check.setToolTip(
            "Include videos without ground truth bounding boxes in the random sample set"
        )
        video_selection_layout.addWidget(self.expanded_video_selection_check)

        # GT video selection probability slider
        gt_prob_layout = QHBoxLayout()
        gt_prob_layout.addWidget(QLabel("GT Video Probability:"))
        self.gt_prob_slider = QSlider(Qt.Horizontal)
        self.gt_prob_slider.setRange(10, 90)  # 10% to 90%
        self.gt_prob_slider.setValue(70)  # Default 70%
        self.gt_prob_slider.setEnabled(True)
        gt_prob_layout.addWidget(self.gt_prob_slider)

        # GT probability label
        self.gt_prob_label = QLabel("70%")
        gt_prob_layout.addWidget(self.gt_prob_label)

        video_selection_layout.addLayout(gt_prob_layout)

        # Connect video selection controls
        self.expanded_video_selection_check.toggled.connect(
            self.on_expanded_video_selection_toggled
        )
        self.gt_prob_slider.valueChanged.connect(self.on_gt_prob_changed)

        video_selection_group.setLayout(video_selection_layout)
        control_layout.addWidget(video_selection_group)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Summarization Model Group
        summarization_group = QGroupBox("Summarization Model")
        summarization_layout = QVBoxLayout()

        # Summarization model selection
        self.summarization_model_combo = QComboBox()
        self.summarization_model_combo.addItems(
            [
                "T5 Base",
                "T5 Large",
                "Pegasus XSum",
                "Pegasus CNN/DailyMail",
                "Pegasus Newsroom",
                "Pegasus Multi-News",
            ]
        )
        self.summarization_model_combo.setCurrentText("T5 Base")
        summarization_layout.addWidget(QLabel("Model:"))
        summarization_layout.addWidget(self.summarization_model_combo)

        # Summarization parameters
        self.summary_length_spin = QSpinBox()
        self.summary_length_spin.setRange(50, 500)
        self.summary_length_spin.setValue(150)
        self.summary_length_spin.setSuffix(" chars")
        summarization_layout.addWidget(QLabel("Max Length:"))
        summarization_layout.addWidget(self.summary_length_spin)

        # Auto-summarize checkbox
        self.auto_summarize_check = QCheckBox("Auto Summarize")
        self.auto_summarize_check.setChecked(True)
        summarization_layout.addWidget(self.auto_summarize_check)

        # Connect summarization model changes
        self.summarization_model_combo.currentTextChanged.connect(
            self.on_summarization_model_changed
        )
        self.summary_length_spin.valueChanged.connect(
            self.on_summary_parameters_changed
        )
        self.auto_summarize_check.toggled.connect(self.on_summary_parameters_changed)

        summarization_group.setLayout(summarization_layout)
        layout.addWidget(summarization_group)

        # Status Group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()

        # Create splitter for vertical layout
        splitter = QSplitter(Qt.Vertical)

        # Video display
        self.video_display = VideoDisplay()
        splitter.addWidget(self.video_display)

        # Scene graph visualization
        self.scene_graph_viz = SceneGraphVisualizer()
        splitter.addWidget(self.scene_graph_viz)

        # Temporal graph visualization
        self.temporal_graph_viz = TemporalGraphVisualizer()
        splitter.addWidget(self.temporal_graph_viz)

        # Set initial visibility - temporal graph hidden by default
        self.temporal_graph_viz.setVisible(False)

        # Text summarization
        self.text_summarizer = TextSummarizer()
        splitter.addWidget(self.text_summarizer)

        # Set splitter proportions
        splitter.setSizes([400, 300, 400, 200])

        layout.addWidget(splitter)
        panel.setLayout(layout)
        return panel

    def browse_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.file_path_label.setText(file_path)

    def start_processing(self):
        try:
            # Get current settings
            model_type = self.model_type_combo.currentText()
            dataset_type = self.dataset_type_combo.currentText()
            mode = self.mode_combo.currentText()

            # Find the latest checkpoint
            checkpoint_path = find_latest_checkpoint(dataset_type, model_type, mode)
            if checkpoint_path is None:
                error_msg = f"No checkpoint found for:\nDataset: {dataset_type}\nModel: {model_type}\nMode: {mode}\n\nExpected path: output/{dataset_type}/{model_type}/{mode}/[timestamp]/model_best.tar"
                QMessageBox.critical(self, "Checkpoint Not Found", error_msg)
                return

            # Create video processor
            self.video_processor = VideoProcessor(
                None, model_type, dataset_type, mode, checkpoint_path
            )

            # Set frames per video from GUI
            self.video_processor.frames_per_video = self.frames_per_video_slider.value()

            # Set video selection parameters from GUI
            if hasattr(self, "expanded_video_selection_check"):
                self.video_processor.use_expanded_video_selection = (
                    self.expanded_video_selection_check.isChecked()
                )
            if hasattr(self, "gt_prob_slider"):
                self.video_processor.gt_video_selection_probability = (
                    self.gt_prob_slider.value() / 100.0
                )

            # Setup models first
            if not self.video_processor.setup_models():
                QMessageBox.critical(self, "Error", "Failed to setup models")
                return

            # Setup video source based on mode
            if mode in ["predcls", "sgcls"]:
                # Use dataset video for predcls/sgcls modes
                if not self.video_processor.setup_dataset_video():
                    QMessageBox.warning(self, "Error", "Failed to setup dataset video")
                    return
            else:
                # Use camera/video file for sgdet mode
                if self.camera_radio.isChecked():
                    camera_id = self.camera_id_spin.value()
                    # Test camera access
                    test_cap = cv2.VideoCapture(camera_id)
                    if not test_cap.isOpened():
                        QMessageBox.warning(
                            self, "Error", f"Failed to open camera {camera_id}"
                        )
                        return
                    test_cap.release()

                    if not self.video_processor.setup_camera(camera_id):
                        return
                else:
                    video_path = self.file_path_label.text()
                    if video_path == "No file selected":
                        QMessageBox.warning(self, "Error", "Please select a video file")
                        return
                    if not os.path.exists(video_path):
                        QMessageBox.warning(
                            self, "Error", f"Video file not found: {video_path}"
                        )
                        return

                    if not self.video_processor.setup_video_file(video_path):
                        return

            # Connect signals
            self.video_processor.frame_processed.connect(self.on_frame_processed)
            self.video_processor.error_occurred.connect(self.on_error)
            self.video_processor.video_info_updated.connect(self.on_video_info_updated)
            self.video_processor.frame_info_updated.connect(self.on_frame_info_updated)

            # Start processing
            self.video_processor.start()

            # Update UI
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Processing...")

            # Update status bar
            if hasattr(self, "statusBar"):
                self.statusBar.showMessage(
                    "Processing - Use Left/Right arrows for frame navigation, Space to pause, N to skip video"
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start processing: {str(e)}")
            import traceback

            traceback.print_exc()

    def toggle_pause(self):
        if self.video_processor:
            if self.video_processor.paused:
                # Resume
                self.video_processor.resume()
                self.pause_button.setText("Pause")
                self.status_label.setText("Processing...")
            else:
                # Pause
                self.video_processor.pause()
                self.pause_button.setText("Resume")
                self.status_label.setText("Paused")

    def stop_processing(self):
        if self.video_processor:
            self.video_processor.stop()
            self.video_processor.wait()
            self.video_processor = None

        # Clear frame counters
        if hasattr(self, "_frame_count"):
            del self._frame_count
        if hasattr(self, "_last_status_update"):
            del self._last_status_update

        # Update UI
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.stop_button.setEnabled(False)
        self.status_label.setText("Ready")

        # Update status bar
        if hasattr(self, "statusBar"):
            self.statusBar.showMessage(
                "Ready - Use Left/Right arrows for frame navigation, Space to pause, N to skip video"
            )

    def on_mode_changed(self):
        """Handle mode changes - auto-start for predcls mode"""
        mode = self.mode_combo.currentText()

        # Stop current processing if any
        if self.video_processor:
            self.stop_processing()

        # Auto-start for predcls mode
        if mode == "predcls":
            # Use a timer to delay the auto-start slightly to allow UI to update
            QTimer.singleShot(100, self.auto_start_processing)

    def on_video_info_updated(self, video_info):
        """Handle video info updates from the processor"""
        self.dataset_video_label.setText(video_info)

    def on_multiple_view_toggled(self, checked):
        """Handle multiple view checkbox toggle"""
        if checked:
            # Enable column controls
            self.column_controls_group.setEnabled(True)
            # Update display to show 3 columns
            self.update_column_labels()
        else:
            # Disable column controls
            self.column_controls_group.setEnabled(False)
            # Update display to show single column (only first column active)
            self.update_column_labels()

    def on_column_mode_changed(self, mode):
        self.update_column_labels()

        # Enable/disable carousel controls based on mode
        if mode == "Temporal Sequence":
            self.carousel_speed_slider.setEnabled(True)
            # Hide scene graph and show temporal graph in temporal mode
            self.scene_graph_viz.setVisible(False)
            self.temporal_graph_viz.setVisible(True)
            # Hide columns 2 and 3 in temporal sequence mode
            if hasattr(self, "video_display"):
                self.video_display.hide_columns_2_and_3()
                self.video_display.stop_temporal_carousel()
        else:
            self.carousel_speed_slider.setEnabled(False)
            # Show scene graph and hide temporal graph in other modes
            self.scene_graph_viz.setVisible(True)
            self.temporal_graph_viz.setVisible(False)
            # Show all columns in other modes
            if hasattr(self, "video_display"):
                self.video_display.show_all_columns()
                self.video_display.stop_temporal_carousel()
                self.video_display.clear_temporal_data()
                self.temporal_graph_viz.clear_temporal_data()

    def update_column_labels(self):
        """Update column labels based on current mode and settings"""
        if not self.multiple_view_check.isChecked():
            # Single view mode - only first column active
            for i, label in enumerate(self.column_labels):
                if i == 0:
                    label.setText("Column 1: Raw Video")
                else:
                    label.setText(f"Column {i+1}: Disabled")
        else:
            mode = self.column_mode_combo.currentText()
            if mode == "Column Mode":
                # Update column labels to reflect the new functionality
                self.column_labels[0].setText("Column 1: Raw Video")
                self.column_labels[1].setText("Column 2: With BBoxes")
                self.column_labels[2].setText("Column 3: With BBoxes")
            elif mode == "Temporal Sequence":
                # In temporal sequence mode, show only one video with temporal data
                self.column_labels[0].setText("Column 1: Temporal Sequence")
                self.column_labels[1].setText("Column 2: Disabled")
                self.column_labels[2].setText("Column 3: Disabled")

    def on_frame_processed(self, entry, pred, frame, im_info=None):
        """Handle frame processing with multiple column support"""
        try:
            # Get dataset from video processor for visualizers
            if self.video_processor and hasattr(self.video_processor, "dataset"):
                dataset = self.video_processor.dataset
                self.scene_graph_viz.set_dataset(dataset)
                self.temporal_graph_viz.set_dataset(dataset)
                self.video_display.set_dataset(dataset)

            # Update video display based on multiple view setting
            if self.multiple_view_check.isChecked():
                mode = self.column_mode_combo.currentText()
                if mode == "Column Mode":
                    # Show same frame in all columns
                    # Column 0: Raw video, Column 1&2: With bounding boxes
                    self.video_display.update_all_columns(entry, pred, frame, im_info)
                elif mode == "Temporal Sequence":
                    try:
                        # Update temporal sequence with new frame data
                        self.video_display.update_temporal_sequence(
                            entry, pred, frame, im_info
                        )

                        # Show only the current frame in the first column for temporal mode
                        # Columns 2 and 3 are hidden, so only update column 0
                        self.video_display.update_frame_no_boxes(frame, 0)

                        # Update temporal graph visualization less frequently to prevent halting
                        if (
                            len(self.video_display.temporal_frames) >= 2
                            and self._frame_count % 5 == 0
                        ):
                            try:
                                # Add a small delay to prevent overwhelming the system
                                import time

                                time.sleep(0.01)

                                self.temporal_graph_viz.update_temporal_graph(
                                    self.video_display.temporal_entries,
                                    self.video_display.temporal_preds,
                                    self.video_display.temporal_frames,
                                )
                            except Exception as e:
                                print(
                                    f"Error updating temporal graph visualization: {str(e)}"
                                )
                                # Don't let temporal graph errors crash the application

                        # Start carousel if we have enough frames
                        if len(self.video_display.temporal_frames) >= 5:
                            self.video_display.start_temporal_carousel()
                    except Exception as e:
                        print(f"Error in temporal sequence processing: {str(e)}")
                        # Reset temporal data if there are too many errors to prevent halting
                        if hasattr(self, "_temporal_error_count"):
                            self._temporal_error_count += 1
                        else:
                            self._temporal_error_count = 1

                        if self._temporal_error_count > 3:
                            print(
                                "Too many temporal sequence errors, clearing temporal data"
                            )
                            self.video_display.clear_temporal_data()
                            self.temporal_graph_viz.clear_temporal_data()
                            self._temporal_error_count = 0
                        # Don't let temporal sequence errors crash the application
            else:
                # Single view mode - show raw video in first column
                self.video_display.update_frame_no_boxes(frame, 0)

            # Update scene graph visualization (less frequently to improve performance)
            if hasattr(self, "_frame_count"):
                self._frame_count += 1
            else:
                self._frame_count = 1

            # Only update scene graph every 3 frames to reduce computational load
            # Skip scene graph updates in temporal sequence mode to prevent halting
            mode = self.column_mode_combo.currentText()
            if self._frame_count % 3 == 0 and mode != "Temporal Sequence":
                self.scene_graph_viz.update_scene_graph(entry, pred, frame)

            # Update text summarization (less frequently)
            if self._frame_count % 5 == 0:
                self.text_summarizer.update_summary(entry, pred, frame)

            # Update status with FPS info
            if hasattr(self, "_last_status_update"):
                current_time = time.time()
                if current_time - self._last_status_update > 1.0:  # Update every second
                    fps = self.fps_slider.value()
                    self.status_label.setText(f"Processing... (Target FPS: {fps})")
                    self._last_status_update = current_time
            else:
                self._last_status_update = time.time()

        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            # Don't let frame processing errors crash the application

    def update_compatibility_warning(self):
        model_type = self.model_type_combo.currentText()
        dataset_type = self.dataset_type_combo.currentText()

        compatibility = {
            "sttran": {
                "action_genome": "✅ Fully supported",
                "EASG": "✅ Fully supported",
            },
            "stket": {
                "action_genome": "⚠️  May have embedding size issues",
                "EASG": "❌ Not supported",
            },
            "tempura": {
                "action_genome": "✅ Fully supported",
                "EASG": "❌ Not supported",
            },
        }

        if model_type in compatibility and dataset_type in compatibility[model_type]:
            status = compatibility[model_type][dataset_type]
            self.compatibility_label.setText(f"Compatibility: {status}")

            if "❌" in status:
                self.compatibility_label.setStyleSheet("color: red; font-weight: bold;")
            elif "⚠️" in status:
                self.compatibility_label.setStyleSheet(
                    "color: orange; font-weight: bold;"
                )
            else:
                self.compatibility_label.setStyleSheet(
                    "color: green; font-weight: bold;"
                )
        else:
            self.compatibility_label.setText("Compatibility: Unknown")
            self.compatibility_label.setStyleSheet("color: gray; font-weight: bold;")

    def update_model_path_display(self):
        model_type = self.model_type_combo.currentText()
        dataset_type = self.dataset_type_combo.currentText()
        mode = self.mode_combo.currentText()
        checkpoint_path = find_latest_checkpoint(dataset_type, model_type, mode)

        if checkpoint_path:
            self.model_path_edit.setText(checkpoint_path)
        else:
            self.model_path_edit.setText(
                "No checkpoints found for this model/dataset/mode."
            )

    def update_video_source_controls(self):
        """Update video source controls based on selected mode"""
        mode = self.mode_combo.currentText()

        if mode in ["predcls", "sgcls"]:
            # Disable video file/camera controls for predcls/sgcls modes
            self.camera_radio.setEnabled(False)
            self.file_radio.setEnabled(False)
            self.camera_id_spin.setEnabled(False)
            self.file_path_label.setEnabled(False)
            self.browse_button.setEnabled(False)

            # Show dataset video info
            self.dataset_video_label.setVisible(True)

            # Update label text based on mode
            if mode == "predcls":
                self.dataset_video_label.setText(
                    "Using dataset video (predcls mode - ground truth objects)"
                )
            else:
                self.dataset_video_label.setText(
                    "Using dataset video (sgcls mode - ground truth objects)"
                )

        else:
            # Enable video file/camera controls for sgdet mode
            self.camera_radio.setEnabled(True)
            self.file_radio.setEnabled(True)
            self.camera_id_spin.setEnabled(True)
            self.file_path_label.setEnabled(True)
            self.browse_button.setEnabled(True)

            # Hide dataset video info
            self.dataset_video_label.setVisible(False)

    def update_video_cycling_controls(self):
        """Update video cycling controls based on selected mode"""
        mode = self.mode_combo.currentText()

        if mode in ["predcls", "sgcls"]:
            # Enable video cycling controls for predcls/sgcls modes
            self.video_cycling_group.setEnabled(True)
            self.frames_per_video_slider.setEnabled(True)
            self.skip_video_button.setEnabled(True)
            self.prev_frame_button.setEnabled(True)
            self.next_frame_button.setEnabled(True)
            self.frame_info_label.setVisible(True)
            self.video_cycling_info.setText(
                "Video cycling: Enabled - Will cycle through different videos"
            )
        else:
            # Disable video cycling controls for sgdet mode
            self.video_cycling_group.setEnabled(False)
            self.frames_per_video_slider.setEnabled(False)
            self.skip_video_button.setEnabled(False)
            self.prev_frame_button.setEnabled(False)
            self.next_frame_button.setEnabled(False)
            self.frame_info_label.setVisible(False)
            self.video_cycling_info.setText(
                "Video cycling: Disabled (only for predcls/sgcls modes)"
            )

    def auto_start_processing(self):
        """Auto-start processing for predcls mode with dataset video"""
        try:
            # Only auto-start if we're in predcls mode
            mode = self.mode_combo.currentText()
            if mode != "predcls":
                return

            # Check if we have a valid checkpoint
            model_type = self.model_type_combo.currentText()
            dataset_type = self.dataset_type_combo.currentText()
            checkpoint_path = find_latest_checkpoint(dataset_type, model_type, mode)

            if checkpoint_path is None:
                print("No checkpoint found for auto-start")
                return

            print("Auto-starting processing for predcls mode...")

            # Create video processor
            self.video_processor = VideoProcessor(
                None, model_type, dataset_type, mode, checkpoint_path
            )

            # Set frames per video from GUI
            self.video_processor.frames_per_video = self.frames_per_video_slider.value()

            # Setup models first to ensure dataset is loaded
            if not self.video_processor.setup_models():
                print("Failed to setup models for auto-start")
                return

            # Now setup dataset video with a random selection
            if not self.video_processor.setup_dataset_video():
                print("Failed to setup dataset video for auto-start")
                return

            # Connect signals
            self.video_processor.frame_processed.connect(self.on_frame_processed)
            self.video_processor.error_occurred.connect(self.on_error)
            self.video_processor.frame_info_updated.connect(self.on_frame_info_updated)

            # Start processing
            self.video_processor.start()

            # Update UI
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Auto-processing dataset video...")

        except Exception as e:
            print(f"Auto-start failed: {str(e)}")
            import traceback

            traceback.print_exc()

    # FIX: Fix real-time
    def on_summarization_model_changed(self):
        """Handle summarization model selection changes"""
        try:
            model_name = self.summarization_model_combo.currentText()
            self.summarization_wrapper = None

            # Create new wrapper based on selection
            # if model_name.startswith("T5"):
            #     if "Large" in model_name:
            #         self.summarization_wrapper = T5SummarizationWrapper("google-t5/t5-large")
            #     else:
            #         self.summarization_wrapper = T5SummarizationWrapper("google-t5/t5-base")
            # elif model_name.startswith("Pegasus"):
            #     if "CNN/DailyMail" in model_name:
            #         self.summarization_wrapper = PegasusSummarizationWrapper("google/pegasus-cnn_dailymail")
            #     elif "Newsroom" in model_name:
            #         self.summarization_wrapper = PegasusSummarizationWrapper("google/pegasus-newsroom")
            #     elif "Multi-News" in model_name:
            #         self.summarization_wrapper = PegasusSummarizationWrapper("google/pegasus-multi_news")
            #     else:
            #         # Default to XSum
            #         self.summarization_wrapper = PegasusSummarizationWrapper("google/pegasus-xsum")

            print(f"Loaded summarization model: {model_name}")

            # Update text summarizer with new wrapper
            if hasattr(self, "text_summarizer"):
                self.text_summarizer.set_summarization_wrapper(
                    self.summarization_wrapper
                )

        except Exception as e:
            print(f"Error loading summarization model: {str(e)}")
            QMessageBox.warning(
                self, "Error", f"Failed to load summarization model: {str(e)}"
            )

    def on_summary_parameters_changed(self):
        """Handle summarization parameter changes"""
        try:
            if hasattr(self, "text_summarizer"):
                # Update parameters in text summarizer
                max_length = self.summary_length_spin.value()
                auto_summarize = self.auto_summarize_check.isChecked()

                self.text_summarizer.update_parameters(max_length, auto_summarize)

        except Exception as e:
            print(f"Error updating summarization parameters: {str(e)}")

    def on_fps_changed(self, fps_value):
        """Handle FPS slider changes"""
        if self.video_processor:
            self.video_processor.set_fps(fps_value)

    def on_frames_per_video_changed(self, frames_value):
        """Handle frames per video slider changes"""
        if self.video_processor:
            self.video_processor.frames_per_video = frames_value
            print(f"Updated frames per video: {frames_value}")

    def on_carousel_speed_changed(self, speed_value):
        """Handle carousel speed slider changes"""
        if hasattr(self, "video_display"):
            self.video_display.set_temporal_carousel_interval(speed_value)

    def on_error(self, error_msg):
        # Prevent infinite error dialogs by checking if we're already showing an error
        if not hasattr(self, "_showing_error"):
            self._showing_error = True
            QMessageBox.warning(self, "Error", error_msg)
            self._showing_error = False
            self.stop_processing()

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for navigation"""
        if event.key() == Qt.Key_Left:
            # Left arrow - previous frame
            if hasattr(self, "video_processor") and self.video_processor:
                self.previous_frame()
        elif event.key() == Qt.Key_Right:
            # Right arrow - next frame
            if hasattr(self, "video_processor") and self.video_processor:
                self.next_frame()
        elif event.key() == Qt.Key_Space:
            # Space - toggle pause/resume
            if hasattr(self, "video_processor") and self.video_processor:
                self.toggle_pause()
        elif event.key() == Qt.Key_N:
            # N key - skip to next video
            if hasattr(self, "video_processor") and self.video_processor:
                self.skip_to_next_video()
        else:
            # Call parent keyPressEvent for other keys
            super().keyPressEvent(event)

    def select_specific_video(self, video_index):
        """Manually select a specific video for testing"""
        if self.video_processor and hasattr(self.video_processor, "dataset"):
            if video_index < len(self.video_processor.dataset.video_list):
                print(f"Manually selecting video {video_index}")
                self.video_processor.setup_dataset_video(video_index)
                return True
            else:
                print(
                    f"Video index {video_index} is out of range (max: {len(self.video_processor.dataset.video_list)-1})"
                )
                return False
        else:
            print("No video processor available")
            return False

    def get_video_cycling_info(self):
        """Get information about current video cycling status"""
        if self.video_processor and hasattr(
            self.video_processor, "available_video_indices"
        ):
            info = {
                "total_videos": len(self.video_processor.available_video_indices),
                "current_video": self.video_processor.current_video_index,
                "current_cycle": self.video_processor.video_cycle_index + 1,
                "frames_per_video": self.video_processor.frames_per_video,
                "frames_in_current_video": self.video_processor.frame_count_in_current_video,
                "available_videos": self.video_processor.available_video_indices,
            }
            return info
        else:
            return None

    def test_video_selection(self):
        """Test method to manually select a video - can be called from console"""
        try:
            # Get current settings
            model_type = self.model_type_combo.currentText()
            dataset_type = self.dataset_type_combo.currentText()
            mode = self.mode_combo.currentText()

            # Find the latest checkpoint
            checkpoint_path = find_latest_checkpoint(dataset_type, model_type, mode)
            if checkpoint_path is None:
                print("No checkpoint found for testing")
                return False

            # Create video processor
            self.video_processor = VideoProcessor(
                None, model_type, dataset_type, mode, checkpoint_path
            )

            # Setup models
            if not self.video_processor.setup_models():
                print("Failed to setup models for testing")
                return False

            # Test video selection
            if self.video_processor.setup_dataset_video():
                print("Video selection test successful!")
                return True
            else:
                print("Video selection test failed!")
                return False

        except Exception as e:
            print(f"Test failed: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def on_scaling_mode_changed(self, mode):
        """Handle scaling mode changes"""
        scaling_mode_map = {"Fill": "fill", "Fit": "fit", "Stretch": "stretch"}

        if hasattr(self, "video_display"):
            self.video_display.set_scaling_mode(scaling_mode_map.get(mode, "fill"))
            print(f"Video scaling mode changed to: {mode}")

    def on_expanded_video_selection_toggled(self, checked):
        """Handle expanded video selection checkbox changes"""
        if hasattr(self, "video_processor") and self.video_processor:
            self.video_processor.use_expanded_video_selection = checked
            print(f"Expanded video selection {'enabled' if checked else 'disabled'}")

            # Re-initialize video cycling with new settings
            if hasattr(self.video_processor, "_initialize_video_cycling"):
                self.video_processor._initialize_video_cycling()

    def on_gt_prob_changed(self, value):
        """Handle GT video selection probability slider changes"""
        if hasattr(self, "video_processor") and self.video_processor:
            self.video_processor.gt_video_selection_probability = value / 100.0
            print(f"GT video selection probability changed to: {value}%")

            # Update the label
            if hasattr(self, "gt_prob_label"):
                self.gt_prob_label.setText(f"{value}%")

    def skip_to_next_video(self):
        """Skip to the next video in the cycling sequence"""
        if hasattr(self, "video_processor") and self.video_processor:
            if self.video_processor.skip_to_next_video():
                print("Skipped to next video")
                # Update frame info
                self.update_frame_info()
            else:
                print("No more videos to skip to")

    def next_frame(self):
        """Navigate to the next frame"""
        if hasattr(self, "video_processor") and self.video_processor:
            if self.video_processor.navigate_frame("next"):
                print("Navigated to next frame")
                # Enable manual navigation mode
                self.video_processor.set_manual_navigation(True)
                # Update frame info
                self.update_frame_info()
            else:
                print("Already at the last frame")

    def previous_frame(self):
        """Navigate to the previous frame"""
        if hasattr(self, "video_processor") and self.video_processor:
            if self.video_processor.navigate_frame("previous"):
                print("Navigated to previous frame")
                # Enable manual navigation mode
                self.video_processor.set_manual_navigation(True)
                # Update frame info
                self.update_frame_info()
            else:
                print("Already at the first frame")

    def update_frame_info(self):
        """Update the frame information display"""
        if hasattr(self, "video_processor") and self.video_processor:
            frame_info = self.video_processor.get_frame_info()
            if hasattr(self, "frame_info_label"):
                self.frame_info_label.setText(frame_info)

    def on_frame_info_updated(self, frame_info):
        """Handle frame information updates from video processor"""
        if hasattr(self, "frame_info_label"):
            self.frame_info_label.setText(frame_info)


def main():
    if platform.system() == "Windows":
        try:
            import ctypes

            myappid = "dlhm.vidsgg.gui.1.0"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception as e:
            print(f"Pre-app Windows taskbar icon setting failed: {str(e)}")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    icon_path = "assets/tum-logo.png"
    if os.path.exists(icon_path):
        icon = QIcon(icon_path)
        sizes = [16, 32, 48, 64, 128]
        for size in sizes:
            icon.addFile(icon_path, QSize(size, size))
        app.setWindowIcon(icon)
        print(f"Set application icon: {icon_path}")
    else:
        print(f"Icon file not found: {icon_path}")

    set_application_icon(app, icon_path)
    window = SGGGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
