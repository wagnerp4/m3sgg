import copy
import os
from time import time
from typing import Dict

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from dotenv import load_dotenv
from matplotlib import colors as mcolors
from matplotlib.patches import FancyBboxPatch
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

from datasets.action_genome import AG, cuda_collate_fn
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.matcher import *
from lib.object_detector import detector
from lib.sttran import STTran
from lib.track import get_sequence


class ModelCaller:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name

    def call_gpt(self, prompt: str, max_tokens: int = 1000) -> Dict:
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
                seed=42,
            )
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "error": None,
            }
        except Exception as e:
            return {"success": False, "content": None, "error": str(e)}


class SceneGraphDemo:
    def __init__(self, model_path="output/model_best.tar"):
        self.conf = Config()
        self.conf.data_path = "action_genome"
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load dataset
        self.AG_dataset = AG(
            mode="test",
            datasize=self.conf.datasize,
            data_path=self.conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if self.conf.mode == "predcls" else True,
        )

        self.dataloader = torch.utils.data.DataLoader(
            self.AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn
        )

        # Initialize models
        self.setup_models()

        # Color mappings for visualization
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        self.relationship_colors = {
            "attention": "red",
            "spatial": "blue",
            "contact": "green",
        }

        self.pixel_means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.pixel_stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Load OpenAI API key and initialize ModelCaller
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or .env file!")
        self.llm_caller = ModelCaller(openai_api_key=openai_api_key)
        self.llm_model_name = "gpt-3.5-turbo"
        self.frame_summaries = []  # Accumulate per-frame summaries
        self.action_log = []  # Accumulate per-frame actions
        self.summarize_window = (
            1  # Summarize every N frames (default 1, can be changed)
        )

    def setup_models(self):
        """Initialize object detector and scene graph model"""
        # Object detector
        self.object_detector = detector(
            train=False,
            object_classes=self.AG_dataset.object_classes,
            use_SUPPLY=True,
            mode=self.conf.mode,
        ).to(device=self.device)
        self.object_detector.eval()

        # Scene graph model
        self.model = STTran(
            mode=self.conf.mode,
            attention_class_num=len(self.AG_dataset.attention_relationships),
            spatial_class_num=len(self.AG_dataset.spatial_relationships),
            contact_class_num=len(self.AG_dataset.contacting_relationships),
            obj_classes=self.AG_dataset.object_classes,
            enc_layer_num=self.conf.enc_layer,
            dec_layer_num=self.conf.dec_layer,
        ).to(device=self.device)

        # Load trained weights
        if os.path.exists(self.model_path):
            ckpt = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"], strict=False)
            print(f"Loaded model from {self.model_path}")
        else:
            print(f"Warning: Model file {self.model_path} not found!")

        self.model.eval()

        # Matcher for sequence generation
        self.matcher = HungarianMatcher(0.5, 1, 1, 0.5).to(device=self.device)
        self.matcher.eval()

    def draw_bounding_boxes(self, image, boxes, labels, scores, relationships=None):
        """Draw bounding boxes and labels on image"""
        img_copy = image.copy()
        h, w = img_copy.shape[:2]

        # Filter and sort by confidence, keep only top k
        top_k_bboxes = 10
        valid_detections = [
            (box, label, score, i)
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores))
            if score > 0.7
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

            # Use consistent color for each object class
            color_idx = label % len(self.colors)
            color = self.colors[color_idx]
            color_bgr = tuple(int(c * 255) for c in mcolors.to_rgb(color))

            # Draw bounding box with thicker line
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color_bgr, 3)

            # Draw label with better visibility
            if label < len(self.AG_dataset.object_classes):
                label_text = f"{self.AG_dataset.object_classes[label]} ({score:.2f})"
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

    def create_scene_graph(
        self,
        pred_relations,
        object_labels,
        frame_size=(800, 400),
        top_k_relationships=3,
    ):
        """Create scene graph visualization"""
        fig, ax = plt.subplots(
            1, 1, figsize=(frame_size[0] / 100, frame_size[1] / 100), dpi=100
        )
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        fig.patch.set_facecolor("white")

        # Filter relationships to avoid overcrowding - keep only top k most confident
        filtered_relations = (
            pred_relations[:top_k_relationships]
            if len(pred_relations) > top_k_relationships
            else pred_relations
        )

        # Add nodes (objects) involved in relationships
        unique_objects = set()
        for rel in filtered_relations:
            subj_idx, obj_idx, predicate, rel_type, conf = rel
            if subj_idx < len(object_labels) and obj_idx < len(object_labels):
                subj_label = self.AG_dataset.object_classes[object_labels[subj_idx]]
                obj_label = self.AG_dataset.object_classes[object_labels[obj_idx]]
                unique_objects.add((subj_idx, subj_label))
                unique_objects.add((obj_idx, obj_label))

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

        # Position nodes
        positions = {}
        if len(unique_objects) > 0:
            objects_list = list(unique_objects)
            if len(objects_list) == 1:
                positions[objects_list[0][0]] = (5, 3)
            elif len(objects_list) == 2:
                positions[objects_list[0][0]] = (3, 3)
                positions[objects_list[1][0]] = (7, 3)
            else:
                for i, (idx, label) in enumerate(objects_list):
                    if i == 0:  # Person usually at center
                        positions[idx] = (5, 3)
                    else:
                        angle = 2 * np.pi * (i - 1) / max(1, len(objects_list) - 1)
                        x = 5 + 2.5 * np.cos(angle)
                        y = 3 + 1.5 * np.sin(angle)
                        positions[idx] = (x, y)

            # Draw object nodes
            for idx, label in objects_list:
                if idx in positions:
                    x, y = positions[idx]
                    unique_label = idx_to_unique_label[idx]
                    circle = plt.Circle(
                        (x, y),
                        0.6,
                        color="lightblue",
                        alpha=0.8,
                        linewidth=2,
                        edgecolor="darkblue",
                    )
                    ax.add_patch(circle)
                    ax.text(
                        x,
                        y,
                        unique_label,
                        ha="center",
                        va="center",
                        fontsize=10,
                        weight="bold",
                    )

        # Draw relationships
        for rel in filtered_relations:
            subj_idx, obj_idx, predicate, rel_type, conf = rel
            if subj_idx in positions and obj_idx in positions and subj_idx != obj_idx:
                x1, y1 = positions[subj_idx]
                x2, y2 = positions[obj_idx]
                # Determine relationship type and color
                if rel_type == "attention":
                    rel_name = self.AG_dataset.attention_relationships[predicate]
                    color = "red"
                elif rel_type == "spatial":
                    rel_name = self.AG_dataset.spatial_relationships[predicate]
                    color = "blue"
                else:
                    rel_name = self.AG_dataset.contacting_relationships[predicate]
                    color = "green"
                # Draw arrow with better visibility
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=3, alpha=0.8),
                )
                # Add relationship label with better positioning
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                subj_label = idx_to_unique_label[subj_idx]
                obj_label = idx_to_unique_label[obj_idx]
                ax.text(
                    mid_x,
                    mid_y,
                    rel_name,
                    ha="center",
                    va="center",
                    fontsize=8,
                    weight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.9,
                        edgecolor=color,
                    ),
                )

        plt.tight_layout()
        # Convert to image (fix deprecation warning)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]  # Remove alpha channel
        plt.close(fig)
        return img

    def process_frame(self, data):
        """Process a single frame and return predictions"""
        im_data = copy.deepcopy(data[0]).to(device=self.device)
        im_info = copy.deepcopy(data[1]).to(device=self.device)
        gt_boxes = copy.deepcopy(data[2]).to(device=self.device)
        num_boxes = copy.deepcopy(data[3]).to(device=self.device)
        gt_annotation = self.AG_dataset.gt_annotations[data[4]]

        with torch.no_grad():
            # Object detection
            entry = self.object_detector(
                im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None
            )

            # Scene graph generation
            get_sequence(
                entry,
                gt_annotation,
                self.matcher,
                (im_info[0][:2] / im_info[0, 2]),
                self.conf.mode,
            )
            pred = self.model(entry)

        return entry, pred

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

    def format_relations_for_llm(self, pred_relations, labels):
        """
        Format the relationships as a sequence of action sentences for LLM summarization.
        """
        actions = []
        for i, (subj_idx, obj_idx, predicate, rel_type, conf) in enumerate(
            pred_relations
        ):
            if rel_type == "attention":
                rel_name = self.AG_dataset.attention_relationships[predicate]
            elif rel_type == "spatial":
                rel_name = self.AG_dataset.spatial_relationships[predicate]
            else:
                rel_name = self.AG_dataset.contacting_relationships[predicate]
            subj_label = self.AG_dataset.object_classes[labels[subj_idx]]
            obj_label = self.AG_dataset.object_classes[labels[obj_idx]]
            actions.append(f"Action {i+1}: {subj_label} {rel_name} {obj_label}")
        return "\n".join(actions)

    def render_text_image(self, text, width, height=60, font_size=24):
        """
        Render the summary text as an image using PIL.
        """
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        # Try textbbox (Pillow ≥8.0.0), else fallback to textsize
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except AttributeError:
            w, h = draw.textsize(text, font=font)
        draw.text(
            ((width - w) // 2, (height - h) // 2), text, fill=(0, 0, 0), font=font
        )
        arr = np.array(img)
        # Ensure shape is (height, width, 3) and dtype is uint8
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    def create_demo_video(
        self, output_path="demo_output.mp4", video_index=0, max_frames=5
    ):
        """
        Create a demonstration video from a single video sequence
        Args:
            output_path: Path to save the output video
            video_index: Index of the video to process (default: 0)
            max_frames: Maximum number of frames to process (default: 5)
        """
        print(f"Creating demonstration video for video index {video_index}...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 2
        first_batch = True
        frame_width, frame_height = 640, 480
        video_frames = self.AG_dataset.video_list[video_index]
        print(f"Processing video with {len(video_frames)} frames")
        top_k_relationships = 3

        for frame_idx in range(min(len(video_frames), max_frames)):
            print(
                f"Processing frame {frame_idx + 1}/{min(len(video_frames), max_frames)}"
            )
            try:
                data = self.AG_dataset.__getitem__(video_index)
                data = [
                    d.to(self.device) if isinstance(d, torch.Tensor) else d
                    for d in data
                ]

                # Process only the current frame
                im_data = data[0][frame_idx : frame_idx + 1]  # Get single frame
                im_info = data[1][frame_idx : frame_idx + 1]
                gt_boxes = data[2][frame_idx : frame_idx + 1]
                num_boxes = data[3][frame_idx : frame_idx + 1]
                frame_data = [im_data, im_info, gt_boxes, num_boxes, data[4]]

                entry, pred = self.process_frame(frame_data)  # Process frame
                im_tensor = frame_data[0][0]  # CHW, float32, normalised
                im_data = self.tensor_to_uint8(im_tensor)  # HWC, uint8, RGB

                # Ensure video writer is initialized with correct size
                if first_batch:
                    frame_height, frame_width = im_data.shape[:2]
                    summary_height = 60
                    total_height = (
                        frame_height + 400 + summary_height
                    )  # Space for scene graph
                    out = cv2.VideoWriter(
                        output_path, fourcc, fps, (frame_width, total_height)
                    )
                    first_batch = False

                # Get predictions with better filtering
                boxes = entry["boxes"][:, 1:].cpu().numpy()  # Remove batch dimension
                if self.conf.mode == "predcls":
                    labels = entry["labels"].cpu().numpy()
                    scores = entry["scores"].cpu().numpy()
                else:
                    labels = entry["pred_labels"].cpu().numpy()
                    scores = entry["pred_scores"].cpu().numpy()

                print("Number of boxes before NMS:", len(labels))

                # Apply NMS per class
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
                boxes = boxes[final_indices]
                labels = labels[final_indices]
                scores = scores[final_indices]

                # Build mapping from old indices to new indices after NMS
                old_to_new_idx = {
                    old_idx: new_idx for new_idx, old_idx in enumerate(final_indices)
                }

                print("After NMS:")
                for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                    print(
                        f"  Box {idx}: {self.AG_dataset.object_classes[label]}, {box}, Score: {score}"
                    )

                # Scale boxes to image size
                im_info = frame_data[1][0].cpu().numpy()
                scale = im_info[2]
                boxes = boxes * scale

                # Draw bounding boxes (now with better filtering)
                img_with_boxes = self.draw_bounding_boxes(
                    im_data, boxes, labels, scores
                )

                # Extract relationships for scene graph
                pred_relations = []
                if "pair_idx" in entry and entry["pair_idx"].shape[0] > 0:
                    pair_indices = entry["pair_idx"].cpu().numpy()
                    attention_dist = pred["attention_distribution"].cpu().numpy()
                    spatial_dist = pred["spatial_distribution"].cpu().numpy()
                    contact_dist = pred["contact_distribution"].cpu().numpy()

                    for i, (subj_idx, obj_idx) in enumerate(pair_indices):
                        # Only keep relationships where both subject and object survived NMS
                        if subj_idx in old_to_new_idx and obj_idx in old_to_new_idx:
                            new_subj_idx = old_to_new_idx[subj_idx]
                            new_obj_idx = old_to_new_idx[obj_idx]
                            # Attention
                            att_pred = np.argmax(attention_dist[i])
                            att_conf = attention_dist[i][att_pred]
                            if att_conf > 0.5:
                                pred_relations.append(
                                    (
                                        new_subj_idx,
                                        new_obj_idx,
                                        att_pred,
                                        "attention",
                                        att_conf,
                                    )
                                )
                            # Spatial
                            spa_pred = np.argmax(spatial_dist[i])
                            spa_conf = spatial_dist[i][spa_pred]
                            if spa_conf > 0.5:
                                pred_relations.append(
                                    (
                                        new_subj_idx,
                                        new_obj_idx,
                                        spa_pred,
                                        "spatial",
                                        spa_conf,
                                    )
                                )
                            # Contact
                            con_pred = np.argmax(contact_dist[i])
                            con_conf = contact_dist[i][con_pred]
                            if con_conf > 0.5:
                                pred_relations.append(
                                    (
                                        new_subj_idx,
                                        new_obj_idx,
                                        con_pred,
                                        "contact",
                                        con_conf,
                                    )
                                )
                    # Sort by confidence and take top k
                    pred_relations.sort(key=lambda x: x[4], reverse=True)
                    pred_relations = pred_relations[:top_k_relationships]

                # Store actions for this frame
                frame_actions = self.format_relations_for_llm(pred_relations, labels)
                self.action_log.append(frame_actions)

                # Create scene graph
                scene_graph_img = self.create_scene_graph(
                    pred_relations, labels, top_k_relationships=top_k_relationships
                )

                # Ensure scene graph has the right size
                if scene_graph_img.shape[1] != frame_width:
                    scene_graph_img = cv2.resize(scene_graph_img, (frame_width, 400))

                # Ensure both images have same width before stacking
                if img_with_boxes.shape[1] != frame_width:
                    img_with_boxes = cv2.resize(
                        img_with_boxes, (frame_width, frame_height)
                    )

                # Combine images vertically
                combined_frame = np.vstack([img_with_boxes, scene_graph_img])

                # Batch LLM summarization
                do_summarize = (frame_idx + 1) % self.summarize_window == 0
                if do_summarize:
                    start_idx = max(0, len(self.action_log) - self.summarize_window)
                    actions_context = "\n".join(self.action_log[start_idx:])
                    prev_summaries = "\n".join(
                        self.frame_summaries[-3:]
                    )  # last 3 summaries as context
                    llm_prompt = (
                        "You are an assistant who can model human behaviour very well. "
                        "You'll be provided with a sequence of actions retrieved from a first-person view video. "
                        "Your task is to understand the general activity and describe it in one sentence. "
                        "Please, provide a very general summary and try to avoid listing all the 'atomic' activities.\n\n"
                        f"Previous summaries (context):\n{prev_summaries}\n\n"
                        f"Current actions:\n{actions_context}"
                    )
                    llm_response = self.llm_caller.call_gpt(llm_prompt, max_tokens=60)
                    if llm_response["success"] and llm_response["content"]:
                        summary_text = llm_response["content"].strip()
                    else:
                        summary_text = "[LLM error: could not summarize]"
                    self.frame_summaries.append(summary_text)
                else:
                    summary_text = "[No summary this frame]"

                # Render summary text as image and stack below scene graph
                summary_img = self.render_text_image(
                    summary_text, width=frame_width, height=60
                )
                combined_frame = np.vstack([combined_frame, summary_img])

                # Write frame
                out.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue

        if not first_batch:
            out.release()
            print(f"Demo video saved to: {output_path}")
        else:
            print("No frames were processed successfully!")


def main():
    print("Starting Scene Graph Demo...")
    print("=" * 50)
    model_path = "output/model_best.tar"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        print("Please ensure you have a trained model at the specified path.")
        return

    data_path = "action_genome"
    if not os.path.exists(data_path):
        print(f"Data directory {data_path} not found!")
        print("Please ensure the Action Genome dataset is in the correct location.")
        return

    required_files = [
        "action_genome/annotations/object_classes.txt",
        "action_genome/annotations/relationship_classes.txt",
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Required file {file_path} not found!")
            return

    # Create demo
    try:
        demo = SceneGraphDemo(model_path)
        demo.create_demo_video(
            "scene_graph_demo.mp4", video_index=0, max_frames=10
        )  # Process first video
    except Exception as e:
        print(f"Error creating demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
