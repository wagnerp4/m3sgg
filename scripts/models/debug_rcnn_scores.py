#!/usr/bin/env python3
"""
Debug R-CNN Score Threshold Issue
This script examines the raw scores from the R-CNN backbone before filtering
to understand why only door/doorway are being detected.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Add the project root to the path
sys.path.append(".")

from dataloader.action_genome import AG
from lib.object_detector import detector


def debug_rcnn_scores():
    print("=== Debugging R-CNN Score Threshold Issue ===")

    # Load dataset
    print("Loading dataset...")
    dataset = AG(
        mode="test",
        datasize="large",
        data_path="data/action_genome",
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    print(f"Dataset loaded with {len(dataset.object_classes)} object classes")
    print(f"Object classes: {dataset.object_classes}")

    # Initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize object detector
    print("Initializing object detector...")
    object_detector = detector(
        train=False,
        object_classes=dataset.object_classes,
        use_SUPPLY=True,
        mode="sgdet",
    ).to(device=device)
    object_detector.eval()

    # Load test image
    test_image_path = "image.png"
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return

    print(f"Loading test image: {test_image_path}")
    image = cv2.imread(test_image_path)
    if image is None:
        print("Failed to load image")
        return

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize and normalize
    frame_resized = cv2.resize(image_rgb, (600, 600))
    frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0)
    frame_tensor = frame_tensor / 255.0
    frame_tensor = (
        frame_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Create input tensors
    im_data = frame_tensor.to(device)
    im_info = torch.tensor([[600, 600, 1.0]], dtype=torch.float32).to(device)
    gt_boxes = torch.zeros([1, 1, 5]).to(device)
    num_boxes = torch.zeros([1], dtype=torch.int64).to(device)
    empty_annotation = []

    print("Running object detection...")

    with torch.no_grad():
        # Get raw Faster R-CNN outputs
        rois, cls_prob, bbox_pred, base_feat, roi_features = object_detector.fasterRCNN(
            im_data, im_info, gt_boxes, num_boxes
        )

        print(f"\n=== Raw Faster R-CNN Outputs ===")
        print(f"ROIs shape: {rois.shape}")
        print(f"Class probabilities shape: {cls_prob.shape}")
        print(f"Bbox predictions shape: {bbox_pred.shape}")

        # Analyze raw class probabilities
        SCORES = cls_prob.data
        print(f"\nRaw scores shape: {SCORES.shape}")
        print(f"Raw scores for first ROI: {SCORES[0, :10]}")  # First 10 classes

        # Check scores for each class
        print(f"\n=== Score Analysis by Class ===")
        print("Class | Max Score | Mean Score | Detections > 0.1 | Detections > 0.05")
        print("-" * 70)

        for j in range(1, len(dataset.object_classes)):  # Skip background (class 0)
            class_scores = SCORES[:, j]
            max_score = class_scores.max().item()
            mean_score = class_scores.mean().item()
            detections_01 = (class_scores > 0.1).sum().item()
            detections_005 = (class_scores > 0.05).sum().item()

            class_name = dataset.object_classes[j]
            print(
                f"{j:2d} ({class_name:15s}) | {max_score:8.3f} | {mean_score:9.3f} | {detections_01:16d} | {detections_005:15d}"
            )

            # Show top 5 scores for important classes
            if class_name in [
                "person",
                "sofa/couch",
                "shelf",
                "cup/glass/bottle",
                "laptop",
            ]:
                top_scores, top_indices = torch.topk(
                    class_scores, min(5, len(class_scores))
                )
                print(f"    Top 5 scores for {class_name}: {top_scores.cpu().numpy()}")

        # Check what happens with different thresholds
        print(f"\n=== Threshold Analysis ===")
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

        for threshold in thresholds:
            total_detections = 0
            detected_classes = []

            for j in range(1, len(dataset.object_classes)):
                class_scores = SCORES[:, j]
                detections = (class_scores > threshold).sum().item()
                if detections > 0:
                    total_detections += detections
                    detected_classes.append(
                        f"{dataset.object_classes[j]}({detections})"
                    )

            print(
                f"Threshold {threshold:4.2f}: {total_detections:3d} total detections - {', '.join(detected_classes)}"
            )

        # Now run the full object detector to see what it returns
        print(f"\n=== Full Object Detector Output ===")
        entry = object_detector(
            im_data, im_info, gt_boxes, num_boxes, empty_annotation, im_all=None
        )

        print(f"Final detections: {entry['boxes'].shape[0]}")
        if entry["boxes"].shape[0] > 0:
            print(
                f"Detected classes: {[dataset.object_classes[label.item()] for label in entry['pred_labels']]}"
            )
            print(f"Detection scores: {entry['scores'].cpu().numpy()}")


if __name__ == "__main__":
    debug_rcnn_scores()
