#!/usr/bin/env python3
"""
Test Lower R-CNN Threshold
This script temporarily lowers the detection threshold to see if more objects are detected.
"""

import os
import sys

import cv2
import numpy as np
import torch

# Add the project root to the path
sys.path.append(".")

from datasets.action_genome import AG
from lib.object_detector import detector


def test_lower_threshold():
    print("=== Testing Lower R-CNN Threshold ===")

    # Load dataset
    print("Loading dataset...")
    dataset = AG(
        mode="test",
        datasize="large",
        data_path="data/action_genome",
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    # Initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize object detector
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

    image = cv2.imread(test_image_path)
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

    print("Testing different thresholds...")

    with torch.no_grad():
        # Get raw Faster R-CNN outputs
        rois, cls_prob, bbox_pred, base_feat, roi_features = object_detector.fasterRCNN(
            im_data, im_info, gt_boxes, num_boxes
        )

        SCORES = cls_prob.data

        # Test different thresholds
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.2]

        for threshold in thresholds:
            print(f"\n--- Threshold {threshold} ---")

            # Simulate the detection logic with different threshold
            total_detections = 0
            detected_classes = []

            for j in range(1, len(dataset.object_classes)):
                class_scores = SCORES[:, j]
                detections = (class_scores > threshold).sum().item()
                if detections > 0:
                    max_score = class_scores.max().item()
                    detected_classes.append(
                        f"{dataset.object_classes[j]}({detections}, {max_score:.3f})"
                    )
                    total_detections += detections

            print(f"Total detections: {total_detections}")
            print(f"Detected classes: {', '.join(detected_classes)}")

            # Show top 10 highest scoring detections across all classes
            if total_detections > 0:
                all_scores = []
                for j in range(1, len(dataset.object_classes)):
                    class_scores = SCORES[:, j]
                    valid_scores = class_scores[class_scores > threshold]
                    for score in valid_scores:
                        all_scores.append((score.item(), dataset.object_classes[j]))

                all_scores.sort(reverse=True)
                print(f"Top 10 detections: {all_scores[:10]}")


if __name__ == "__main__":
    test_lower_threshold()
