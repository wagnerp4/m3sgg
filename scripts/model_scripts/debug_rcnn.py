#!/usr/bin/env python3

import sys

sys.path.append(".")

import os

import cv2
import torch

from lib.datasets.action_genome import AG
from lib.object_detector import detector


def debug_person_detection():
    # Load dataset
    print("Loading dataset...")
    dataset = AG(mode="test", datasize="mini", data_path="data/action_genome")

    # Initialize object detector
    print("Initializing object detector...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    object_detector = detector(
        train=False,
        object_classes=dataset.object_classes,
        use_SUPPLY=False,
        mode="sgdet",
    ).to(device)
    object_detector.eval()

    # Load test image
    test_image_path = "assets/000165.png"
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return

    print(f"Loading test image: {test_image_path}")
    image = cv2.imread(test_image_path)
    if image is None:
        print("Could not load image")
        return

    # Convert BGR to RGB and prepare for model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to expected input size (600x600)
    resized_image = cv2.resize(image_rgb, (600, 600))

    # Convert to tensor format with same preprocessing as test
    image_tensor = torch.from_numpy(resized_image).float().permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor / 255.0
    image_tensor = (
        image_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    im_data = image_tensor.to(device)
    im_info = torch.tensor([[600, 600, 1.0]]).to(device)

    # Dummy values for training mode parameters
    gt_boxes = torch.tensor([[[0, 0, 0, 0, 0]]]).to(device)
    num_boxes = torch.tensor([0]).to(device)
    gt_annotation = {"bbox": [], "class": []}
    im_all = None

    print("Running object detection...")
    with torch.no_grad():
        result = object_detector(
            im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all
        )

    print("Detection completed!")

    if "scores" in result:
        scores = result["scores"]
        print(f"Detection scores shape: {scores.shape}")
        print(f"Final detection scores: {scores}")

    if "pred_labels" in result:
        labels = result["pred_labels"]
        print(f"Final predicted labels: {labels}")
        label_names = [dataset.object_classes[label.item()] for label in labels]
        print(f"Final predicted class names: {label_names}")

        from collections import Counter

        class_counts = Counter(label_names)
        print(f"Final class distribution: {dict(class_counts)}")

    # Check distribution
    if "distribution" in result:
        dist = result["distribution"]
        print(f"\nDistribution shape: {dist.shape}")
        print(f"Distribution max values per class (first 10 classes):")
        for i in range(min(10, dist.shape[1])):
            max_prob = dist[:, i].max().item()
            # Note: distribution excludes background, so add 1 to get class index
            class_name = (
                dataset.object_classes[i + 1]
                if (i + 1) < len(dataset.object_classes)
                else f"class_{i+1}"
            )
            print(f"  {class_name}: {max_prob:.6f}")


if __name__ == "__main__":
    debug_person_detection()
