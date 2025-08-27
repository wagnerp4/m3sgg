import os
import sys

import cv2
import numpy as np
import torch

# Add the project root to the path
sys.path.append(".")

from dataloader.action_genome import AG
from lib.object_detector import detector


def test_rcnn_with_frame(frame_path=None):
    """Test R-CNN backbone with a specific frame or create a test frame"""

    print("=== Simple R-CNN Backbone Test ===")

    # Load dataset
    dataset = AG(
        mode="test",
        datasize="large",
        data_path="data/action_genome",
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    print(f"Dataset loaded with {len(dataset.object_classes)} object classes")

    # Initialize object detector
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    object_detector = detector(
        train=False,
        object_classes=dataset.object_classes,
        use_SUPPLY=True,
        mode="sgdet",
    ).to(device=device)
    object_detector.eval()

    # Prepare input data
    if frame_path and os.path.exists(frame_path):
        # Use provided frame
        print(f"Using frame: {frame_path}")
        image = cv2.imread(frame_path)
        if image is None:
            print(f"Failed to load frame: {frame_path}")
            return
    else:
        # Create a test frame with some objects
        print("Creating test frame...")
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background

        # Draw some simple shapes to simulate objects
        cv2.rectangle(image, (100, 100), (200, 300), (255, 0, 0), -1)  # Blue rectangle
        cv2.rectangle(image, (400, 150), (550, 250), (0, 255, 0), -1)  # Green rectangle
        cv2.circle(image, (300, 350), 50, (0, 0, 255), -1)  # Red circle

    # Preprocess image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (600, 600))

    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor / 255.0
    image_tensor = (
        image_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Create input tensors
    im_data = image_tensor.to(device)
    im_info = torch.tensor([[600, 600, 1.0]], dtype=torch.float32).to(device)
    gt_boxes = torch.zeros([1, 1, 5]).to(device)
    num_boxes = torch.zeros([1], dtype=torch.int64).to(device)
    empty_annotation = []

    print("Running object detection...")

    with torch.no_grad():
        entry = object_detector(
            im_data, im_info, gt_boxes, num_boxes, empty_annotation, im_all=None
        )

    # Analyze results
    print("\n=== Detection Results ===")

    boxes = entry.get("boxes", torch.empty(0, 5))
    pred_labels = entry.get("pred_labels", torch.empty(0))
    scores = entry.get("scores", torch.empty(0))
    distribution = entry.get("distribution", torch.empty(0))

    print(f"Total detections: {boxes.shape[0]}")

    if boxes.shape[0] > 0:
        print(f"\nDetected objects:")
        for i in range(boxes.shape[0]):
            label_idx = pred_labels[i].item()
            label_name = (
                dataset.object_classes[label_idx]
                if label_idx < len(dataset.object_classes)
                else f"unknown_{label_idx}"
            )
            score = scores[i].item()
            box = boxes[i].cpu().numpy()
            print(f"  {i+1}. {label_name} (score: {score:.3f}) at box {box}")

        # Show class distribution
        label_names = [dataset.object_classes[label.item()] for label in pred_labels]
        from collections import Counter

        class_counts = Counter(label_names)
        print(f"\nClass distribution: {dict(class_counts)}")

        # Show score statistics
        scores_np = scores.cpu().numpy()
        print(f"\nScore statistics:")
        print(f"  Min: {scores_np.min():.3f}")
        print(f"  Max: {scores_np.max():.3f}")
        print(f"  Mean: {scores_np.mean():.3f}")

        # Show detections at different thresholds
        print(f"\nDetections at different thresholds:")
        for threshold in [0.01, 0.05, 0.1, 0.3, 0.5]:
            above_threshold = scores_np > threshold
            count = above_threshold.sum()
            if count > 0:
                above_indices = np.where(above_threshold)[0]
                above_labels = [label_names[i] for i in above_indices]
                above_scores = scores_np[above_threshold]
                print(
                    f"  >{threshold}: {count} detections - {above_labels} (scores: {above_scores})"
                )

        # Show top predictions from distribution
        if distribution.shape[0] > 0:
            print(f"\nTop class predictions for each detection:")
            for i in range(min(3, distribution.shape[0])):
                probs = distribution[i].cpu().numpy()
                top_indices = probs.argsort()[-3:][::-1]
                top_probs = probs[top_indices]
                top_classes = [dataset.object_classes[idx + 1] for idx in top_indices]
                print(f"  Detection {i+1}: {list(zip(top_classes, top_probs))}")

        # Visualize detections
        print(f"\n=== Visualizing Detections ===")

        # Create a copy for drawing
        vis_image = image.copy()

        # Colors for different classes
        colors = [
            (255, 0, 0),  # Blue
            (0, 255, 0),  # Green
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 0),  # Dark Blue
            (0, 128, 0),  # Dark Green
            (0, 0, 128),  # Dark Red
            (128, 128, 0),  # Olive
        ]

        boxes_np = boxes.cpu().numpy()
        pred_labels_np = pred_labels.cpu().numpy()
        scores_np = scores.cpu().numpy()

        print(f"Drawing {len(boxes_np)} bounding boxes...")

        for i, (box, label, score) in enumerate(
            zip(boxes_np, pred_labels_np, scores_np)
        ):
            # Box format: [batch_idx, x1, y1, x2, y2]
            # We need to scale from 600x600 back to original size
            x1, y1, x2, y2 = box[1:]  # Remove batch index

            # Scale coordinates back to original image size
            orig_h, orig_w = image.shape[:2]
            x1 = int(x1 * orig_w / 600)
            y1 = int(y1 * orig_h / 600)
            x2 = int(x2 * orig_w / 600)
            y2 = int(y2 * orig_h / 600)

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            # Get class name
            label_name = (
                dataset.object_classes[label]
                if label < len(dataset.object_classes)
                else f"unknown_{label}"
            )

            # Choose color
            color = colors[label % len(colors)]

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label_text = f"{label_name} ({score:.2f})"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Background for text
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1,
            )

            # Text
            cv2.putText(
                vis_image,
                label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            print(
                f"  Box {i+1}: {label_name} at ({x1},{y1},{x2},{y2}) with score {score:.3f}"
            )

        # Save the visualization
        output_path = "rcnn_simple_detections.png"
        cv2.imwrite(output_path, vis_image)
        print(f"\nVisualization saved to: {output_path}")

        # Display the image (if you have a display)
        try:
            cv2.imshow("R-CNN Simple Detections", vis_image)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Could not display image (no display available)")
    else:
        print("No objects detected!")

    return entry


if __name__ == "__main__":
    # You can provide a frame path as argument
    frame_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_rcnn_with_frame(frame_path)
