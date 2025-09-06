import os
import sys
from time import time

import cv2
import numpy as np
import torch

sys.path.append(".")

from lib.datasets.action_genome import AG
from lib.matcher import HungarianMatcher
from lib.object_detector import detector
from lib.sttran import STTran
from lib.track import get_sequence


def test_rcnn_backbone():
    print("=== Testing R-CNN Backbone Object Detector ===")
    dataset = AG(
        mode="test",
        datasize="large",
        data_path="data/action_genome",
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    print(f"Dataset loaded with {len(dataset.object_classes)} object classes")
    print(f"Object classes: {dataset.object_classes}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    object_detector = detector(
        train=False,
        object_classes=dataset.object_classes,
        use_SUPPLY=True,
        mode="sgdet",
    ).to(device=device)
    object_detector.eval()
    print("Object detector initialized")
    test_image_path = "assets/000070.png"  # 000165.png, image.png 000070.png 000232.png
    print(f"Testing with image: {test_image_path}")

    if os.path.exists(test_image_path):
        print(f"Testing with image: {test_image_path}")
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"Failed to load image: {test_image_path}")
            return
        # Image preprocessing (no normalization - R-CNN expects raw pixel values)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (600, 600))
        image_tensor = (
            torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0)
        )
        # Create input tensors
        im_data = image_tensor.to(device)
        im_info = torch.tensor([[600, 600, 1.0]], dtype=torch.float32).to(device)
        gt_boxes = torch.zeros([1, 1, 5]).to(device)
        num_boxes = torch.zeros([1], dtype=torch.int64).to(device)
        # Empty annotation for sgdet mode
        empty_annotation = []
        print("Running object detection...")
        with torch.no_grad():
            entry = object_detector(
                im_data, im_info, gt_boxes, num_boxes, empty_annotation, im_all=None
            )
        print("\n=== Detection Results ===")
        print(f"Entry keys: {list(entry.keys())}")

        if "boxes" in entry:
            boxes = entry["boxes"]
            print(f"Number of detected boxes: {boxes.shape[0]}")
            print(f"Boxes shape: {boxes.shape}")
            print(f"Boxes content:\n{boxes}")
        if "pred_labels" in entry:
            pred_labels = entry["pred_labels"]
            print(f"Number of predicted labels: {pred_labels.shape[0]}")
            print(f"Predicted labels: {pred_labels}")
            label_names = [
                dataset.object_classes[label.item()] for label in pred_labels
            ]
            print(f"Predicted class names: {label_names}")
            from collections import Counter

            class_counts = Counter(label_names)
            print(f"Class distribution: {dict(class_counts)}")
        if "scores" in entry:
            scores = entry["scores"]
            print(f"Number of scores: {scores.shape[0]}")
            print(f"Scores: {scores}")
            scores_np = scores.cpu().numpy()
            print("Score statistics:")
            print(f"  Min: {scores_np.min():.3f}")
            print(f"  Max: {scores_np.max():.3f}")
            print(f"  Mean: {scores_np.mean():.3f}")
            print(f"  Std: {scores_np.std():.3f}")
            thresholds = [0.1]
            for threshold in thresholds:
                above_threshold = scores_np > threshold
                count = above_threshold.sum()
                print(f"  Detections above {threshold}: {count}")
                if count > 0 and "pred_labels" in entry:
                    above_indices = np.where(above_threshold)[0]
                    above_labels = [
                        dataset.object_classes[entry["pred_labels"][i].item()]
                        for i in above_indices
                    ]
                    above_scores = scores_np[above_threshold]
                    print(f"    Classes: {above_labels}")
                    print(f"    Scores: {above_scores}")

        if "distribution" in entry:
            distribution = entry["distribution"]
            print(f"Distribution shape: {distribution.shape}")
            print(f"Distribution (first few rows):\n{distribution[:5]}")

            if distribution.shape[0] > 0:
                print("\nTop class predictions for each detection:")
                for i in range(min(5, distribution.shape[0])):
                    probs = distribution[i].cpu().numpy()
                    top_indices = probs.argsort()[-5:][::-1]
                    top_probs = probs[top_indices]
                    top_classes = [
                        dataset.object_classes[idx + 1] for idx in top_indices
                    ]  # +1 because distribution excludes background
                    print(f"  Detection {i}: {list(zip(top_classes, top_probs))}")

        # Visualize detections with bounding boxes
        if "boxes" in entry and entry["boxes"].shape[0] > 0:
            print("\n=== Visualizing Detections ===")
            orig_h, orig_w = image.shape[:2]
            print(f"Original image size: {orig_w}x{orig_h}")
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
            boxes = entry["boxes"].cpu().numpy()
            pred_labels = entry["pred_labels"].cpu().numpy()
            scores = entry["scores"].cpu().numpy()
            print(f"Drawing {len(boxes)} bounding boxes...")

            for i, (box, label, score) in enumerate(zip(boxes, pred_labels, scores)):
                # Box format: [batch_idx, x1, y1, x2, y2]
                # We need to scale from 600x600 back to original size
                x1, y1, x2, y2 = box[1:]  # Remove batch index
                # Scale coordinates back to original image size
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

            output_path = "assets/rcnn_detections.png"
            cv2.imwrite(output_path, vis_image)
            print(f"\nVisualization saved to: {output_path}")
            try:
                cv2.imshow("R-CNN Detections", vis_image)
                print("Press any key to close the window...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Could not display image (no display available): {e}")
                print("Continuing with STTran testing...")

        return entry
    else:
        print(f"Test image not found: {test_image_path}")
        return None


def test_sttran_sgdet(entry, dataset, device):
    """Test STTran with sgdet mode using a pre-defined checkpoint path"""
    print("\n" + "=" * 60)
    print("=== Testing STTran with sgdet mode ===")
    print("=" * 60)

    # Pre-defined checkpoint path - modify this as needed
    # checkpoint_path = "output/action_genome9000/sttran/sgdet/20250726_221743/model_best.tar"
    checkpoint_path = "data/checkpoints/action_genome/sgdet_test/model_best.tar"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable in the script")
        return None

    print(f"Loading STTran checkpoint from: {checkpoint_path}")

    # Initialize STTran model
    model = STTran(
        mode="sgdet",
        attention_class_num=len(dataset.attention_relationships),
        spatial_class_num=len(dataset.spatial_relationships),
        contact_class_num=len(dataset.contacting_relationships),
        obj_classes=dataset.object_classes,
        enc_layer_num=6,  # Default values from test.py
        dec_layer_num=6,  # Default values from test.py
    ).to(device=device)

    model.eval()

    # Load checkpoint
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"✓ Checkpoint loaded successfully from {checkpoint_path}")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return None

    # Initialize matcher for sequence processing
    matcher = HungarianMatcher(0.5, 1, 1, 0.5)
    matcher.eval()

    # Create dummy annotation for testing (since we're in sgdet mode)
    dummy_annotation = {"objects": [], "relationships": []}

    # Process the entry through STTran
    print("\n--- Processing through STTran ---")
    start_time = time()

    with torch.no_grad():
        # Fix the entry structure for STTran compatibility
        # STTran expects pred_scores field and proper score handling
        if "distribution" in entry and "pred_scores" not in entry:
            # Create pred_scores from distribution (max probability excluding background)
            if entry["distribution"].shape[1] > 1:
                entry["pred_scores"] = torch.max(entry["distribution"][:, 1:], dim=1)[0]
            else:
                entry["pred_scores"] = torch.max(entry["distribution"], dim=1)[0]
            print(
                f"Created pred_scores from distribution: shape {entry['pred_scores'].shape}"
            )

        # Make sure entry["scores"] is compatible with get_sequence function
        # The function expects to be able to use argmax with dim=1, but we have 1D scores
        # For sgdet mode, we can use the distribution instead
        original_scores = entry["scores"].clone()
        original_boxes = entry["boxes"].clone()

        entry["scores"] = entry["distribution"]  # Use 2D distribution for get_sequence
        # Fix boxes format for NMS: remove batch index column for get_sequence
        entry["boxes"] = entry["boxes"][:, 1:]  # Remove first column (batch index)

        # Get sequence information
        get_sequence(
            entry,
            dummy_annotation,
            matcher,
            torch.tensor([600, 600]).to(device),  # Image dimensions
            "sgdet",
        )

        # Restore original scores and boxes after get_sequence
        entry["scores"] = original_scores
        entry["boxes"] = original_boxes

        # Run STTran inference
        predictions = model(entry)

    inference_time = time() - start_time
    print(f"✓ STTran inference completed in {inference_time:.3f} seconds")

    # Analyze predictions
    print("\n--- STTran Predictions Analysis ---")
    print(f"Prediction keys: {list(predictions.keys())}")

    # Analyze final bounding boxes after STTran processing
    if "final_boxes" in predictions:
        final_boxes = predictions["final_boxes"]
        print("\n--- Final Bounding Boxes after STTran ---")
        print(f"Number of final boxes: {final_boxes.shape[0]}")
        print(f"Final boxes shape: {final_boxes.shape}")

        if "final_labels" in predictions:
            final_labels = predictions["final_labels"]
            print(f"Final labels: {final_labels}")
            final_label_names = [
                dataset.object_classes[label.item()] for label in final_labels
            ]
            print(f"Final class names: {final_label_names}")

    # Analyze relationship predictions
    if "pair_idx" in predictions:
        pair_idx = predictions["pair_idx"]
        print("\n--- Relationship Pairs ---")
        print(f"Number of pairs: {pair_idx.shape[0]}")
        print(f"Pair indices shape: {pair_idx.shape}")
        if pair_idx.shape[0] > 0:
            print(f"First 10 pairs (human_idx, object_idx):\n{pair_idx[:10]}")

    # Analyze attention distributions
    if "attention_distribution" in predictions:
        attention_dist = predictions["attention_distribution"]
        print("\n--- Attention Relationships ---")
        print(f"Attention distribution shape: {attention_dist.shape}")
        if attention_dist.numel() > 0:
            attention_scores = torch.softmax(attention_dist, dim=-1)
            top_attention = torch.topk(attention_scores, k=2, dim=-1)
            print(
                f"Top attention scores (first 5 pairs):\n{top_attention.values[:5].cpu().numpy()}"
            )
            print(
                f"Top attention indices (first 5 pairs):\n{top_attention.indices[:5].cpu().numpy()}"
            )

            # Show actual relationship names for top predictions
            for i in range(min(5, attention_dist.shape[0])):
                top_idx = top_attention.indices[i][0].item()
                top_score = top_attention.values[i][0].item()
                if top_idx < len(dataset.attention_relationships):
                    rel_name = dataset.attention_relationships[top_idx]
                    print(f"  Pair {i}: {rel_name} (score: {top_score:.3f})")

    # Analyze spatial distributions
    if "spatial_distribution" in predictions:
        spatial_dist = predictions["spatial_distribution"]
        print("\n--- Spatial Relationships ---")
        print(f"Spatial distribution shape: {spatial_dist.shape}")
        if spatial_dist.numel() > 0:
            spatial_scores = torch.softmax(spatial_dist, dim=-1)
            top_spatial = torch.topk(spatial_scores, k=2, dim=-1)
            print(
                f"Top spatial scores (first 5 pairs):\n{top_spatial.values[:5].cpu().numpy()}"
            )
            print(
                f"Top spatial indices (first 5 pairs):\n{top_spatial.indices[:5].cpu().numpy()}"
            )

            # Show actual relationship names for top predictions
            for i in range(min(5, spatial_dist.shape[0])):
                top_idx = top_spatial.indices[i][0].item()
                top_score = top_spatial.values[i][0].item()
                if top_idx < len(dataset.spatial_relationships):
                    rel_name = dataset.spatial_relationships[top_idx]
                    print(f"  Pair {i}: {rel_name} (score: {top_score:.3f})")

    # Analyze contact distributions
    if "contact_distribution" in predictions:
        contact_dist = predictions["contact_distribution"]
        print("\n--- Contact Relationships ---")
        print(f"Contact distribution shape: {contact_dist.shape}")
        if contact_dist.numel() > 0:
            contact_scores = torch.softmax(contact_dist, dim=-1)
            top_contact = torch.topk(contact_scores, k=2, dim=-1)
            print(
                f"Top contact scores (first 5 pairs):\n{top_contact.values[:5].cpu().numpy()}"
            )
            print(
                f"Top contact indices (first 5 pairs):\n{top_contact.indices[:5].cpu().numpy()}"
            )

            # Show actual relationship names for top predictions
            for i in range(min(5, contact_dist.shape[0])):
                top_idx = top_contact.indices[i][0].item()
                top_score = top_contact.values[i][0].item()
                if top_idx < len(dataset.contacting_relationships):
                    rel_name = dataset.contacting_relationships[top_idx]
                    print(f"  Pair {i}: {rel_name} (score: {top_score:.3f})")

    # Show human indices if available
    if "human_idx" in predictions:
        human_idx = predictions["human_idx"]
        print("\n--- Human Detection ---")
        print(f"Human indices: {human_idx}")
        if "final_boxes" in predictions and human_idx.numel() > 0:
            for i, h_idx in enumerate(human_idx):
                if h_idx < predictions["final_boxes"].shape[0]:
                    human_box = predictions["final_boxes"][h_idx]
                    print(f"  Human {i}: box {human_box}")

    # Visualize STTran final detections with bounding boxes
    if "final_boxes" in predictions and predictions["final_boxes"].shape[0] > 0:
        print("\n=== Visualizing STTran Final Detections ===")
        # Load the original image again
        test_image_path = "assets/000165.png"
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            if image is not None:
                orig_h, orig_w = image.shape[:2]
                print(f"Original image size: {orig_w}x{orig_h}")
                # Create a copy for drawing
                vis_image = image.copy()
                # Colors for different classes (same as R-CNN visualization)
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

                final_boxes = predictions["final_boxes"].cpu().numpy()

                # Handle final_labels - might be empty
                if (
                    "final_labels" in predictions
                    and predictions["final_labels"].numel() > 0
                ):
                    final_labels = predictions["final_labels"].cpu().numpy()
                else:
                    # Use pred_labels if final_labels is empty
                    if (
                        "pred_labels" in predictions
                        and predictions["pred_labels"].numel() > 0
                    ):
                        final_labels = predictions["pred_labels"].cpu().numpy()
                    else:
                        # Default to "unknown" class (index 0)
                        final_labels = np.zeros(len(final_boxes), dtype=int)

                # Get scores if available
                if "pred_scores" in predictions:
                    final_scores = predictions["pred_scores"].cpu().numpy()
                elif "scores" in predictions:
                    final_scores = predictions["scores"].cpu().numpy()
                else:
                    final_scores = np.ones(len(final_boxes))  # Default scores

                print(f"Drawing {len(final_boxes)} final bounding boxes...")

                for i, (box, label, score) in enumerate(
                    zip(final_boxes, final_labels, final_scores)
                ):
                    # Box format: [x1, y1, x2, y2] (no batch index in final_boxes)
                    x1, y1, x2, y2 = box
                    # Scale coordinates back to original image size (assuming final boxes are in 600x600 format)
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

                output_path = "assets/sttran_final_detections.png"
                cv2.imwrite(output_path, vis_image)
                print(f"\nSTTran visualization saved to: {output_path}")
                try:
                    cv2.imshow("STTran Final Detections", vis_image)
                    print("Press any key to close the window...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except Exception as e:
                    print(f"Could not display image (no display available): {e}")
                    print("Continuing...")

    # Show relationship class information
    print("\n--- Dataset Relationship Classes ---")
    print(f"Total relationships: {len(dataset.relationship_classes)}")
    print(f"Attention relationships: {len(dataset.attention_relationships)}")
    print(f"Spatial relationships: {len(dataset.spatial_relationships)}")
    print(f"Contact relationships: {len(dataset.contacting_relationships)}")

    # Show some example relationship classes
    if len(dataset.attention_relationships) > 0:
        print(f"Sample attention relationships: {dataset.attention_relationships[:5]}")
    if len(dataset.spatial_relationships) > 0:
        print(f"Sample spatial relationships: {dataset.spatial_relationships[:5]}")
    if len(dataset.contacting_relationships) > 0:
        print(f"Sample contact relationships: {dataset.contacting_relationships[:5]}")

    return predictions


def main():
    print("Starting comprehensive R-CNN backbone and STTran testing...")
    entry = test_rcnn_backbone()
    if entry is not None:
        dataset = AG(
            mode="test",
            datasize="large",
            data_path="data/action_genome",
            filter_nonperson_box_frame=True,
            filter_small_box=False,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        predictions = test_sttran_sgdet(entry, dataset, device)
        if predictions is not None:
            print("All tests completed successfully!")
        else:
            print("STTran testing failed")
    else:
        print("R-CNN backbone testing failed")


if __name__ == "__main__":
    main()
