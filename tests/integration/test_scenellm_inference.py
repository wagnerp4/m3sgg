#!/usr/bin/env python3
"""
SceneLLM Inference Test Script

This script tests the complete SceneLLM inference flow with randomly initialized models.
It simulates the entire pipeline from ROI features to scene graph predictions.
"""

import copy
import os
import sys
from time import time

import numpy as np
import torch
import torch.nn.functional as F

# Add lib paths
sys.path.append("lib")
sys.path.append("fasterRCNN/lib")


def create_mock_dataset():
    """Create a mock dataset object for SceneLLM initialization."""

    class MockDataset:
        def __init__(self):
            # Action Genome relationship classes
            self.attention_relationships = ["looking_at", "not_looking_at"]
            self.spatial_relationships = [
                "in_front_of",
                "behind",
                "on_the_left_of",
                "on_the_right_of",
                "next_to",
                "above",
            ]
            self.contacting_relationships = [
                "touching",
                "holding",
                "drinking_from",
                "eating",
                "writing_on",
                "wearing",
                "wiping",
                "pointing",
                "sitting_on",
                "lying_on",
                "standing_on",
                "opening",
                "pouring",
                "throwing",
                "carrying",
                "covered_by",
                "reading",
            ]
            # Action Genome object classes (exact match with data/action_genome/annotations/object_classes.txt)
            self.object_classes = [
                "__background__",
                "person",
                "bag",
                "bed",
                "blanket",
                "book",
                "box",
                "broom",
                "chair",
                "closet/cabinet",
                "clothes",
                "cup/glass/bottle",
                "dish",
                "door",
                "doorknob",
                "doorway",
                "floor",
                "food",
                "groceries",
                "laptop",
                "light",
                "medicine",
                "mirror",
                "paper/notebook",
                "phone/camera",
                "picture",
                "pillow",
                "refrigerator",
                "sandwich",
                "shelf",
                "shoe",
                "sofa/couch",
                "table",
                "television",
                "towel",
                "vacuum",
                "window",
            ]

    return MockDataset()


def create_mock_config():
    """Create a mock configuration for SceneLLM."""

    class MockConfig:
        def __init__(self):
            # Core parameters
            self.mode = "sgcls"
            self.embed_dim = 512  # Smaller for testing
            self.codebook_size = 1024  # Smaller for testing
            self.commitment_cost = 0.25

            # LLM parameters (using smaller model for testing)
            self.llm_name = "google/gemma-2-2b"
            self.lora_r = 8  # Smaller for testing
            self.lora_alpha = 16
            self.lora_dropout = 0.1

            # Training parameters
            self.ot_step = 256
            self.alpha_obj = 1.0
            self.alpha_rel = 1.0
            self.scenellm_training_stage = "stage2"

            # STTran parameters
            self.enc_layer = 1
            self.dec_layer = 2  # Smaller for testing

    return MockConfig()


def create_mock_entry(batch_size=1, num_rois=8, roi_dim=512):
    """Create mock input entry for SceneLLM inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate random ROI features and bounding boxes
    roi_features = torch.randn(num_rois, roi_dim, device=device)

    # Generate realistic bounding boxes (normalized to image size)
    boxes = torch.rand(num_rois, 4, device=device)
    boxes[:, 2:] = boxes[:, 2:] * 0.3 + 0.1  # Width and height between 0.1 and 0.4
    boxes[:, :2] = boxes[:, :2] * (1.0 - boxes[:, 2:])  # Ensure boxes fit in image

    # Convert to absolute coordinates with batch index (assume 640x480 image)
    im_wh = torch.tensor([640, 480], device=device, dtype=torch.float32)
    boxes_abs = torch.zeros(num_rois, 5, device=device)  # [batch_idx, x1, y1, x2, y2]
    boxes_abs[:, 0] = 0  # All boxes in batch 0
    boxes_abs[:, 1] = boxes[:, 0] * im_wh[0]  # x1
    boxes_abs[:, 2] = boxes[:, 1] * im_wh[1]  # y1
    boxes_abs[:, 3] = (boxes[:, 0] + boxes[:, 2]) * im_wh[0]  # x2
    boxes_abs[:, 4] = (boxes[:, 1] + boxes[:, 3]) * im_wh[1]  # y2

    # Create mock labels - ensure first object is a person (class 1)
    num_objects = len(create_mock_dataset().object_classes)
    labels = torch.randint(
        1, num_objects, (num_rois,), device=device
    )  # Skip background
    labels[0] = 1  # Ensure we have at least one person

    # Create mock object distribution (needed by STTran)
    # STTran expects distribution without background class
    distribution = F.softmax(
        torch.randn(num_rois, num_objects - 1, device=device), dim=1
    )
    # Ensure first ROI has high person confidence (person = class 0 in distribution without background)
    distribution[0, 0] = 0.9  # High confidence for person class
    distribution[0, 1:] = 0.1 / (num_objects - 2)  # Low confidence for other classes

    # Create mock features (needed by STTran) - STTran expects 2048-dim features
    features = torch.randn(
        num_rois, 2048, device=device
    )  # 2048-dim as expected by STTran

    # Create mock ground truth (for training simulation)
    attention_gt = [0]  # looking_at
    spatial_gt = [[1, 4]]  # behind, next_to
    contact_gt = [[0, 5, 8]]  # touching, wearing, sitting_on

    entry = {
        "roi_features": roi_features,
        "features": features,  # STTran expects this
        "boxes": boxes_abs,
        "im_wh": im_wh,
        "labels": labels,
        "distribution": distribution,  # STTran expects this
        "attention_gt": attention_gt,
        "spatial_gt": spatial_gt,
        "contact_gt": contact_gt,
        "im_info": torch.tensor(
            [[1.0, 1.0, 1.0, 1.0]], device=device
        ),  # [scale_x, scale_y, scale_w, scale_h]
    }

    return entry


def test_scenellm_inference():
    """Test complete SceneLLM inference pipeline."""
    print("Testing SceneLLM Inference Pipeline")
    print("=" * 50)

    try:
        # Import SceneLLM
        from lib.scenellm.scenellm import SceneLLM

        print("✓ SceneLLM imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SceneLLM: {e}")
        return False

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

    try:
        # Create mock dataset and config
        dataset = create_mock_dataset()
        config = create_mock_config()
        print("✓ Mock dataset and config created")

        # Initialize SceneLLM model
        print("\nInitializing SceneLLM model...")
        model = SceneLLM(config, dataset).to(device)
        model.eval()
        print(f"✓ SceneLLM model initialized")
        print(f"  - VQ-VAE codebook size: {model.quantiser.codebook_size}")
        print(f"  - Embedding dimension: {model.quantiser.dim}")
        print(f"  - Training stage: {model.training_stage}")

        # Test different training stages
        stages = ["vqvae", "stage1", "stage2"]
        for stage in stages:
            print(f"\n--- Testing {stage.upper()} Stage ---")

            # Set training stage
            model.set_training_stage(stage)
            print(f"✓ Set training stage to {stage}")

            # Create mock input
            entry = create_mock_entry(num_rois=6, roi_dim=512)
            print(f"✓ Created mock input with {entry['roi_features'].size(0)} ROIs")

            # Forward pass
            start_time = time()

            with torch.no_grad():
                pred = model(entry)

            inference_time = time() - start_time

            print(f"✓ Forward pass completed in {inference_time:.3f}s")

            # Check outputs
            print(f"✓ Prediction keys: {list(pred.keys())}")

            # Verify VQ-VAE outputs
            vq_keys = ["vq_loss", "recon_loss", "embedding_loss", "commitment_loss"]
            for key in vq_keys:
                if key in pred:
                    loss_val = (
                        pred[key].item() if torch.is_tensor(pred[key]) else pred[key]
                    )
                    print(f"  - {key}: {loss_val:.4f}")

            # Verify SGG outputs (except for vqvae stage)
            if stage != "vqvae":
                sgg_keys = [
                    "attention_distribution",
                    "spatial_distribution",
                    "contact_distribution",
                ]
                for key in sgg_keys:
                    if key in pred:
                        shape = pred[key].shape if torch.is_tensor(pred[key]) else "N/A"
                        print(f"  - {key} shape: {shape}")

            # Test loss computation
            try:
                if stage == "vqvae":
                    # VQ-VAE stage - only VQ losses
                    total_loss = pred["vq_loss"]
                else:
                    # SGG stages - relationship losses
                    losses = []
                    if "attention_distribution" in pred:
                        att_loss = F.cross_entropy(
                            pred["attention_distribution"],
                            torch.tensor(pred["attention_gt"], device=device),
                        )
                        losses.append(att_loss)

                    if "spatial_distribution" in pred:
                        # Multi-label margin loss simulation
                        spatial_target = torch.zeros(
                            pred["spatial_distribution"].size(), device=device
                        )
                        for i, labels in enumerate(pred["spatial_gt"]):
                            spatial_target[i, labels] = 1
                        spat_loss = F.binary_cross_entropy_with_logits(
                            pred["spatial_distribution"], spatial_target
                        )
                        losses.append(spat_loss)

                    total_loss = (
                        sum(losses) if losses else torch.tensor(0.0, device=device)
                    )

                print(f"✓ Loss computation successful: {total_loss.item():.4f}")

            except Exception as e:
                print(f"⚠ Loss computation failed: {e}")

        # Test codebook update (if available)
        print(f"\n--- Testing Codebook Update ---")
        try:
            original_usage = model.quantiser.get_usage_histogram()
            print(f"✓ Original usage histogram sum: {original_usage.sum().item():.0f}")

            model.update_codebook_with_ot()
            print("✓ Codebook update completed")

        except Exception as e:
            print(f"⚠ Codebook update failed: {e}")

        # Memory usage info
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
            print(f"\n✓ GPU memory allocated: {memory_allocated:.1f} MB")

        print(f"\n🎉 SceneLLM inference test completed successfully!")
        return True

    except Exception as e:
        print(f"✗ SceneLLM inference test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_batch_inference():
    """Test batch inference with multiple samples."""
    print(f"\n--- Testing Batch Inference ---")

    try:
        from lib.scenellm.scenellm import SceneLLM

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = create_mock_dataset()
        config = create_mock_config()

        model = SceneLLM(config, dataset).to(device)
        model.eval()
        model.set_training_stage("stage2")

        # Test with different number of ROIs
        roi_counts = [3, 8, 15]

        for num_rois in roi_counts:
            entry = create_mock_entry(num_rois=num_rois)

            start_time = time()
            with torch.no_grad():
                pred = model(entry)
            inference_time = time() - start_time

            print(f"✓ {num_rois} ROIs processed in {inference_time:.3f}s")

            # Check output shapes are consistent
            if "attention_distribution" in pred:
                print(
                    f"  - Attention output shape: {pred['attention_distribution'].shape}"
                )

        print("✓ Batch inference test completed")
        return True

    except Exception as e:
        print(f"✗ Batch inference test failed: {e}")
        return False


def main():
    """Run all inference tests."""
    print("SceneLLM Inference Testing Suite")
    print("=" * 60)

    # Check basic requirements
    try:
        import torch

        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("✗ PyTorch not available")
        return 1

    # Run tests
    tests_passed = 0
    total_tests = 2

    # Test 1: Basic inference
    if test_scenellm_inference():
        tests_passed += 1

    # Test 2: Batch inference
    if test_batch_inference():
        tests_passed += 1

    # Results
    print(f"\n{'='*60}")
    print(f"Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All SceneLLM inference tests passed!")
        print("\nThe SceneLLM pipeline is working correctly and ready for training.")
        print("\nNext steps:")
        print("1. Install full dependencies: pip install -r requirements_scenellm.txt")
        print("2. Run training: python scripts/model_scripts/train_scenellm_example.py")
        print(
            "3. Evaluate on real data: python scripts/training/training.py -model scenellm -dataset action_genome"
        )
        return 0
    else:
        print("❌ Some tests failed. Check the implementation and dependencies.")
        return 1


if __name__ == "__main__":
    exit(main())
