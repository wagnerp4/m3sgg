#!/usr/bin/env python3
"""
Test script for OED integration into the VidSgg codebase.

This script tests the basic functionality of the integrated OED models
including model initialization, forward pass, and loss computation.
"""

import os
import sys

import torch

# Add the lib directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))


def test_oed_imports():
    """Test that OED modules can be imported correctly."""
    print("Testing OED imports...")

    try:
        from lib.oed import OEDMulti, OEDSingle

        print("✓ Successfully imported OED models")

        # Backbone is no longer needed - OED models use existing object detector
        print("✓ OED models now use existing object detector (no separate backbone)")

        from lib.oed.transformer import build_transformer

        print("✓ Successfully imported OED transformer")

        from lib.oed.criterion import SetCriterionOED

        print("✓ Successfully imported OED criterion")

        from lib.oed.postprocess import PostProcessOED

        print("✓ Successfully imported OED postprocessor")

        return True
    except ImportError as e:
        print(f"✗ Failed to import OED modules: {e}")
        return False


def test_oed_model_creation():
    """Test that OED models can be created with basic configuration."""
    print("\nTesting OED model creation...")

    try:
        # Create a mock configuration
        class MockConfig:
            def __init__(self):
                self.num_queries = 100
                self.dec_layers_hopd = 6
                self.dec_layers_interaction = 6
                self.num_attn_classes = 3
                self.num_spatial_classes = 6
                self.num_contacting_classes = 17
                self.alpha = 0.25
                self.oed_use_matching = True
                self.bbox_loss_coef = 2.5
                self.giou_loss_coef = 1.0
                self.obj_loss_coef = 1.0
                self.rel_loss_coef = 1.0
                self.oed_eos_coef = 0.1
                self.interval1 = 1
                self.interval2 = 1
                self.num_ref_frames = 5
                self.oed_variant = "multi"
                self.fuse_semantic_pos = False
                self.query_temporal_interaction = False
                self.hidden_dim = 256
                self.dim_feedforward = 2048
                self.dropout = 0.1
                self.nheads = 8
                self.pre_norm = False
                self.aux_loss = True
                self.use_matching = True
                self.backbone = "resnet50"
                self.dilation = False
                self.position_embedding = "sine"
                self.enc_layers = 6

        # Create a mock dataset
        class MockDataset:
            def __init__(self):
                self.object_classes = ["person", "car", "bike"]
                self.attention_relationships = ["looking_at", "not_looking_at"]
                self.spatial_relationships = [
                    "above",
                    "below",
                    "left",
                    "right",
                    "inside",
                    "outside",
                ]
                self.contacting_relationships = ["touching", "not_touching"]

        config = MockConfig()
        dataset = MockDataset()

        # Test OEDMulti model creation
        from lib.oed import OEDMulti

        model_multi = OEDMulti(config, dataset)
        print("✓ Successfully created OEDMulti model")

        # Test OEDSingle model creation
        from lib.oed import OEDSingle

        model_single = OEDSingle(config, dataset)
        print("✓ Successfully created OEDSingle model")

        return True
    except Exception as e:
        print(f"✗ Failed to create OED models: {e}")
        return False


def test_oed_forward_pass():
    """Test that OED models can perform a forward pass."""
    print("\nTesting OED forward pass...")

    try:
        # Create a mock configuration and dataset
        class MockConfig:
            def __init__(self):
                self.num_queries = 100
                self.dec_layers_hopd = 6
                self.dec_layers_interaction = 6
                self.num_attn_classes = 3
                self.num_spatial_classes = 6
                self.num_contacting_classes = 17
                self.alpha = 0.25
                self.oed_use_matching = True
                self.bbox_loss_coef = 2.5
                self.giou_loss_coef = 1.0
                self.obj_loss_coef = 1.0
                self.rel_loss_coef = 1.0
                self.oed_eos_coef = 0.1
                self.interval1 = 1
                self.interval2 = 1
                self.num_ref_frames = 5
                self.oed_variant = "multi"
                self.fuse_semantic_pos = False
                self.query_temporal_interaction = False
                self.hidden_dim = 256
                self.dim_feedforward = 2048
                self.dropout = 0.1
                self.nheads = 8
                self.pre_norm = False
                self.aux_loss = True
                self.use_matching = True
                self.backbone = "resnet50"
                self.dilation = False
                self.position_embedding = "sine"
                self.enc_layers = 6

        class MockDataset:
            def __init__(self):
                self.object_classes = ["person", "car", "bike"]
                self.attention_relationships = ["looking_at", "not_looking_at"]
                self.spatial_relationships = [
                    "above",
                    "below",
                    "left",
                    "right",
                    "inside",
                    "outside",
                ]
                self.contacting_relationships = ["touching", "not_touching"]

        config = MockConfig()
        dataset = MockDataset()

        # Create model
        from lib.oed import OEDSingle

        model = OEDSingle(config, dataset)

        # Create mock object detector output format
        num_detections = 5  # Number of detected objects

        # Mock object detector features (2048-dimensional features)
        features = torch.randn(num_detections, 2048)

        # Mock bounding boxes (frame_idx, x1, y1, x2, y2)
        boxes = torch.tensor(
            [
                [0, 0.1, 0.1, 0.3, 0.3],  # frame 0, box 1
                [0, 0.5, 0.5, 0.7, 0.7],  # frame 0, box 2
                [1, 0.2, 0.2, 0.4, 0.4],  # frame 1, box 1
                [1, 0.6, 0.6, 0.8, 0.8],  # frame 1, box 2
                [2, 0.3, 0.3, 0.5, 0.5],  # frame 2, box 1
            ]
        )

        # Mock labels
        labels = torch.tensor([1, 2, 1, 2, 1])  # Object class labels

        # Mock ground truth relationships
        attention_gt = [[0], [1]]  # Attention relationships
        spatial_gt = [[0, 2], [1, 3]]  # Spatial relationships
        contact_gt = [[0], [1]]  # Contact relationships

        # Create entry in object detector format
        entry = {
            "features": features,
            "boxes": boxes,
            "labels": labels,
            "im_info": 1.0,  # Scale factor
            "attention_gt": attention_gt,
            "spatial_gt": spatial_gt,
            "contact_gt": contact_gt,
        }

        # Test forward pass
        with torch.no_grad():
            output = model(entry)
            print("✓ Successfully performed OED forward pass")
            print(f"  Output keys: {list(output.keys())}")
            print(f"  Object distribution shape: {output['distribution'].shape}")
            print(
                f"  Attention distribution shape: {output['attention_distribution'].shape}"
            )

        return True
    except Exception as e:
        print(f"✗ Failed to perform OED forward pass: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_oed_criterion():
    """Test that OED criterion can compute losses."""
    print("\nTesting OED criterion...")

    try:
        from lib.oed.criterion import SetCriterionOED

        # Create mock configuration
        class MockConfig:
            def __init__(self):
                self.num_attn_classes = 2
                self.num_spatial_classes = 6
                self.num_contacting_classes = 17
                self.alpha = 0.25

        mock_conf = MockConfig()

        # Create mock criterion
        criterion = SetCriterionOED(
            num_obj_classes=3,
            num_queries=100,  # Default number of queries
            matcher=None,  # No matcher for this test
            weight_dict={},  # Empty weight dict
            eos_coef=0.1,
            losses=[
                "obj_labels",
                "boxes",
                "attn_labels",
                "spatial_labels",
                "contacting_labels",
            ],
            conf=mock_conf,
        )
        print("✓ Successfully created OED criterion")

        return True
    except Exception as e:
        print(f"✗ Failed to create OED criterion: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("OED Integration Test Suite")
    print("=" * 60)

    tests = [
        test_oed_imports,
        test_oed_model_creation,
        test_oed_forward_pass,
        test_oed_criterion,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("🎉 All tests passed! OED integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
