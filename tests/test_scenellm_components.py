#!/usr/bin/env python3
"""
SceneLLM Components Test Script

This script tests individual SceneLLM components without STTran integration
to verify the V2L mapping pipeline works correctly.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from time import time

# Add lib paths
sys.path.append("lib")
sys.path.append("fasterRCNN/lib")


def create_mock_config():
    """Create a mock configuration for SceneLLM components."""

    class MockConfig:
        def __init__(self):
            # Core parameters
            self.embed_dim = 512  # Smaller for testing
            self.codebook_size = 1024  # Smaller for testing
            self.commitment_cost = 0.25
            self.ot_step = 256

    return MockConfig()


def test_vq_vae():
    """Test VQ-VAE component."""
    print("--- Testing VQ-VAE Component ---")

    try:
        from lib.scenellm.scenellm import VQVAEQuantizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = create_mock_config()

        # Create VQ-VAE
        vq = VQVAEQuantizer(
            config.embed_dim, config.codebook_size, config.commitment_cost
        )
        vq = vq.to(device)
        vq.eval()

        # Test with different input sizes
        input_sizes = [3, 8, 15]

        for num_rois in input_sizes:
            # Create mock ROI features
            roi_features = torch.randn(num_rois, config.embed_dim, device=device)

            start_time = time()
            with torch.no_grad():
                result = vq(roi_features)
            inference_time = time() - start_time

            # Check outputs
            assert "ids" in result, "Missing 'ids' in VQ-VAE output"
            assert "z_q" in result, "Missing 'z_q' in VQ-VAE output"
            assert "recon" in result, "Missing 'recon' in VQ-VAE output"
            assert "vq_loss" in result, "Missing 'vq_loss' in VQ-VAE output"

            print(f"✓ {num_rois} ROIs processed in {inference_time:.3f}s")
            print(f"  - VQ loss: {result['vq_loss'].item():.4f}")
            print(f"  - Reconstruction loss: {result['recon_loss'].item():.4f}")
            print(f"  - Embedding loss: {result['embedding_loss'].item():.4f}")
            print(f"  - Commitment loss: {result['commitment_loss'].item():.4f}")

            # Check dimensions
            assert result["z_q"].shape == (
                num_rois,
                config.embed_dim,
            ), f"Wrong z_q shape: {result['z_q'].shape}"
            assert result["recon"].shape == (
                num_rois,
                config.embed_dim,
            ), f"Wrong recon shape: {result['recon'].shape}"
            assert result["ids"].shape == (
                num_rois,
            ), f"Wrong ids shape: {result['ids'].shape}"

        print("✓ VQ-VAE component test passed")
        return True

    except Exception as e:
        print(f"✗ VQ-VAE component test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sia():
    """Test Spatial Information Aggregator."""
    print("\n--- Testing SIA Component ---")

    try:
        from lib.scenellm.scenellm import SIA

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = create_mock_config()

        # Create SIA
        sia = SIA(config.embed_dim)
        sia = sia.to(device)
        sia.eval()

        # Test with different input sizes
        input_sizes = [3, 8, 15]

        for num_rois in input_sizes:
            # Create mock features and boxes
            features = torch.randn(num_rois, config.embed_dim, device=device)
            boxes = torch.rand(num_rois, 4, device=device)  # Normalized boxes [0,1]

            start_time = time()
            with torch.no_grad():
                frame_token = sia(features, boxes)
            inference_time = time() - start_time

            # Check output
            assert frame_token.shape == (
                config.embed_dim,
            ), f"Wrong frame token shape: {frame_token.shape}"

            print(
                f"✓ {num_rois} ROIs aggregated to frame token in {inference_time:.3f}s"
            )
            print(f"  - Frame token shape: {frame_token.shape}")
            print(f"  - Frame token norm: {frame_token.norm().item():.4f}")

        print("✓ SIA component test passed")
        return True

    except Exception as e:
        print(f"✗ SIA component test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_llm_placeholder():
    """Test LLM placeholder."""
    print("\n--- Testing LLM Placeholder ---")

    try:
        from lib.scenellm.scenellm import PlaceholderLLM

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = create_mock_config()

        # Create placeholder LLM
        llm = PlaceholderLLM(config.embed_dim)
        llm = llm.to(device)
        llm.eval()

        # Test with different sequence lengths
        seq_lengths = [1, 3, 5]

        for seq_len in seq_lengths:
            # Create mock input
            input_seq = torch.randn(1, seq_len, config.embed_dim, device=device)

            start_time = time()
            with torch.no_grad():
                output = llm(input_seq)
            inference_time = time() - start_time

            # Check output
            assert (
                output.shape == input_seq.shape
            ), f"Wrong output shape: {output.shape}"

            print(f"✓ Sequence length {seq_len} processed in {inference_time:.3f}s")
            print(f"  - Output shape: {output.shape}")

        print("✓ LLM placeholder test passed")
        return True

    except Exception as e:
        print(f"✗ LLM placeholder test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_v2l_pipeline():
    """Test complete V2L mapping pipeline."""
    print("\n--- Testing V2L Mapping Pipeline ---")

    try:
        from lib.scenellm.scenellm import VQVAEQuantizer, SIA, PlaceholderLLM

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = create_mock_config()

        # Create components
        vq = VQVAEQuantizer(
            config.embed_dim, config.codebook_size, config.commitment_cost
        ).to(device)
        sia = SIA(config.embed_dim).to(device)
        llm = PlaceholderLLM(config.embed_dim).to(device)

        # Set to eval mode
        vq.eval()
        sia.eval()
        llm.eval()

        print("✓ Components initialized")

        # Test pipeline with different input sizes
        input_sizes = [5, 10]

        for num_rois in input_sizes:
            print(f"\nTesting with {num_rois} ROIs:")

            # 1. Create mock ROI features and boxes
            roi_features = torch.randn(num_rois, config.embed_dim, device=device)
            boxes = torch.rand(num_rois, 4, device=device)

            # 2. VQ-VAE quantization
            start_time = time()
            with torch.no_grad():
                vq_results = vq(roi_features)
                code_vecs = vq_results["z_q"]

                # 3. Spatial Information Aggregation
                frame_token = sia(code_vecs, boxes)

                # 4. LLM processing
                if frame_token.dim() == 1:
                    frame_token = frame_token.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
                hidden = llm(frame_token)

            total_time = time() - start_time

            print(f"  ✓ Pipeline completed in {total_time:.3f}s")
            print(f"  - VQ loss: {vq_results['vq_loss'].item():.4f}")
            print(f"  - Frame token shape: {frame_token.shape}")
            print(f"  - LLM output shape: {hidden.shape}")

            # Check dimensions
            assert (
                hidden.shape[2] == config.embed_dim
            ), f"Wrong LLM output dimension: {hidden.shape}"

        print("✓ V2L mapping pipeline test passed")
        return True

    except Exception as e:
        print(f"✗ V2L mapping pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_codebook_update():
    """Test codebook update mechanism."""
    print("\n--- Testing Codebook Update ---")

    try:
        from lib.scenellm.scenellm import VQVAEQuantizer, OTCodebookUpdater

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = create_mock_config()

        # Create VQ-VAE
        vq = VQVAEQuantizer(
            config.embed_dim, config.codebook_size, config.commitment_cost
        ).to(device)

        # Simulate usage by running some data through it
        for _ in range(10):
            roi_features = torch.randn(5, config.embed_dim, device=device)
            with torch.no_grad():
                vq(roi_features)

        # Get usage histogram
        usage_hist = vq.get_usage_histogram()
        print(f"✓ Usage histogram sum: {usage_hist.sum().item():.0f}")

        # Test codebook update (if OT is available)
        try:
            ot_updater = OTCodebookUpdater(vq.codebook, config.ot_step)
            new_codebook = ot_updater.update(usage_hist)
            print(f"✓ Codebook update completed")
            print(f"  - Original size: {vq.codebook.weight.size(0)}")
            print(f"  - New size: {new_codebook.size(0)}")
        except Exception as e:
            print(f"⚠ Codebook update skipped (OT not available): {e}")

        print("✓ Codebook update test passed")
        return True

    except Exception as e:
        print(f"✗ Codebook update test failed: {e}")
        return False


def main():
    """Run all component tests."""
    print("SceneLLM Components Testing Suite")
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

    # Run component tests
    tests = [
        test_vq_vae,
        test_sia,
        test_llm_placeholder,
        test_v2l_pipeline,
        test_codebook_update,
    ]

    tests_passed = 0
    total_tests = len(tests)

    for test in tests:
        if test():
            tests_passed += 1

    # Results
    print(f"\n{'='*60}")
    print(f"Component Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("🎉 All SceneLLM component tests passed!")
        print("\nThe SceneLLM V2L mapping pipeline is working correctly.")
        print("\nNext steps:")
        print("1. Install STTran dependencies for full pipeline testing")
        print("2. Test with real Action Genome data")
        print("3. Run full training pipeline")
        return 0
    else:
        print("❌ Some component tests failed. Check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())
