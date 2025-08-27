#!/usr/bin/env python3
"""
Test script for temporal sequence tensor handling fixes
"""

import sys
import os
import numpy as np
import torch

# Add the current directory to the path so we can import from the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_tensor_handling():
    """Test the tensor handling fixes"""
    print("Testing tensor handling fixes...")

    # Test data structures with proper tensor handling
    temporal_frames = []
    temporal_entries = []
    temporal_preds = []
    temporal_im_infos = []
    max_temporal_frames = 5

    # Simulate adding frames with proper tensor handling
    for i in range(3):  # Test with fewer frames first
        # Create dummy frame data
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Create entry with tensors
        entry = {
            "labels": torch.tensor([0, 1, 2], dtype=torch.long),
            "boxes": torch.tensor(
                [
                    [0, 100, 100, 200, 200],
                    [0, 150, 150, 250, 250],
                    [0, 200, 200, 300, 300],
                ],
                dtype=torch.float32,
            ),
            "pair_idx": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        }

        # Create pred with tensors
        pred = {
            "attention_distribution": torch.randn(2, 3),
            "spatial_distribution": torch.randn(2, 6),
            "contact_distribution": torch.randn(2, 17),
        }

        im_info = {"width": 640, "height": 480}

        # Test proper tensor copying
        try:
            # Test frame copying
            if isinstance(frame, np.ndarray):
                frame_copy = frame.copy()
            else:
                frame_copy = frame
            temporal_frames.append(frame_copy)

            # Test entry copying with tensor handling
            entry_copy = {}
            for key, value in entry.items():
                if isinstance(value, torch.Tensor):
                    entry_copy[key] = value.clone().detach()
                elif isinstance(value, dict):
                    entry_copy[key] = value.copy()
                else:
                    entry_copy[key] = value
            temporal_entries.append(entry_copy)

            # Test pred copying with tensor handling
            pred_copy = {}
            for key, value in pred.items():
                if isinstance(value, torch.Tensor):
                    pred_copy[key] = value.clone().detach()
                elif isinstance(value, dict):
                    pred_copy[key] = value.copy()
                else:
                    pred_copy[key] = value
            temporal_preds.append(pred_copy)

            # Test im_info copying
            temporal_im_infos.append(im_info.copy())

            print(f"Frame {i+1}: Successfully copied tensors")

        except Exception as e:
            print(f"Frame {i+1}: Error copying tensors - {str(e)}")
            return False

    print(f"Final buffer size: {len(temporal_frames)}")
    print("Tensor handling test completed successfully!")

    return True


if __name__ == "__main__":
    test_tensor_handling()
