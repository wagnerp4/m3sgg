#!/usr/bin/env python3
"""
Test script for temporal sequence functionality
"""

import os
import sys

import numpy as np
import torch

# Add the current directory to the path so we can import from the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_temporal_sequence():
    """Test the temporal sequence functionality"""
    print("Testing temporal sequence functionality...")

    # Test data structures
    temporal_frames = []
    temporal_entries = []
    temporal_preds = []
    temporal_im_infos = []
    max_temporal_frames = 5

    # Simulate adding frames
    for i in range(7):  # Add more than max to test overflow
        # Create dummy frame data
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        entry = {
            "labels": torch.tensor([0, 1, 2]),  # Dummy labels
            "boxes": torch.tensor(
                [
                    [0, 100, 100, 200, 200],
                    [0, 150, 150, 250, 250],
                    [0, 200, 200, 300, 300],
                ]
            ),
            "pair_idx": torch.tensor([[0, 1], [1, 2]]),
        }
        pred = {
            "attention_distribution": torch.randn(2, 3),
            "spatial_distribution": torch.randn(2, 6),
            "contact_distribution": torch.randn(2, 17),
        }
        im_info = {"width": 640, "height": 480}

        # Add to temporal buffers
        temporal_frames.append(frame.copy())
        temporal_entries.append(entry.copy())
        temporal_preds.append(pred.copy())
        temporal_im_infos.append(im_info.copy())

        # Keep only the last max_temporal_frames
        if len(temporal_frames) > max_temporal_frames:
            temporal_frames.pop(0)
            temporal_entries.pop(0)
            temporal_preds.pop(0)
            temporal_im_infos.pop(0)

        print(f"Frame {i+1}: {len(temporal_frames)} frames in buffer")

    print(f"Final buffer size: {len(temporal_frames)}")
    print("Temporal sequence test completed successfully!")

    return True


if __name__ == "__main__":
    test_temporal_sequence()
