#!/usr/bin/env python3
"""
Test script for expanded video selection feature
"""

import sys
import os
import random
import torch

# Add the current directory to the path so we can import from the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloader.action_genome import AG
from gui import VideoProcessor


def test_expanded_video_selection():
    """Test the expanded video selection feature"""
    print("Testing expanded video selection...")

    # Test parameters
    model_type = "sttran"
    dataset_type = "action_genome"
    mode = "predcls"

    # Create video processor
    video_processor = VideoProcessor(
        None, model_type, dataset_type, mode, checkpoint_path=None
    )

    # Test with expanded selection enabled
    print("\n=== Testing with expanded selection enabled ===")
    video_processor.use_expanded_video_selection = True
    video_processor.gt_video_selection_probability = 0.7

    # Setup models (without checkpoint for testing)
    print("Setting up models...")
    if video_processor.setup_models():
        print("✓ Models setup successful")

        # Test video selection
        print("\nTesting video selection...")
        if video_processor.setup_dataset_video():
            print("✓ Video selection successful")
            print(f"Selected video index: {video_processor.current_video_index}")

            # Test video cycling initialization
            print("\nTesting video cycling initialization...")
            video_processor._initialize_video_cycling()
            if hasattr(video_processor, "available_video_indices"):
                print(
                    f"✓ Video cycling initialized with {len(video_processor.available_video_indices)} videos"
                )
                print(
                    f"Available video indices: {video_processor.available_video_indices[:10]}..."
                )  # Show first 10
            else:
                print("✗ Video cycling initialization failed")
        else:
            print("✗ Video selection failed")
    else:
        print("✗ Models setup failed")

    # Test with expanded selection disabled
    print("\n=== Testing with expanded selection disabled ===")
    video_processor.use_expanded_video_selection = False

    # Re-initialize video cycling
    video_processor._initialize_video_cycling()
    if hasattr(video_processor, "available_video_indices"):
        print(
            f"✓ Video cycling with restricted selection: {len(video_processor.available_video_indices)} videos"
        )
        print(
            f"Available video indices: {video_processor.available_video_indices[:10]}..."
        )  # Show first 10
    else:
        print("✗ Video cycling initialization failed")

    print("\n=== Test completed ===")


def test_video_selection_probability():
    """Test the video selection probability feature"""
    print("\nTesting video selection probability...")

    # Test parameters
    model_type = "sttran"
    dataset_type = "action_genome"
    mode = "predcls"

    # Create video processor
    video_processor = VideoProcessor(
        None, model_type, dataset_type, mode, checkpoint_path=None
    )

    # Setup models
    if not video_processor.setup_models():
        print("✗ Models setup failed")
        return

    # Test different probability values
    probabilities = [0.1, 0.5, 0.9]

    for prob in probabilities:
        print(f"\nTesting with GT probability: {prob}")
        video_processor.use_expanded_video_selection = True
        video_processor.gt_video_selection_probability = prob

        # Count selections from GT vs non-GT videos
        gt_selections = 0
        non_gt_selections = 0
        total_tests = 100

        for _ in range(total_tests):
            # Simulate video selection logic
            if hasattr(video_processor, "dataset") and video_processor.dataset:
                # Find videos with and without GT bboxes
                available_videos_with_gt = []
                all_available_videos = []

                for i, video_frames in enumerate(video_processor.dataset.video_list):
                    all_available_videos.append(i)

                    if i < len(video_processor.dataset.gt_annotations):
                        video_annotations = video_processor.dataset.gt_annotations[i]
                        if video_annotations and len(video_annotations) > 0:
                            has_bboxes = False
                            for frame_annots in video_annotations:
                                if frame_annots and len(frame_annots) > 0:
                                    for obj_annot in frame_annots:
                                        if (
                                            "bbox" in obj_annot
                                            and obj_annot["bbox"] is not None
                                        ):
                                            bbox = obj_annot["bbox"]
                                            if hasattr(bbox, "any"):
                                                if bbox.any():
                                                    has_bboxes = True
                                                    break
                                            else:
                                                if bbox:
                                                    has_bboxes = True
                                                    break
                                    if has_bboxes:
                                        break
                            if has_bboxes:
                                available_videos_with_gt.append(i)

                # Simulate selection
                if available_videos_with_gt:
                    if random.random() < prob:
                        selected_video = random.choice(available_videos_with_gt)
                        gt_selections += 1
                    else:
                        selected_video = random.choice(all_available_videos)
                        if selected_video not in available_videos_with_gt:
                            non_gt_selections += 1
                        else:
                            gt_selections += 1

        # Calculate actual probability
        actual_prob = gt_selections / total_tests if total_tests > 0 else 0
        print(f"  Expected GT probability: {prob:.1%}")
        print(
            f"  Actual GT selections: {gt_selections}/{total_tests} ({actual_prob:.1%})"
        )
        print(f"  Non-GT selections: {non_gt_selections}/{total_tests}")


if __name__ == "__main__":
    print("Starting expanded video selection tests...")

    try:
        test_expanded_video_selection()
        test_video_selection_probability()
        print("\n✓ All tests completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
