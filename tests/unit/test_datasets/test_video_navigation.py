#!/usr/bin/env python3
"""
Test script for video navigation functionality
"""

import os
import sys
import time

# Add the current directory to the path so we can import from the project
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .gui import VideoProcessor


def test_video_navigation():
    """Test the video navigation functionality"""
    print("Testing video navigation functionality...")

    # Test parameters
    model_type = "sttran"
    dataset_type = "action_genome"
    mode = "predcls"

    # Create video processor
    video_processor = VideoProcessor(
        None, model_type, dataset_type, mode, checkpoint_path=None
    )

    # Setup models (without checkpoint for testing)
    print("Setting up models...")
    if video_processor.setup_models():
        print("✓ Models setup successful")

        # Test video selection
        print("\nTesting video selection...")
        if video_processor.setup_dataset_video():
            print("✓ Video selection successful")
            print(f"Selected video index: {video_processor.current_video_index}")

            # Test frame navigation
            print("\n=== Testing frame navigation ===")

            # Get initial frame info
            initial_frame_info = video_processor.get_frame_info()
            print(f"Initial frame info: {initial_frame_info}")

            # Test next frame navigation
            print("\nTesting next frame navigation...")
            for i in range(5):
                if video_processor.navigate_frame("next"):
                    frame_info = video_processor.get_frame_info()
                    print(f"  Navigated to next frame: {frame_info}")
                else:
                    print(f"  Could not navigate to next frame (iteration {i})")
                    break

            # Test previous frame navigation
            print("\nTesting previous frame navigation...")
            for i in range(3):
                if video_processor.navigate_frame("previous"):
                    frame_info = video_processor.get_frame_info()
                    print(f"  Navigated to previous frame: {frame_info}")
                else:
                    print(f"  Could not navigate to previous frame (iteration {i})")
                    break

            # Test first frame navigation
            print("\nTesting first frame navigation...")
            if video_processor.navigate_frame("first"):
                frame_info = video_processor.get_frame_info()
                print(f"  Navigated to first frame: {frame_info}")
            else:
                print("  Could not navigate to first frame")

            # Test last frame navigation
            print("\nTesting last frame navigation...")
            if video_processor.navigate_frame("last"):
                frame_info = video_processor.get_frame_info()
                print(f"  Navigated to last frame: {frame_info}")
            else:
                print("  Could not navigate to last frame")

            # Test video cycling
            print("\n=== Testing video cycling ===")
            video_processor._initialize_video_cycling()
            if hasattr(video_processor, "available_video_indices"):
                print(
                    f"Available videos for cycling: {len(video_processor.available_video_indices)}"
                )

                # Test skip to next video
                print("\nTesting skip to next video...")
                if video_processor.skip_to_next_video():
                    print(
                        f"  Skipped to next video: {video_processor.current_video_index}"
                    )
                    frame_info = video_processor.get_frame_info()
                    print(f"  New video frame info: {frame_info}")
                else:
                    print("  Could not skip to next video")
            else:
                print("  No video cycling available")

            # Test manual navigation mode
            print("\n=== Testing manual navigation mode ===")
            video_processor.set_manual_navigation(True)
            print("  Manual navigation enabled")

            # Test frame navigation in manual mode
            print("\nTesting frame navigation in manual mode...")
            for i in range(3):
                if video_processor.navigate_frame("next"):
                    frame_info = video_processor.get_frame_info()
                    print(f"  Manual navigation to next frame: {frame_info}")
                else:
                    print(f"  Could not navigate manually (iteration {i})")
                    break

        else:
            print("✗ Video selection failed")
    else:
        print("✗ Models setup failed")

    print("\n=== Test completed ===")


def test_navigation_edge_cases():
    """Test navigation edge cases"""
    print("\nTesting navigation edge cases...")

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

    # Setup dataset video
    if not video_processor.setup_dataset_video():
        print("✗ Video selection failed")
        return

    print("✓ Setup successful")

    # Test edge cases
    print("\nTesting edge cases...")

    # Test navigation with invalid directions
    invalid_directions = ["invalid", "random", "unknown"]
    for direction in invalid_directions:
        result = video_processor.navigate_frame(direction)
        print(
            f"  Navigation with '{direction}': {'Success' if result else 'Failed (expected)'}"
        )

    # Test navigation when at boundaries
    print("\nTesting boundary navigation...")

    # Go to first frame
    video_processor.navigate_frame("first")
    frame_info = video_processor.get_frame_info()
    print(f"  At first frame: {frame_info}")

    # Try to go previous (should fail)
    result = video_processor.navigate_frame("previous")
    print(
        f"  Try to go previous from first frame: {'Success' if result else 'Failed (expected)'}"
    )

    # Go to last frame
    video_processor.navigate_frame("last")
    frame_info = video_processor.get_frame_info()
    print(f"  At last frame: {frame_info}")

    # Try to go next (should fail)
    result = video_processor.navigate_frame("next")
    print(
        f"  Try to go next from last frame: {'Success' if result else 'Failed (expected)'}"
    )

    print("\n=== Edge case tests completed ===")


if __name__ == "__main__":
    print("Starting video navigation tests...")

    try:
        test_video_navigation()
        test_navigation_edge_cases()
        print("\n✓ All tests completed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
