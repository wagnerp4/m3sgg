#!/usr/bin/env python3
"""
Test script to verify video cycling logic for predcls/sgcls modes
"""

import os
import random
import sys
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_video_cycling_logic():
    """Test the video cycling logic without loading actual models"""

    print("Testing video cycling logic...")

    # Simulate the video cycling parameters
    available_video_indices = [0, 1, 2, 3, 4]  # 5 available videos
    video_cycle_index = 0
    frames_per_video = 30
    frame_count_in_current_video = 0
    current_video_index = available_video_indices[0]
    current_frame_index = 0

    print(f"Available videos: {available_video_indices}")
    print(f"Frames per video: {frames_per_video}")
    print(f"Starting with video {current_video_index}")

    # Simulate processing frames and switching videos
    for frame_num in range(200):  # Test 200 frames
        # Simulate reaching end of video
        if current_frame_index >= 50:  # Assume each video has 50 frames
            # Check if we should switch to next video
            if (
                frame_count_in_current_video >= frames_per_video
                and len(available_video_indices) > 1
            ):
                # Switch to next video in cycle
                video_cycle_index = (video_cycle_index + 1) % len(
                    available_video_indices
                )
                current_video_index = available_video_indices[video_cycle_index]
                current_frame_index = 0
                frame_count_in_current_video = 0
                print(
                    f"Frame {frame_num}: Switched to video {current_video_index} (cycle {video_cycle_index + 1}/{len(available_video_indices)})"
                )
            else:
                # Loop back to start of current video
                current_frame_index = 0
                frame_count_in_current_video = 0
                print(
                    f"Frame {frame_num}: Looped back to start of video {current_video_index}"
                )

        # Process frame
        current_frame_index += 1
        frame_count_in_current_video += 1

        # Show progress every 10 frames
        if frame_num % 10 == 0:
            print(
                f"Frame {frame_num}: Video {current_video_index}, Frame {current_frame_index}, Count {frame_count_in_current_video}/{frames_per_video}"
            )

    print("Video cycling test completed!")
    print("Expected behavior:")
    print("- Should cycle through 5 different videos")
    print("- Each video should show for 30 frames before switching")
    print("- Should loop back to first video after showing all 5 videos")


if __name__ == "__main__":
    test_video_cycling_logic()
