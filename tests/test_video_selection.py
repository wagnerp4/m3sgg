#!/usr/bin/env python3
"""
Test script to verify video selection with ground truth bounding boxes
"""

import os
import random
import sys

from dataloader.action_genome import AG


def test_video_selection():
    """Test video selection with GT bboxes"""
    print("Testing video selection with ground truth bounding boxes...")

    try:
        # Load dataset
        dataset = AG(
            mode="test",
            datasize="large",
            data_path="data/action_genome",
            filter_nonperson_box_frame=True,
            filter_small_box=False,  # For predcls mode
        )

        print(f"Dataset loaded successfully")
        print(f"Total videos: {len(dataset.video_list)}")
        print(f"Total GT annotations: {len(dataset.gt_annotations)}")

        # Find videos with ground truth bounding boxes
        available_videos = []
        for i, video_frames in enumerate(dataset.video_list):
            if i < len(dataset.gt_annotations):
                # Check if this video has ground truth annotations
                video_annotations = dataset.gt_annotations[i]
                if video_annotations and len(video_annotations) > 0:
                    # Check if any frame has bounding boxes
                    has_bboxes = False
                    bbox_count = 0
                    for frame_annots in video_annotations:
                        if frame_annots and len(frame_annots) > 0:
                            for obj_annot in frame_annots:
                                if "bbox" in obj_annot and obj_annot["bbox"]:
                                    has_bboxes = True
                                    bbox_count += 1
                                    break
                        if has_bboxes:
                            break
                    if has_bboxes:
                        available_videos.append(i)
                        print(
                            f"Video {i}: {len(video_frames)} frames, {bbox_count} bboxes"
                        )

        print(f"Found {len(available_videos)} videos with GT bboxes")

        if not available_videos:
            print("Error: No videos with ground truth bounding boxes found")
            return False

        # Select a random video
        selected_video = random.choice(available_videos)
        print(f"Selected video {selected_video}")

        # Test getting data for this video
        try:
            data = dataset.__getitem__(selected_video)
            print(f"Successfully loaded data for video {selected_video}")
            print(
                f"Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}"
            )
            return True
        except Exception as e:
            print(f"Error loading data for video {selected_video}: {str(e)}")
            return False

    except Exception as e:
        print(f"Error in test: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_video_selection()
    if success:
        print("Test PASSED!")
    else:
        print("Test FAILED!")
        sys.exit(1)
