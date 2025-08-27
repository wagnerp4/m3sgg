#!/usr/bin/env python3
"""
Test script to simulate GUI video selection process
"""

import os
import random
import sys

import torch

from dataloader.action_genome import AG


def test_gui_video_selection():
    """Test the GUI's video selection process"""
    print("Testing GUI video selection process...")

    try:
        # Simulate GUI settings
        model_type = "sttran"
        dataset_type = "action_genome"
        mode = "predcls"

        print(f"Model: {model_type}")
        print(f"Dataset: {dataset_type}")
        print(f"Mode: {mode}")

        # Load dataset (simulating VideoProcessor.setup_models)
        dataset = AG(
            mode="test",
            datasize="large",
            data_path="data/action_genome",
            filter_nonperson_box_frame=True,
            filter_small_box=False if mode == "predcls" else True,
        )

        print(f"Dataset loaded successfully!")
        print(f"Total videos: {len(dataset.video_list)}")
        print(f"Total GT annotations: {len(dataset.gt_annotations)}")

        # Simulate setup_dataset_video method
        print("\n--- Simulating setup_dataset_video ---")

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
                                if (
                                    "bbox" in obj_annot
                                    and obj_annot["bbox"] is not None
                                ):
                                    # Check if bbox has valid coordinates (not all zeros)
                                    bbox = obj_annot["bbox"]
                                    if hasattr(bbox, "any"):  # numpy array
                                        if (
                                            bbox.any()
                                        ):  # Check if any element is non-zero
                                            has_bboxes = True
                                            bbox_count += 1
                                            break
                                    else:  # regular value
                                        if bbox:
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

        # Select a random video from those with GT bboxes
        selected_video = random.choice(available_videos)
        print(
            f"Selected random video {selected_video} from {len(available_videos)} available videos with GT bboxes"
        )

        # Test getting data for this video (simulating process_frame_demo_style)
        print(f"\n--- Testing data loading for video {selected_video} ---")
        try:
            data = dataset.__getitem__(selected_video)
            print(f"Successfully loaded data for video {selected_video}")

            # Check data structure
            if isinstance(data, (list, tuple)) and len(data) >= 5:
                print(f"Data structure: {len(data)} elements")
                print(
                    f"Element 0 (im_data) shape: {data[0].shape if hasattr(data[0], 'shape') else 'No shape'}"
                )
                print(
                    f"Element 1 (im_info) shape: {data[1].shape if hasattr(data[1], 'shape') else 'No shape'}"
                )
                print(
                    f"Element 2 (gt_boxes) shape: {data[2].shape if hasattr(data[2], 'shape') else 'No shape'}"
                )
                print(
                    f"Element 3 (num_boxes) shape: {data[3].shape if hasattr(data[3], 'shape') else 'No shape'}"
                )
                print(f"Element 4 (annotation_id): {data[4]}")
            else:
                print(f"Unexpected data structure: {type(data)}")

            return True

        except Exception as e:
            print(f"Error loading data for video {selected_video}: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        print(f"Error in test: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gui_video_selection()
    if success:
        print("\nTest PASSED!")
    else:
        print("\nTest FAILED!")
        sys.exit(1)
