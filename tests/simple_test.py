#!/usr/bin/env python3
"""
Simple test to check if dataset can be loaded
"""

import os
import sys


def test_dataset_loading():
    """Test if dataset can be loaded"""
    print("Testing dataset loading...")

    try:
        # Check if data directory exists
        if not os.path.exists("data/action_genome"):
            print("Error: data/action_genome directory not found")
            return False

        # Check if required files exist
        required_files = [
            "data/action_genome/annotations/object_classes.txt",
            "data/action_genome/annotations/relationship_classes.txt",
            "data/action_genome/annotations/object_bbox_and_relationship.pkl",
            "data/action_genome/annotations/person_bbox.pkl",
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"Error: Required file not found: {file_path}")
                return False

        print("All required files found")

        # Try to import and load dataset
        try:
            from datasets.action_genome import AG

            print("Successfully imported AG dataset")

            # Try to create dataset instance
            dataset = AG(
                mode="test",
                datasize="large",
                data_path="data/action_genome",
                filter_nonperson_box_frame=True,
                filter_small_box=False,
            )

            print(f"Dataset loaded successfully!")
            print(f"Total videos: {len(dataset.video_list)}")
            print(f"Total GT annotations: {len(dataset.gt_annotations)}")

            return True

        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        print(f"Error in test: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dataset_loading()
    if success:
        print("Test PASSED!")
    else:
        print("Test FAILED!")
        sys.exit(1)
