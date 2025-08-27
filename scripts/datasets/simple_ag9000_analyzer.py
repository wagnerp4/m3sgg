#!/usr/bin/env python3
"""
Simple Action Genome 9000 Dataset Statistics Analyzer

This script provides comprehensive statistics for the action_genome9000 dataset
including video samples, frames, objects, relationships, and other metrics.
"""

import json
import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def load_object_classes(annotations_path):
    """Load object class definitions"""
    object_classes = ["__background__"]
    with open(annotations_path / "object_classes.txt", "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            object_classes.append(line)

    # Apply corrections as in the dataloader
    corrections = {
        9: "closet/cabinet",
        11: "cup/glass/bottle",
        23: "paper/notebook",
        24: "phone/camera",
        31: "sofa/couch",
    }
    for idx, corrected_name in corrections.items():
        if idx < len(object_classes):
            object_classes[idx] = corrected_name

    return object_classes


def load_relationship_classes(annotations_path):
    """Load relationship class definitions"""
    relationship_classes = []
    with open(annotations_path / "relationship_classes.txt", "r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            relationship_classes.append(line)

    # Apply corrections as in the dataloader
    corrections = {
        0: "looking_at",
        1: "not_looking_at",
        5: "in_front_of",
        7: "on_the_side_of",
        10: "covered_by",
        11: "drinking_from",
        13: "have_it_on_the_back",
        15: "leaning_on",
        16: "lying_on",
        17: "not_contacting",
        18: "other_relationship",
        19: "sitting_on",
        20: "standing_on",
        25: "writing_on",
    }
    for idx, corrected_name in corrections.items():
        if idx < len(relationship_classes):
            relationship_classes[idx] = corrected_name

    return relationship_classes


def analyze_action_genome9000(data_path="data/action_genome9000"):
    """Analyze Action Genome 9000 dataset statistics"""

    print("=" * 80)
    print("ACTION GENOME 9000 DATASET ANALYSIS")
    print("=" * 80)

    data_path = Path(data_path)
    annotations_path = data_path / "annotations"
    frames_path = data_path / "frames"

    # Check if paths exist
    if not data_path.exists():
        print(f"❌ Error: Data path '{data_path}' does not exist!")
        return None

    if not annotations_path.exists():
        print(f"❌ Error: Annotations path '{annotations_path}' does not exist!")
        return None

    print(f"📁 Data path: {data_path}")
    print(f"📁 Annotations path: {annotations_path}")
    print(f"📁 Frames path: {frames_path}")

    # Load class definitions
    print("\n1. LOADING CLASS DEFINITIONS")
    print("-" * 40)

    object_classes = load_object_classes(annotations_path)
    relationship_classes = load_relationship_classes(annotations_path)

    print(f"Object classes: {len(object_classes) - 1} (excluding background)")
    print(f"Relationship classes: {len(relationship_classes)}")

    # Load annotations
    print("\n2. LOADING ANNOTATIONS")
    print("-" * 40)

    try:
        with open(annotations_path / "person_bbox.pkl", "rb") as f:
            person_bbox = pickle.load(f)
        print(f"✅ Loaded person bbox annotations: {len(person_bbox):,} entries")
    except Exception as e:
        print(f"❌ Error loading person bbox: {e}")
        return None

    try:
        with open(annotations_path / "object_bbox_and_relationship.pkl", "rb") as f:
            object_bbox = pickle.load(f)
        print(f"✅ Loaded object bbox annotations: {len(object_bbox):,} entries")
    except Exception as e:
        print(f"❌ Error loading object bbox: {e}")
        return None

    try:
        frame_list = []
        with open(annotations_path / "frame_list.txt", "r") as f:
            for line in f.readlines():
                frame_list.append(line.strip())
        print(f"✅ Loaded frame list: {len(frame_list):,} frames")
    except Exception as e:
        print(f"❌ Error loading frame list: {e}")
        return None

    # Analyze basic structure
    print("\n3. BASIC DATASET STRUCTURE")
    print("-" * 40)

    # Count video directories
    video_dirs = []
    if frames_path.exists():
        video_dirs = [d for d in frames_path.iterdir() if d.is_dir()]

    print(f"Total video directories: {len(video_dirs):,}")
    print(f"Total frames in annotation list: {len(frame_list):,}")
    print(f"Person bbox entries: {len(person_bbox):,}")
    print(f"Object bbox entries: {len(object_bbox):,}")

    # Analyze video and frame statistics
    print("\n4. VIDEO AND FRAME STATISTICS")
    print("-" * 40)

    # Group frames by video
    video_frames = defaultdict(list)
    for frame_path in frame_list:
        if "/" in frame_path:
            video_name, frame_name = frame_path.split("/", 1)
            video_frames[video_name].append(frame_path)

    video_stats = []
    for video_name, frames in video_frames.items():
        video_stats.append({"video_name": video_name, "frame_count": len(frames)})

    # Sort by frame count
    video_stats.sort(key=lambda x: x["frame_count"], reverse=True)
    frame_counts = [v["frame_count"] for v in video_stats]

    print(f"Total videos: {len(video_stats):,}")
    print(f"Total frames: {sum(frame_counts):,}")
    print(f"Average frames per video: {np.mean(frame_counts):.2f}")
    print(f"Median frames per video: {np.median(frame_counts):.2f}")
    print(f"Min frames per video: {min(frame_counts)}")
    print(f"Max frames per video: {max(frame_counts)}")
    print(f"Std frames per video: {np.std(frame_counts):.2f}")

    # Frame distribution
    videos_1_frame = sum(1 for c in frame_counts if c == 1)
    videos_2_5_frames = sum(1 for c in frame_counts if 2 <= c <= 5)
    videos_6_10_frames = sum(1 for c in frame_counts if 6 <= c <= 10)
    videos_11_20_frames = sum(1 for c in frame_counts if 11 <= c <= 20)
    videos_21_50_frames = sum(1 for c in frame_counts if 21 <= c <= 50)
    videos_50_plus_frames = sum(1 for c in frame_counts if c > 50)

    print("\nFrame distribution:")
    print(f"  1 frame: {videos_1_frame:,} videos")
    print(f"  2-5 frames: {videos_2_5_frames:,} videos")
    print(f"  6-10 frames: {videos_6_10_frames:,} videos")
    print(f"  11-20 frames: {videos_11_20_frames:,} videos")
    print(f"  21-50 frames: {videos_21_50_frames:,} videos")
    print(f"  50+ frames: {videos_50_plus_frames:,} videos")

    # Analyze object statistics
    print("\n5. OBJECT STATISTICS")
    print("-" * 40)

    object_counts = defaultdict(int)
    visible_objects = 0
    total_objects = 0

    for frame_path, objects in object_bbox.items():
        for obj in objects:
            total_objects += 1
            if obj["visible"]:
                visible_objects += 1
                object_counts[obj["class"]] += 1

    # Sort objects by frequency
    sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"Total object instances: {total_objects:,}")
    print(f"Visible object instances: {visible_objects:,}")
    print(f"Object visibility rate: {visible_objects / total_objects:.3f}")
    print(f"Unique object types: {len(object_counts)}")

    print("\nMost common objects:")
    for i, (obj_class, count) in enumerate(sorted_objects[:10]):
        print(f"  {i+1:2d}. {obj_class:20s}: {count:6,} instances")

    # Analyze relationship statistics
    print("\n6. RELATIONSHIP STATISTICS")
    print("-" * 40)

    # Categorize relationships
    attention_relationships = relationship_classes[
        0:3
    ]  # looking_at, not_looking_at, unsure
    spatial_relationships = relationship_classes[
        3:9
    ]  # above, beneath, in_front_of, behind, on_the_side_of, in
    contacting_relationships = relationship_classes[9:]  # carrying, covered_by, etc.

    relationship_counts = defaultdict(int)
    attention_counts = defaultdict(int)
    spatial_counts = defaultdict(int)
    contacting_counts = defaultdict(int)

    for frame_path, objects in object_bbox.items():
        for obj in objects:
            if obj["visible"]:
                # Count attention relationships
                for rel in obj.get("attention_relationship", []):
                    if rel in attention_relationships:
                        attention_counts[rel] += 1
                        relationship_counts[rel] += 1

                # Count spatial relationships
                for rel in obj.get("spatial_relationship", []):
                    if rel in spatial_relationships:
                        spatial_counts[rel] += 1
                        relationship_counts[rel] += 1

                # Count contacting relationships
                for rel in obj.get("contacting_relationship", []):
                    if rel in contacting_relationships:
                        contacting_counts[rel] += 1
                        relationship_counts[rel] += 1

    # Sort relationships by frequency
    sorted_relationships = sorted(
        relationship_counts.items(), key=lambda x: x[1], reverse=True
    )
    sorted_attention = sorted(
        attention_counts.items(), key=lambda x: x[1], reverse=True
    )
    sorted_spatial = sorted(spatial_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_contacting = sorted(
        contacting_counts.items(), key=lambda x: x[1], reverse=True
    )

    print(f"Total relationships: {sum(relationship_counts.values()):,}")
    print(f"Attention relationships: {len(attention_relationships)}")
    print(f"Spatial relationships: {len(spatial_relationships)}")
    print(f"Contacting relationships: {len(contacting_relationships)}")

    print("\nMost common relationships:")
    for i, (rel, count) in enumerate(sorted_relationships[:10]):
        print(f"  {i+1:2d}. {rel:20s}: {count:6,} instances")

    print("\nMost common attention relationships:")
    for rel, count in sorted_attention:
        print(f"  - {rel:20s}: {count:6,} instances")

    print("\nMost common spatial relationships:")
    for rel, count in sorted_spatial:
        print(f"  - {rel:20s}: {count:6,} instances")

    print("\nMost common contacting relationships:")
    for i, (rel, count) in enumerate(sorted_contacting[:10]):
        print(f"  {i+1:2d}. {rel:20s}: {count:6,} instances")

    # Analyze temporal statistics
    print("\n7. TEMPORAL STATISTICS")
    print("-" * 40)

    # Analyze temporal patterns
    video_temporal_stats = []

    for frame_path, objects in object_bbox.items():
        if "/" in frame_path:
            video_name, frame_name = frame_path.split("/", 1)

            # Extract frame number if possible
            frame_num = 0
            try:
                if frame_name.endswith(".jpg") or frame_name.endswith(".png"):
                    frame_num = int(frame_name.split(".")[0])
            except:
                pass

            # Count objects and relationships in this frame
            visible_objects = sum(1 for obj in objects if obj["visible"])
            total_relationships = 0
            for obj in objects:
                if obj["visible"]:
                    total_relationships += len(obj.get("attention_relationship", []))
                    total_relationships += len(obj.get("spatial_relationship", []))
                    total_relationships += len(obj.get("contacting_relationship", []))

            video_temporal_stats.append(
                {
                    "video": video_name,
                    "frame": frame_name,
                    "frame_num": frame_num,
                    "visible_objects": visible_objects,
                    "total_relationships": total_relationships,
                }
            )

    # Overall temporal statistics
    all_object_counts = [f["visible_objects"] for f in video_temporal_stats]
    all_relationship_counts = [f["total_relationships"] for f in video_temporal_stats]

    print(f"Total frames analyzed: {len(video_temporal_stats):,}")
    print(f"Average objects per frame: {np.mean(all_object_counts):.2f}")
    print(f"Std objects per frame: {np.std(all_object_counts):.2f}")
    print(f"Min objects per frame: {min(all_object_counts)}")
    print(f"Max objects per frame: {max(all_object_counts)}")
    print(f"Average relationships per frame: {np.mean(all_relationship_counts):.2f}")
    print(f"Std relationships per frame: {np.std(all_relationship_counts):.2f}")
    print(f"Min relationships per frame: {min(all_relationship_counts)}")
    print(f"Max relationships per frame: {max(all_relationship_counts)}")

    # Analyze spatial statistics
    print("\n8. SPATIAL STATISTICS")
    print("-" * 40)

    # Analyze bounding box statistics
    bbox_areas = []
    bbox_aspect_ratios = []

    for frame_path, objects in object_bbox.items():
        for obj in objects:
            if obj["visible"] and "bbox" in obj and obj["bbox"] is not None:
                bbox = obj["bbox"]
                if len(bbox) == 4:  # [x1, y1, x2, y2]
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 0

                    bbox_areas.append(area)
                    bbox_aspect_ratios.append(aspect_ratio)

    # Analyze person bounding boxes
    person_bbox_areas = []
    person_bbox_aspect_ratios = []

    for frame_path, person_data in person_bbox.items():
        if "bbox" in person_data and person_data["bbox"] is not None:
            bbox = person_data["bbox"]
            if len(bbox.shape) > 1 and bbox.shape[0] > 0:
                for i in range(bbox.shape[0]):
                    width = bbox[i, 2] - bbox[i, 0]
                    height = bbox[i, 3] - bbox[i, 1]
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 0

                    person_bbox_areas.append(area)
                    person_bbox_aspect_ratios.append(aspect_ratio)

    print(f"Total bounding boxes: {len(bbox_areas):,}")
    print(f"Total person bounding boxes: {len(person_bbox_areas):,}")
    print(f"Average bbox area: {np.mean(bbox_areas):.2f}")
    print(f"Std bbox area: {np.std(bbox_areas):.2f}")
    print(f"Min bbox area: {min(bbox_areas):.2f}")
    print(f"Max bbox area: {max(bbox_areas):.2f}")
    print(f"Average bbox aspect ratio: {np.mean(bbox_aspect_ratios):.3f}")
    print(f"Std bbox aspect ratio: {np.std(bbox_aspect_ratios):.3f}")
    print(f"Average person bbox area: {np.mean(person_bbox_areas):.2f}")
    print(f"Std person bbox area: {np.std(person_bbox_areas):.2f}")
    print(f"Average person bbox aspect ratio: {np.mean(person_bbox_aspect_ratios):.3f}")
    print(f"Std person bbox aspect ratio: {np.std(person_bbox_aspect_ratios):.3f}")

    # Comprehensive summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DATASET SUMMARY")
    print("=" * 80)

    print(f"\nDataset: Action Genome 9000")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data Path: {data_path}")

    print(f"\n📊 DATASET OVERVIEW:")
    print(f"  • Total Videos: {len(video_stats):,}")
    print(f"  • Total Frames: {sum(frame_counts):,}")
    print(f"  • Object Classes: {len(object_classes) - 1}")
    print(f"  • Relationship Classes: {len(relationship_classes)}")

    print(f"\n🎬 VIDEO STATISTICS:")
    print(f"  • Average Frames per Video: {np.mean(frame_counts):.2f}")
    print(f"  • Median Frames per Video: {np.median(frame_counts):.2f}")
    print(f"  • Frame Range: {min(frame_counts)} - {max(frame_counts)}")

    print(f"\n🎯 OBJECT STATISTICS:")
    print(f"  • Total Object Instances: {total_objects:,}")
    print(f"  • Visible Object Instances: {visible_objects:,}")
    print(f"  • Object Visibility Rate: {visible_objects / total_objects:.1%}")
    print(f"  • Unique Object Types: {len(object_counts)}")

    print(f"\n🔗 RELATIONSHIP STATISTICS:")
    print(f"  • Total Relationships: {sum(relationship_counts.values()):,}")
    print(f"  • Attention Relationships: {len(attention_relationships)}")
    print(f"  • Spatial Relationships: {len(spatial_relationships)}")
    print(f"  • Contacting Relationships: {len(contacting_relationships)}")

    print(f"\n⏱️  TEMPORAL STATISTICS:")
    print(f"  • Average Objects per Frame: {np.mean(all_object_counts):.2f}")
    print(
        f"  • Average Relationships per Frame: {np.mean(all_relationship_counts):.2f}"
    )
    print(f"  • Frames Analyzed: {len(video_temporal_stats):,}")

    print(f"\n📐 SPATIAL STATISTICS:")
    print(f"  • Total Bounding Boxes: {len(bbox_areas):,}")
    print(f"  • Total Person Bounding Boxes: {len(person_bbox_areas):,}")
    print(f"  • Average Bbox Area: {np.mean(bbox_areas):.2f}")
    print(f"  • Average Bbox Aspect Ratio: {np.mean(bbox_aspect_ratios):.3f}")

    print(f"\n📈 DISTRIBUTION BREAKDOWN:")
    print(f"  • Videos with 1 frame: {videos_1_frame:,}")
    print(f"  • Videos with 2-5 frames: {videos_2_5_frames:,}")
    print(f"  • Videos with 6-10 frames: {videos_6_10_frames:,}")
    print(f"  • Videos with 11-20 frames: {videos_11_20_frames:,}")
    print(f"  • Videos with 21-50 frames: {videos_21_50_frames:,}")
    print(f"  • Videos with 50+ frames: {videos_50_plus_frames:,}")

    print(f"\n🏆 TOP 5 MOST COMMON OBJECTS:")
    for i, (obj_class, count) in enumerate(sorted_objects[:5]):
        print(f"  {i+1}. {obj_class:20s}: {count:6,} instances")

    print(f"\n🏆 TOP 5 MOST COMMON RELATIONSHIPS:")
    for i, (rel, count) in enumerate(sorted_relationships[:5]):
        print(f"  {i+1}. {rel:20s}: {count:6,} instances")

    print("\n" + "=" * 80)

    # Return comprehensive statistics
    stats = {
        "dataset_info": {
            "name": "Action Genome 9000",
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_path": str(data_path),
        },
        "basic_structure": {
            "total_videos": len(video_stats),
            "total_frames": sum(frame_counts),
            "object_classes": len(object_classes) - 1,
            "relationship_classes": len(relationship_classes),
            "person_bbox_entries": len(person_bbox),
            "object_bbox_entries": len(object_bbox),
        },
        "video_frame_stats": {
            "avg_frames_per_video": float(np.mean(frame_counts)),
            "median_frames_per_video": float(np.median(frame_counts)),
            "min_frames_per_video": min(frame_counts),
            "max_frames_per_video": max(frame_counts),
            "std_frames_per_video": float(np.std(frame_counts)),
            "frame_distribution": {
                "videos_with_1_frame": videos_1_frame,
                "videos_with_2_5_frames": videos_2_5_frames,
                "videos_with_6_10_frames": videos_6_10_frames,
                "videos_with_11_20_frames": videos_11_20_frames,
                "videos_with_21_50_frames": videos_21_50_frames,
                "videos_with_50_plus_frames": videos_50_plus_frames,
            },
        },
        "object_stats": {
            "total_object_instances": total_objects,
            "visible_object_instances": visible_objects,
            "visibility_rate": float(visible_objects / total_objects),
            "unique_object_types": len(object_counts),
            "most_common_objects": sorted_objects[:10],
        },
        "relationship_stats": {
            "total_relationships": sum(relationship_counts.values()),
            "attention_relationships": len(attention_relationships),
            "spatial_relationships": len(spatial_relationships),
            "contacting_relationships": len(contacting_relationships),
            "most_common_relationships": sorted_relationships[:10],
        },
        "temporal_stats": {
            "total_frames_analyzed": len(video_temporal_stats),
            "avg_objects_per_frame": float(np.mean(all_object_counts)),
            "std_objects_per_frame": float(np.std(all_object_counts)),
            "min_objects_per_frame": min(all_object_counts),
            "max_objects_per_frame": max(all_object_counts),
            "avg_relationships_per_frame": float(np.mean(all_relationship_counts)),
            "std_relationships_per_frame": float(np.std(all_relationship_counts)),
            "min_relationships_per_frame": min(all_relationship_counts),
            "max_relationships_per_frame": max(all_relationship_counts),
        },
        "spatial_stats": {
            "total_bboxes": len(bbox_areas),
            "total_person_bboxes": len(person_bbox_areas),
            "avg_bbox_area": float(np.mean(bbox_areas)),
            "std_bbox_area": float(np.std(bbox_areas)),
            "min_bbox_area": float(min(bbox_areas)),
            "max_bbox_area": float(max(bbox_areas)),
            "avg_bbox_aspect_ratio": float(np.mean(bbox_aspect_ratios)),
            "std_bbox_aspect_ratio": float(np.std(bbox_aspect_ratios)),
            "avg_person_bbox_area": float(np.mean(person_bbox_areas)),
            "std_person_bbox_area": float(np.std(person_bbox_areas)),
            "avg_person_bbox_aspect_ratio": float(np.mean(person_bbox_aspect_ratios)),
            "std_person_bbox_aspect_ratio": float(np.std(person_bbox_aspect_ratios)),
        },
    }

    return stats


def save_stats(stats, output_path="action_genome9000_stats.json"):
    """Save statistics to JSON file"""
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\n📊 Statistics saved to: {output_path}")


def main():
    """Main function"""
    data_path = "data/action_genome9000"

    print("🚀 Starting Action Genome 9000 Dataset Analysis...")

    try:
        # Perform analysis
        stats = analyze_action_genome9000(data_path)

        if stats:
            # Save results
            save_stats(stats)

            print(f"\n✅ Analysis completed successfully!")
            print(f"📊 Statistics saved to: action_genome9000_stats.json")
        else:
            print("❌ Analysis failed!")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
