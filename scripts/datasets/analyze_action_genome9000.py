#!/usr/bin/env python3
"""
Action Genome 9000 Dataset Statistics Analyzer

This script provides comprehensive statistics for the action_genome9000 dataset
including video samples, frames, objects, relationships, and other metrics.
"""

import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


class ActionGenomeAnalyzer:
    def __init__(self, data_path="data/action_genome9000"):
        self.data_path = Path(data_path)
        self.annotations_path = self.data_path / "annotations"
        self.frames_path = self.data_path / "frames"
        self.videos_path = self.data_path / "videos"

        # Load class definitions
        self.object_classes = self._load_object_classes()
        self.relationship_classes = self._load_relationship_classes()

        # Load annotations
        self.person_bbox = self._load_person_bbox()
        self.object_bbox = self._load_object_bbox()
        self.frame_list = self._load_frame_list()

        # Statistics containers
        self.stats = {}

    def _load_object_classes(self):
        """Load object class definitions"""
        object_classes = ["__background__"]
        with open(self.annotations_path / "object_classes.txt", "r") as f:
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

    def _load_relationship_classes(self):
        """Load relationship class definitions"""
        relationship_classes = []
        with open(self.annotations_path / "relationship_classes.txt", "r") as f:
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

    def _load_person_bbox(self):
        """Load person bounding box annotations"""
        with open(self.annotations_path / "person_bbox.pkl", "rb") as f:
            return pickle.load(f)

    def _load_object_bbox(self):
        """Load object bounding box and relationship annotations"""
        with open(
            self.annotations_path / "object_bbox_and_relationship.pkl", "rb"
        ) as f:
            return pickle.load(f)

    def _load_frame_list(self):
        """Load frame list"""
        frame_list = []
        with open(self.annotations_path / "frame_list.txt", "r") as f:
            for line in f.readlines():
                frame_list.append(line.strip())
        return frame_list

    def analyze_dataset(self):
        """Perform comprehensive dataset analysis"""
        print("=" * 80)
        print("ACTION GENOME 9000 DATASET ANALYSIS")
        print("=" * 80)

        # Basic dataset structure
        self._analyze_basic_structure()

        # Video and frame statistics
        self._analyze_video_frame_stats()

        # Object statistics
        self._analyze_object_stats()

        # Relationship statistics
        self._analyze_relationship_stats()

        # Temporal statistics
        self._analyze_temporal_stats()

        # Spatial statistics
        self._analyze_spatial_stats()

        # Print comprehensive report
        self._print_comprehensive_report()

        return self.stats

    def _analyze_basic_structure(self):
        """Analyze basic dataset structure"""
        print("\n1. BASIC DATASET STRUCTURE")
        print("-" * 40)

        # Check directory structure
        frames_exist = self.frames_path.exists()
        videos_exist = self.videos_path.exists()

        # Count files in frames directory
        frame_files = 0
        if frames_exist:
            frame_files = len(list(self.frames_path.rglob("*.jpg"))) + len(
                list(self.frames_path.rglob("*.png"))
            )

        # Count video files
        video_files = 0
        if videos_exist:
            video_files = len(list(self.videos_path.rglob("*.mp4"))) + len(
                list(self.videos_path.rglob("*.avi"))
            )

        self.stats["basic_structure"] = {
            "total_frames_in_list": len(self.frame_list),
            "frame_files_on_disk": frame_files,
            "video_files_on_disk": video_files,
            "object_classes": len(self.object_classes) - 1,  # Exclude background
            "relationship_classes": len(self.relationship_classes),
            "person_bbox_entries": len(self.person_bbox),
            "object_bbox_entries": len(self.object_bbox),
        }

        print(
            f"Total frames in annotation list: {self.stats['basic_structure']['total_frames_in_list']:,}"
        )
        print(
            f"Frame files on disk: {self.stats['basic_structure']['frame_files_on_disk']:,}"
        )
        print(
            f"Video files on disk: {self.stats['basic_structure']['video_files_on_disk']:,}"
        )
        print(f"Object classes: {self.stats['basic_structure']['object_classes']}")
        print(
            f"Relationship classes: {self.stats['basic_structure']['relationship_classes']}"
        )
        print(
            f"Person bbox entries: {self.stats['basic_structure']['person_bbox_entries']:,}"
        )
        print(
            f"Object bbox entries: {self.stats['basic_structure']['object_bbox_entries']:,}"
        )

    def _analyze_video_frame_stats(self):
        """Analyze video and frame statistics"""
        print("\n2. VIDEO AND FRAME STATISTICS")
        print("-" * 40)

        # Group frames by video
        video_frames = defaultdict(list)
        for frame_path in self.frame_list:
            if "/" in frame_path:
                video_name, frame_name = frame_path.split("/", 1)
                video_frames[video_name].append(frame_path)

        # Analyze video statistics
        video_stats = []
        for video_name, frames in video_frames.items():
            video_stats.append(
                {"video_name": video_name, "frame_count": len(frames), "frames": frames}
            )

        # Sort by frame count
        video_stats.sort(key=lambda x: x["frame_count"], reverse=True)

        frame_counts = [v["frame_count"] for v in video_stats]

        self.stats["video_frame_stats"] = {
            "total_videos": len(video_stats),
            "total_frames": sum(frame_counts),
            "avg_frames_per_video": np.mean(frame_counts),
            "median_frames_per_video": np.median(frame_counts),
            "min_frames_per_video": min(frame_counts),
            "max_frames_per_video": max(frame_counts),
            "std_frames_per_video": np.std(frame_counts),
            "videos_with_1_frame": sum(1 for c in frame_counts if c == 1),
            "videos_with_2_5_frames": sum(1 for c in frame_counts if 2 <= c <= 5),
            "videos_with_6_10_frames": sum(1 for c in frame_counts if 6 <= c <= 10),
            "videos_with_11_20_frames": sum(1 for c in frame_counts if 11 <= c <= 20),
            "videos_with_21_50_frames": sum(1 for c in frame_counts if 21 <= c <= 50),
            "videos_with_50_plus_frames": sum(1 for c in frame_counts if c > 50),
        }

        print(f"Total videos: {self.stats['video_frame_stats']['total_videos']:,}")
        print(f"Total frames: {self.stats['video_frame_stats']['total_frames']:,}")
        print(
            f"Average frames per video: {self.stats['video_frame_stats']['avg_frames_per_video']:.2f}"
        )
        print(
            f"Median frames per video: {self.stats['video_frame_stats']['median_frames_per_video']:.2f}"
        )
        print(
            f"Min frames per video: {self.stats['video_frame_stats']['min_frames_per_video']}"
        )
        print(
            f"Max frames per video: {self.stats['video_frame_stats']['max_frames_per_video']}"
        )
        print(
            f"Std frames per video: {self.stats['video_frame_stats']['std_frames_per_video']:.2f}"
        )

        print("\nFrame distribution:")
        print(
            f"  1 frame: {self.stats['video_frame_stats']['videos_with_1_frame']:,} videos"
        )
        print(
            f"  2-5 frames: {self.stats['video_frame_stats']['videos_with_2_5_frames']:,} videos"
        )
        print(
            f"  6-10 frames: {self.stats['video_frame_stats']['videos_with_6_10_frames']:,} videos"
        )
        print(
            f"  11-20 frames: {self.stats['video_frame_stats']['videos_with_11_20_frames']:,} videos"
        )
        print(
            f"  21-50 frames: {self.stats['video_frame_stats']['videos_with_21_50_frames']:,} videos"
        )
        print(
            f"  50+ frames: {self.stats['video_frame_stats']['videos_with_50_plus_frames']:,} videos"
        )

        # Store top videos for reference
        self.stats["top_videos_by_frames"] = video_stats[:10]

    def _analyze_object_stats(self):
        """Analyze object statistics"""
        print("\n3. OBJECT STATISTICS")
        print("-" * 40)

        object_counts = defaultdict(int)
        object_instances = defaultdict(int)
        visible_objects = 0
        total_objects = 0

        for frame_path, objects in self.object_bbox.items():
            for obj in objects:
                total_objects += 1
                if obj["visible"]:
                    visible_objects += 1
                    object_counts[obj["class"]] += 1
                    object_instances[obj["class"]] += 1

        # Sort objects by frequency
        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)

        self.stats["object_stats"] = {
            "total_object_instances": total_objects,
            "visible_object_instances": visible_objects,
            "visibility_rate": visible_objects / total_objects
            if total_objects > 0
            else 0,
            "unique_object_types": len(object_counts),
            "object_frequency": dict(sorted_objects),
            "most_common_objects": sorted_objects[:10],
        }

        print(
            f"Total object instances: {self.stats['object_stats']['total_object_instances']:,}"
        )
        print(
            f"Visible object instances: {self.stats['object_stats']['visible_object_instances']:,}"
        )
        print(
            f"Object visibility rate: {self.stats['object_stats']['visibility_rate']:.3f}"
        )
        print(
            f"Unique object types: {self.stats['object_stats']['unique_object_types']}"
        )

        print("\nMost common objects:")
        for i, (obj_class, count) in enumerate(
            self.stats["object_stats"]["most_common_objects"]
        ):
            print(f"  {i+1:2d}. {obj_class:20s}: {count:6,} instances")

    def _analyze_relationship_stats(self):
        """Analyze relationship statistics"""
        print("\n4. RELATIONSHIP STATISTICS")
        print("-" * 40)

        # Categorize relationships
        attention_relationships = self.relationship_classes[
            0:3
        ]  # looking_at, not_looking_at, unsure
        spatial_relationships = self.relationship_classes[
            3:9
        ]  # above, beneath, in_front_of, behind, on_the_side_of, in
        contacting_relationships = self.relationship_classes[
            9:
        ]  # carrying, covered_by, etc.

        relationship_counts = defaultdict(int)
        attention_counts = defaultdict(int)
        spatial_counts = defaultdict(int)
        contacting_counts = defaultdict(int)

        for frame_path, objects in self.object_bbox.items():
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
        sorted_spatial = sorted(
            spatial_counts.items(), key=lambda x: x[1], reverse=True
        )
        sorted_contacting = sorted(
            contacting_counts.items(), key=lambda x: x[1], reverse=True
        )

        self.stats["relationship_stats"] = {
            "total_relationships": sum(relationship_counts.values()),
            "attention_relationships": len(attention_relationships),
            "spatial_relationships": len(spatial_relationships),
            "contacting_relationships": len(contacting_relationships),
            "relationship_frequency": dict(sorted_relationships),
            "attention_frequency": dict(sorted_attention),
            "spatial_frequency": dict(sorted_spatial),
            "contacting_frequency": dict(sorted_contacting),
            "most_common_relationships": sorted_relationships[:10],
            "most_common_attention": sorted_attention,
            "most_common_spatial": sorted_spatial,
            "most_common_contacting": sorted_contacting[:10],
        }

        print(
            f"Total relationships: {self.stats['relationship_stats']['total_relationships']:,}"
        )
        print(
            f"Attention relationships: {self.stats['relationship_stats']['attention_relationships']}"
        )
        print(
            f"Spatial relationships: {self.stats['relationship_stats']['spatial_relationships']}"
        )
        print(
            f"Contacting relationships: {self.stats['relationship_stats']['contacting_relationships']}"
        )

        print("\nMost common relationships:")
        for i, (rel, count) in enumerate(
            self.stats["relationship_stats"]["most_common_relationships"]
        ):
            print(f"  {i+1:2d}. {rel:20s}: {count:6,} instances")

        print("\nMost common attention relationships:")
        for rel, count in self.stats["relationship_stats"]["most_common_attention"]:
            print(f"  - {rel:20s}: {count:6,} instances")

        print("\nMost common spatial relationships:")
        for rel, count in self.stats["relationship_stats"]["most_common_spatial"]:
            print(f"  - {rel:20s}: {count:6,} instances")

        print("\nMost common contacting relationships:")
        for i, (rel, count) in enumerate(
            self.stats["relationship_stats"]["most_common_contacting"]
        ):
            print(f"  {i+1:2d}. {rel:20s}: {count:6,} instances")

    def _analyze_temporal_stats(self):
        """Analyze temporal statistics"""
        print("\n5. TEMPORAL STATISTICS")
        print("-" * 40)

        # Analyze temporal patterns
        video_temporal_stats = []

        for frame_path, objects in self.object_bbox.items():
            if "/" in frame_path:
                video_name, frame_name = frame_path.split("/", 1)

                # Extract frame number if possible
                frame_num = 0
                try:
                    # Try to extract frame number from filename
                    if frame_name.endswith(".jpg") or frame_name.endswith(".png"):
                        frame_num = int(frame_name.split(".")[0])
                except:
                    pass

                # Count objects and relationships in this frame
                visible_objects = sum(1 for obj in objects if obj["visible"])
                total_relationships = 0
                for obj in objects:
                    if obj["visible"]:
                        total_relationships += len(
                            obj.get("attention_relationship", [])
                        )
                        total_relationships += len(obj.get("spatial_relationship", []))
                        total_relationships += len(
                            obj.get("contacting_relationship", [])
                        )

                video_temporal_stats.append(
                    {
                        "video": video_name,
                        "frame": frame_name,
                        "frame_num": frame_num,
                        "visible_objects": visible_objects,
                        "total_relationships": total_relationships,
                    }
                )

        # Group by video
        video_stats = defaultdict(list)
        for stat in video_temporal_stats:
            video_stats[stat["video"]].append(stat)

        # Calculate temporal statistics per video
        temporal_analysis = []
        for video_name, frames in video_stats.items():
            frames.sort(key=lambda x: x["frame_num"])

            object_counts = [f["visible_objects"] for f in frames]
            relationship_counts = [f["total_relationships"] for f in frames]

            temporal_analysis.append(
                {
                    "video": video_name,
                    "frame_count": len(frames),
                    "avg_objects_per_frame": np.mean(object_counts),
                    "std_objects_per_frame": np.std(object_counts),
                    "avg_relationships_per_frame": np.mean(relationship_counts),
                    "std_relationships_per_frame": np.std(relationship_counts),
                    "object_count_sequence": object_counts,
                    "relationship_count_sequence": relationship_counts,
                }
            )

        # Overall temporal statistics
        all_object_counts = [f["visible_objects"] for f in video_temporal_stats]
        all_relationship_counts = [
            f["total_relationships"] for f in video_temporal_stats
        ]

        self.stats["temporal_stats"] = {
            "total_frames_analyzed": len(video_temporal_stats),
            "avg_objects_per_frame": np.mean(all_object_counts),
            "std_objects_per_frame": np.std(all_object_counts),
            "min_objects_per_frame": min(all_object_counts),
            "max_objects_per_frame": max(all_object_counts),
            "avg_relationships_per_frame": np.mean(all_relationship_counts),
            "std_relationships_per_frame": np.std(all_relationship_counts),
            "min_relationships_per_frame": min(all_relationship_counts),
            "max_relationships_per_frame": max(all_relationship_counts),
            "videos_with_increasing_objects": sum(
                1
                for v in temporal_analysis
                if self._is_increasing(v["object_count_sequence"])
            ),
            "videos_with_decreasing_objects": sum(
                1
                for v in temporal_analysis
                if self._is_decreasing(v["object_count_sequence"])
            ),
            "videos_with_stable_objects": sum(
                1
                for v in temporal_analysis
                if self._is_stable(v["object_count_sequence"])
            ),
        }

        print(
            f"Total frames analyzed: {self.stats['temporal_stats']['total_frames_analyzed']:,}"
        )
        print(
            f"Average objects per frame: {self.stats['temporal_stats']['avg_objects_per_frame']:.2f}"
        )
        print(
            f"Std objects per frame: {self.stats['temporal_stats']['std_objects_per_frame']:.2f}"
        )
        print(
            f"Min objects per frame: {self.stats['temporal_stats']['min_objects_per_frame']}"
        )
        print(
            f"Max objects per frame: {self.stats['temporal_stats']['max_objects_per_frame']}"
        )
        print(
            f"Average relationships per frame: {self.stats['temporal_stats']['avg_relationships_per_frame']:.2f}"
        )
        print(
            f"Std relationships per frame: {self.stats['temporal_stats']['std_relationships_per_frame']:.2f}"
        )
        print(
            f"Min relationships per frame: {self.stats['temporal_stats']['min_relationships_per_frame']}"
        )
        print(
            f"Max relationships per frame: {self.stats['temporal_stats']['max_relationships_per_frame']}"
        )

        print(f"\nTemporal patterns:")
        print(
            f"  Videos with increasing object count: {self.stats['temporal_stats']['videos_with_increasing_objects']}"
        )
        print(
            f"  Videos with decreasing object count: {self.stats['temporal_stats']['videos_with_decreasing_objects']}"
        )
        print(
            f"  Videos with stable object count: {self.stats['temporal_stats']['videos_with_stable_objects']}"
        )

    def _analyze_spatial_stats(self):
        """Analyze spatial statistics"""
        print("\n6. SPATIAL STATISTICS")
        print("-" * 40)

        # Analyze bounding box statistics
        bbox_areas = []
        bbox_aspect_ratios = []
        bbox_positions = []

        for frame_path, objects in self.object_bbox.items():
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
                        bbox_positions.append((bbox[0], bbox[1]))  # top-left corner

        # Analyze person bounding boxes
        person_bbox_areas = []
        person_bbox_aspect_ratios = []

        for frame_path, person_data in self.person_bbox.items():
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

        self.stats["spatial_stats"] = {
            "total_bboxes": len(bbox_areas),
            "total_person_bboxes": len(person_bbox_areas),
            "avg_bbox_area": np.mean(bbox_areas) if bbox_areas else 0,
            "std_bbox_area": np.std(bbox_areas) if bbox_areas else 0,
            "min_bbox_area": min(bbox_areas) if bbox_areas else 0,
            "max_bbox_area": max(bbox_areas) if bbox_areas else 0,
            "avg_bbox_aspect_ratio": np.mean(bbox_aspect_ratios)
            if bbox_aspect_ratios
            else 0,
            "std_bbox_aspect_ratio": np.std(bbox_aspect_ratios)
            if bbox_aspect_ratios
            else 0,
            "avg_person_bbox_area": np.mean(person_bbox_areas)
            if person_bbox_areas
            else 0,
            "std_person_bbox_area": np.std(person_bbox_areas)
            if person_bbox_areas
            else 0,
            "avg_person_bbox_aspect_ratio": np.mean(person_bbox_aspect_ratios)
            if person_bbox_aspect_ratios
            else 0,
            "std_person_bbox_aspect_ratio": np.std(person_bbox_aspect_ratios)
            if person_bbox_aspect_ratios
            else 0,
        }

        print(f"Total bounding boxes: {self.stats['spatial_stats']['total_bboxes']:,}")
        print(
            f"Total person bounding boxes: {self.stats['spatial_stats']['total_person_bboxes']:,}"
        )
        print(f"Average bbox area: {self.stats['spatial_stats']['avg_bbox_area']:.2f}")
        print(f"Std bbox area: {self.stats['spatial_stats']['std_bbox_area']:.2f}")
        print(f"Min bbox area: {self.stats['spatial_stats']['min_bbox_area']:.2f}")
        print(f"Max bbox area: {self.stats['spatial_stats']['max_bbox_area']:.2f}")
        print(
            f"Average bbox aspect ratio: {self.stats['spatial_stats']['avg_bbox_aspect_ratio']:.3f}"
        )
        print(
            f"Std bbox aspect ratio: {self.stats['spatial_stats']['std_bbox_aspect_ratio']:.3f}"
        )
        print(
            f"Average person bbox area: {self.stats['spatial_stats']['avg_person_bbox_area']:.2f}"
        )
        print(
            f"Std person bbox area: {self.stats['spatial_stats']['std_person_bbox_area']:.2f}"
        )
        print(
            f"Average person bbox aspect ratio: {self.stats['spatial_stats']['avg_person_bbox_aspect_ratio']:.3f}"
        )
        print(
            f"Std person bbox aspect ratio: {self.stats['spatial_stats']['std_person_bbox_aspect_ratio']:.3f}"
        )

    def _is_increasing(self, sequence):
        """Check if sequence is generally increasing"""
        if len(sequence) < 2:
            return False
        return sequence[-1] > sequence[0]

    def _is_decreasing(self, sequence):
        """Check if sequence is generally decreasing"""
        if len(sequence) < 2:
            return False
        return sequence[-1] < sequence[0]

    def _is_stable(self, sequence):
        """Check if sequence is relatively stable"""
        if len(sequence) < 2:
            return True
        return abs(sequence[-1] - sequence[0]) <= 2  # Allow small variation

    def _print_comprehensive_report(self):
        """Print comprehensive dataset report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE DATASET SUMMARY")
        print("=" * 80)

        print(f"\nDataset: Action Genome 9000")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Path: {self.data_path}")

        print(f"\n📊 DATASET OVERVIEW:")
        print(f"  • Total Videos: {self.stats['video_frame_stats']['total_videos']:,}")
        print(f"  • Total Frames: {self.stats['video_frame_stats']['total_frames']:,}")
        print(f"  • Object Classes: {self.stats['basic_structure']['object_classes']}")
        print(
            f"  • Relationship Classes: {self.stats['basic_structure']['relationship_classes']}"
        )

        print(f"\n🎬 VIDEO STATISTICS:")
        print(
            f"  • Average Frames per Video: {self.stats['video_frame_stats']['avg_frames_per_video']:.2f}"
        )
        print(
            f"  • Median Frames per Video: {self.stats['video_frame_stats']['median_frames_per_video']:.2f}"
        )
        print(
            f"  • Frame Range: {self.stats['video_frame_stats']['min_frames_per_video']} - {self.stats['video_frame_stats']['max_frames_per_video']}"
        )

        print(f"\n🎯 OBJECT STATISTICS:")
        print(
            f"  • Total Object Instances: {self.stats['object_stats']['total_object_instances']:,}"
        )
        print(
            f"  • Visible Object Instances: {self.stats['object_stats']['visible_object_instances']:,}"
        )
        print(
            f"  • Object Visibility Rate: {self.stats['object_stats']['visibility_rate']:.1%}"
        )
        print(
            f"  • Unique Object Types: {self.stats['object_stats']['unique_object_types']}"
        )

        print(f"\n🔗 RELATIONSHIP STATISTICS:")
        print(
            f"  • Total Relationships: {self.stats['relationship_stats']['total_relationships']:,}"
        )
        print(
            f"  • Attention Relationships: {self.stats['relationship_stats']['attention_relationships']}"
        )
        print(
            f"  • Spatial Relationships: {self.stats['relationship_stats']['spatial_relationships']}"
        )
        print(
            f"  • Contacting Relationships: {self.stats['relationship_stats']['contacting_relationships']}"
        )

        print(f"\n⏱️  TEMPORAL STATISTICS:")
        print(
            f"  • Average Objects per Frame: {self.stats['temporal_stats']['avg_objects_per_frame']:.2f}"
        )
        print(
            f"  • Average Relationships per Frame: {self.stats['temporal_stats']['avg_relationships_per_frame']:.2f}"
        )
        print(
            f"  • Frames Analyzed: {self.stats['temporal_stats']['total_frames_analyzed']:,}"
        )

        print(f"\n📐 SPATIAL STATISTICS:")
        print(
            f"  • Total Bounding Boxes: {self.stats['spatial_stats']['total_bboxes']:,}"
        )
        print(
            f"  • Total Person Bounding Boxes: {self.stats['spatial_stats']['total_person_bboxes']:,}"
        )
        print(
            f"  • Average Bbox Area: {self.stats['spatial_stats']['avg_bbox_area']:.2f}"
        )
        print(
            f"  • Average Bbox Aspect Ratio: {self.stats['spatial_stats']['avg_bbox_aspect_ratio']:.3f}"
        )

        print(f"\n📈 DISTRIBUTION BREAKDOWN:")
        print(
            f"  • Videos with 1 frame: {self.stats['video_frame_stats']['videos_with_1_frame']:,}"
        )
        print(
            f"  • Videos with 2-5 frames: {self.stats['video_frame_stats']['videos_with_2_5_frames']:,}"
        )
        print(
            f"  • Videos with 6-10 frames: {self.stats['video_frame_stats']['videos_with_6_10_frames']:,}"
        )
        print(
            f"  • Videos with 11-20 frames: {self.stats['video_frame_stats']['videos_with_11_20_frames']:,}"
        )
        print(
            f"  • Videos with 21-50 frames: {self.stats['video_frame_stats']['videos_with_21_50_frames']:,}"
        )
        print(
            f"  • Videos with 50+ frames: {self.stats['video_frame_stats']['videos_with_50_plus_frames']:,}"
        )

        print(f"\n🏆 TOP 5 MOST COMMON OBJECTS:")
        for i, (obj_class, count) in enumerate(
            self.stats["object_stats"]["most_common_objects"][:5]
        ):
            print(f"  {i+1}. {obj_class:20s}: {count:6,} instances")

        print(f"\n🏆 TOP 5 MOST COMMON RELATIONSHIPS:")
        for i, (rel, count) in enumerate(
            self.stats["relationship_stats"]["most_common_relationships"][:5]
        ):
            print(f"  {i+1}. {rel:20s}: {count:6,} instances")

        print("\n" + "=" * 80)

    def save_stats(self, output_path="action_genome9000_stats.json"):
        """Save statistics to JSON file"""
        with open(output_path, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)
        print(f"\nStatistics saved to: {output_path}")

    def save_detailed_report(self, output_path="action_genome9000_detailed_report.txt"):
        """Save detailed report to text file"""
        with open(output_path, "w") as f:
            f.write("ACTION GENOME 9000 DATASET DETAILED REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Path: {self.data_path}\n\n")

            # Write all statistics in a structured format
            for section_name, section_data in self.stats.items():
                f.write(f"{section_name.upper().replace('_', ' ')}\n")
                f.write("-" * len(section_name) + "\n")
                for key, value in section_data.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

        print(f"Detailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Action Genome 9000 dataset statistics"
    )
    parser.add_argument(
        "--data_path",
        default="data/action_genome9000",
        help="Path to the action_genome9000 dataset",
    )
    parser.add_argument(
        "--output_json",
        default="action_genome9000_stats.json",
        help="Output JSON file for statistics",
    )
    parser.add_argument(
        "--output_report",
        default="action_genome9000_detailed_report.txt",
        help="Output text file for detailed report",
    )

    args = parser.parse_args()

    # Check if data path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist!")
        return

    try:
        # Initialize analyzer
        analyzer = ActionGenomeAnalyzer(args.data_path)

        # Perform analysis
        stats = analyzer.analyze_dataset()

        # Save results
        analyzer.save_stats(args.output_json)
        analyzer.save_detailed_report(args.output_report)

        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Statistics saved to: {args.output_json}")
        print(f"📄 Detailed report saved to: {args.output_report}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
