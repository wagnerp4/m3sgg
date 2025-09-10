"""
ActionGenome dataset loader for ground truth scene graphs.

This module provides functionality to load ActionGenome dataset with ground truth
scene graph annotations for summarization evaluation.
"""

import os
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ActionGenomeLoader:
    """Loader for ActionGenome dataset with ground truth scene graphs.

    :param data_path: Path to ActionGenome dataset
    :type data_path: str
    :param split: Dataset split (train/val/test)
    :type split: str
    :param max_videos: Maximum number of videos to load
    :type max_videos: int, optional
    """

    def __init__(
        self, data_path: str, split: str = "val", max_videos: Optional[int] = None
    ):
        """Initialize ActionGenome loader.

        :param data_path: Path to ActionGenome dataset
        :type data_path: str
        :param split: Dataset split (train/val/test)
        :type split: str
        :param max_videos: Maximum number of videos to load
        :type max_videos: int, optional
        """
        self.data_path = Path(data_path)
        self.split = split
        self.max_videos = max_videos

        # Load class mappings
        self.object_classes = self._load_object_classes()
        self.relationship_classes = self._load_relationship_classes()

        # Load annotations
        self.annotations = self._load_annotations()

        logger.info(
            f"Loaded ActionGenome {split} split with {len(self.annotations)} videos"
        )

    def _load_object_classes(self) -> List[str]:
        """Load object class names.

        :return: List of object class names
        :rtype: List[str]
        """
        classes_path = self.data_path / "annotations" / "object_classes.txt"
        with open(classes_path, "r") as f:
            classes = ["__background__"] + [line.strip() for line in f.readlines()]

        # Apply known corrections
        if len(classes) > 9:
            classes[9] = "closet/cabinet"
        if len(classes) > 11:
            classes[11] = "cup/glass/bottle"
        if len(classes) > 23:
            classes[23] = "paper/notebook"
        if len(classes) > 24:
            classes[24] = "phone/camera"
        if len(classes) > 31:
            classes[31] = "sofa/couch"

        return classes

    def _load_relationship_classes(self) -> List[str]:
        """Load relationship class names.

        :return: List of relationship class names
        :rtype: List[str]
        """
        classes_path = self.data_path / "annotations" / "relationship_classes.txt"
        with open(classes_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Apply known corrections
        if len(classes) > 0:
            classes[0] = "looking_at"
        if len(classes) > 1:
            classes[1] = "not_looking_at"
        if len(classes) > 5:
            classes[5] = "in_front_of"
        if len(classes) > 7:
            classes[7] = "on_the_side_of"
        if len(classes) > 10:
            classes[10] = "covered_by"
        if len(classes) > 11:
            classes[11] = "drinking_from"
        if len(classes) > 13:
            classes[13] = "have_it_on_the_back"
        if len(classes) > 15:
            classes[15] = "leaning_on"
        if len(classes) > 16:
            classes[16] = "lying_on"
        if len(classes) > 17:
            classes[17] = "not_contacting"
        if len(classes) > 18:
            classes[18] = "other_relationship"
        if len(classes) > 19:
            classes[19] = "sitting_on"
        if len(classes) > 20:
            classes[20] = "standing_on"
        if len(classes) > 25:
            classes[25] = "writing_on"

        return classes

    def _load_annotations(self) -> Dict[str, Any]:
        """Load video annotations.

        :return: Dictionary mapping video IDs to annotations
        :rtype: Dict[str, Any]
        """
        # Load person bboxes
        person_bbox_path = self.data_path / "annotations" / "person_bbox.pkl"
        with open(person_bbox_path, "rb") as f:
            person_bbox = pickle.load(f)

        # Load object bboxes and relationships
        object_bbox_path = (
            self.data_path / "annotations" / "object_bbox_and_relationship.pkl"
        )
        with open(object_bbox_path, "rb") as f:
            object_bbox = pickle.load(f)

        # Filter by split and process
        annotations = {}
        video_count = 0
        processed_frames = 0

        logger.info(f"Processing {len(object_bbox)} frames from object_bbox")
        logger.info(f"Looking for split: {self.split}")

        for frame_key, frame_data in object_bbox.items():
            if not frame_data or len(frame_data) == 0:
                continue

            frame_split = frame_data[0]["metadata"]["set"]
            if frame_split != self.split:
                continue

            video_name, frame_num = frame_key.split("/")
            processed_frames += 1

            if video_name not in annotations:
                annotations[video_name] = {
                    "video_id": video_name,
                    "frames": [],
                    "frame_count": 0,
                }
                video_count += 1

                if self.max_videos and video_count > self.max_videos:
                    break

            # Process frame data
            frame_annotation = self._process_frame_annotation(
                frame_key, frame_data, person_bbox.get(frame_key, {})
            )
            if frame_annotation:
                annotations[video_name]["frames"].append(frame_annotation)
                annotations[video_name]["frame_count"] += 1

        logger.info(f"Processed {processed_frames} frames for {video_count} videos")

        # Filter out videos with too few frames
        filtered_annotations = {}
        for video_id, video_data in annotations.items():
            logger.debug(f"Video {video_id}: {video_data['frame_count']} frames")
            if video_data["frame_count"] >= 1:  # At least 1 frame (more lenient)
                filtered_annotations[video_id] = video_data

        logger.info(
            f"Filtered to {len(filtered_annotations)} videos with sufficient frames"
        )
        return filtered_annotations

    def _process_frame_annotation(
        self, frame_key: str, frame_data: List[Dict], person_data: Dict
    ) -> Optional[Dict[str, Any]]:
        """Process a single frame annotation.

        :param frame_key: Frame identifier
        :type frame_key: str
        :param frame_data: Frame annotation data
        :type frame_data: List[Dict]
        :param person_data: Person bbox data
        :type person_data: Dict
        :return: Processed frame annotation
        :rtype: Optional[Dict[str, Any]]
        """
        frame_annotation = {"frame_id": frame_key, "objects": [], "relationships": []}

        # Add person objects
        if "bbox" in person_data and person_data["bbox"].shape[0] > 0:
            for i, bbox in enumerate(person_data["bbox"]):
                person_obj = {
                    "object_id": f"person_{i}",
                    "class": "person",
                    "bbox": bbox.tolist() if hasattr(bbox, "tolist") else bbox,
                    "visible": True,
                }
                frame_annotation["objects"].append(person_obj)

        # Add other objects
        for obj_data in frame_data:
            if not obj_data.get("visible", False):
                continue

            # Handle class as string or integer
            if isinstance(obj_data["class"], str):
                class_name = obj_data["class"]
            else:
                class_name = (
                    self.object_classes[obj_data["class"]]
                    if obj_data["class"] < len(self.object_classes)
                    else "unknown"
                )

            obj = {
                "object_id": f"obj_{len(frame_annotation['objects'])}",
                "class": class_name,
                "bbox": obj_data["bbox"].tolist()
                if hasattr(obj_data["bbox"], "tolist")
                else obj_data["bbox"],
                "visible": True,
            }
            frame_annotation["objects"].append(obj)

            # Add relationships
            self._add_relationships(obj_data, obj["object_id"], frame_annotation)

        return frame_annotation if frame_annotation["objects"] else None

    def _add_relationships(self, obj_data: Dict, obj_id: str, frame_annotation: Dict):
        """Add relationships for an object.

        :param obj_data: Object data
        :type obj_data: Dict
        :param obj_id: Object identifier
        :type obj_id: str
        :param frame_annotation: Frame annotation to update
        :type frame_annotation: Dict
        """
        # Add attention relationships
        if "attention_relationship" in obj_data:
            for rel_idx in obj_data["attention_relationship"]:
                if isinstance(rel_idx, str):
                    predicate = rel_idx
                elif isinstance(rel_idx, int) and rel_idx < len(
                    self.relationship_classes
                ):
                    predicate = self.relationship_classes[rel_idx]
                else:
                    continue

                rel = {
                    "subject_id": "person_0",  # Assume first person
                    "object_id": obj_id,
                    "predicate": predicate,
                }
                frame_annotation["relationships"].append(rel)

        # Add spatial relationships
        if "spatial_relationship" in obj_data:
            for rel_idx in obj_data["spatial_relationship"]:
                if isinstance(rel_idx, str):
                    predicate = rel_idx
                elif isinstance(rel_idx, int) and rel_idx < len(
                    self.relationship_classes
                ):
                    predicate = self.relationship_classes[rel_idx]
                else:
                    continue

                rel = {
                    "subject_id": "person_0",  # Assume first person
                    "object_id": obj_id,
                    "predicate": predicate,
                }
                frame_annotation["relationships"].append(rel)

        # Add contacting relationships
        if "contacting_relationship" in obj_data:
            for rel_idx in obj_data["contacting_relationship"]:
                if isinstance(rel_idx, str):
                    predicate = rel_idx
                elif isinstance(rel_idx, int) and rel_idx < len(
                    self.relationship_classes
                ):
                    predicate = self.relationship_classes[rel_idx]
                else:
                    continue

                rel = {
                    "subject_id": "person_0",  # Assume first person
                    "object_id": obj_id,
                    "predicate": predicate,
                }
                frame_annotation["relationships"].append(rel)

    def get_video_list(self) -> List[str]:
        """Get list of video IDs.

        :return: List of video IDs
        :rtype: List[str]
        """
        return list(self.annotations.keys())

    def get_video_annotations(self, video_id: str) -> Dict[str, Any]:
        """Get annotations for a specific video.

        :param video_id: Video identifier
        :type video_id: str
        :return: Video annotations
        :rtype: Dict[str, Any]
        """
        return self.annotations.get(video_id, {})

    def extract_scene_graph_triples(self, video_id: str) -> List[Tuple[str, str, str]]:
        """Extract scene graph triples for a video.

        :param video_id: Video identifier
        :type video_id: str
        :return: List of (subject, predicate, object) triples
        :rtype: List[Tuple[str, str, str]]
        """
        video_data = self.get_video_annotations(video_id)
        if not video_data:
            return []

        triples = []

        # Collect all objects across frames
        all_objects = {}
        for frame in video_data.get("frames", []):
            for obj in frame.get("objects", []):
                obj_id = obj["object_id"]
                if obj_id not in all_objects:
                    all_objects[obj_id] = {"class": obj["class"], "count": 0}
                all_objects[obj_id]["count"] += 1

        # Extract relationships
        for frame in video_data.get("frames", []):
            for rel in frame.get("relationships", []):
                subject_id = rel["subject_id"]
                object_id = rel["object_id"]
                predicate = rel["predicate"]

                # Get object classes
                subject_class = all_objects.get(subject_id, {}).get("class", "unknown")
                object_class = all_objects.get(object_id, {}).get("class", "unknown")

                triple = (subject_class, predicate, object_class)
                if triple not in triples:
                    triples.append(triple)

        return triples

    def create_subset(self, subset_size: int) -> Dict[str, List[Dict[str, Any]]]:
        """Create a subset of the dataset for evaluation.

        :param subset_size: Number of samples in subset
        :type subset_size: int
        :return: Subset data
        :rtype: Dict[str, List[Dict[str, Any]]]
        """
        video_ids = self.get_video_list()
        subset_videos = video_ids[:subset_size]

        subset_data = []
        for video_id in subset_videos:
            triples = self.extract_scene_graph_triples(video_id)

            # Create a simple caption based on the scene graph
            caption = self._generate_caption_from_triples(triples)

            sample = {
                "video_id": video_id,
                "caption": caption,
                "triples": triples,
                "frame_count": self.annotations[video_id]["frame_count"],
            }
            subset_data.append(sample)

        return {"test": subset_data}

    def _generate_caption_from_triples(
        self, triples: List[Tuple[str, str, str]]
    ) -> str:
        """Generate a simple caption from scene graph triples.

        :param triples: List of (subject, predicate, object) triples
        :type triples: List[Tuple[str, str, str]]
        :return: Generated caption
        :rtype: str
        """
        if not triples:
            return "A scene with no visible objects or relationships."

        # Simple aggregation: count most common objects and relationships
        object_counts = {}
        relationship_counts = {}

        for subject, predicate, obj in triples:
            if subject != "unknown":
                object_counts[subject] = object_counts.get(subject, 0) + 1
            if obj != "unknown":
                object_counts[obj] = object_counts.get(obj, 0) + 1
            if predicate != "other_relationship":
                relationship_counts[predicate] = (
                    relationship_counts.get(predicate, 0) + 1
                )

        # Create description
        parts = []

        # Add objects
        if object_counts:
            sorted_objects = sorted(
                object_counts.items(), key=lambda x: x[1], reverse=True
            )
            top_objects = [obj for obj, count in sorted_objects[:3]]
            if top_objects:
                parts.append(f"Objects: {', '.join(top_objects)}")

        # Add relationships
        if relationship_counts:
            sorted_rels = sorted(
                relationship_counts.items(), key=lambda x: x[1], reverse=True
            )
            top_rels = [rel for rel, count in sorted_rels[:3]]
            if top_rels:
                parts.append(f"Activities: {', '.join(top_rels)}")

        return (
            ". ".join(parts)
            if parts
            else "A scene with various objects and activities."
        )
