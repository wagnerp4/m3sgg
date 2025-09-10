"""Constants and class definitions for M3SGG.

This module provides centralized definitions for object classes, relationship classes,
and other constants used throughout the M3SGG framework.

:author: M3SGG Team
:version: 0.1.0
"""

from typing import List, Optional
import os


# Action Genome Object Classes
# These match the classes defined in data/action_genome/annotations/object_classes.txt
ACTION_GENOME_OBJECT_CLASSES = [
    "person",
    "bag",
    "bed",
    "blanket",
    "book",
    "box",
    "broom",
    "chair",
    "closetcabinet",
    "clothes",
    "cupglassbottle",
    "dish",
    "door",
    "doorknob",
    "doorway",
    "floor",
    "food",
    "groceries",
    "laptop",
    "light",
    "medicine",
    "mirror",
    "papernotebook",
    "phonecamera",
    "picture",
    "pillow",
    "refrigerator",
    "sandwich",
    "shelf",
    "shoe",
    "sofacouch",
    "table",
    "television",
    "towel",
    "vacuum",
    "window",
]

# Action Genome Relationship Classes (raw format from file)
# These match the classes defined in data/action_genome/annotations/relationship_classes.txt
ACTION_GENOME_RELATIONSHIP_CLASSES_RAW = [
    "lookingat",
    "notlookingat",
    "unsure",
    "above",
    "beneath",
    "infrontof",
    "behind",
    "onthesideof",
    "in",
    "carrying",
    "coveredby",
    "drinkingfrom",
    "eating",
    "haveitontheback",
    "holding",
    "leaningon",
    "lyingon",
    "notcontacting",
    "otherrelationship",
    "sittingon",
    "standingon",
    "touching",
    "twisting",
    "wearing",
    "wiping",
    "writingon",
]

# Action Genome Relationship Classes (corrected format with underscores)
# This applies the same corrections as done in the dataset classes
ACTION_GENOME_RELATIONSHIP_CLASSES = ACTION_GENOME_RELATIONSHIP_CLASSES_RAW.copy()
ACTION_GENOME_RELATIONSHIP_CLASSES[0] = "looking_at"
ACTION_GENOME_RELATIONSHIP_CLASSES[1] = "not_looking_at"
ACTION_GENOME_RELATIONSHIP_CLASSES[5] = "in_front_of"
ACTION_GENOME_RELATIONSHIP_CLASSES[7] = "on_the_side_of"
ACTION_GENOME_RELATIONSHIP_CLASSES[10] = "covered_by"
ACTION_GENOME_RELATIONSHIP_CLASSES[11] = "drinking_from"
ACTION_GENOME_RELATIONSHIP_CLASSES[13] = "have_it_on_the_back"
ACTION_GENOME_RELATIONSHIP_CLASSES[15] = "leaning_on"
ACTION_GENOME_RELATIONSHIP_CLASSES[16] = "lying_on"
ACTION_GENOME_RELATIONSHIP_CLASSES[17] = "not_contacting"
ACTION_GENOME_RELATIONSHIP_CLASSES[18] = "other_relationship"
ACTION_GENOME_RELATIONSHIP_CLASSES[19] = "sitting_on"
ACTION_GENOME_RELATIONSHIP_CLASSES[20] = "standing_on"
ACTION_GENOME_RELATIONSHIP_CLASSES[25] = "writing_on"

# Relationship class categories for filtering
ATTENTION_RELATIONSHIPS = ACTION_GENOME_RELATIONSHIP_CLASSES[
    0:3
]  # looking_at, not_looking_at, unsure
SPATIAL_RELATIONSHIPS = ACTION_GENOME_RELATIONSHIP_CLASSES[
    3:9
]  # above, beneath, in_front_of, behind, on_the_side_of, in
CONTACT_RELATIONSHIPS = ACTION_GENOME_RELATIONSHIP_CLASSES[
    9:
]  # carrying, covered_by, etc.

# Backward compatibility aliases
OBJECT_CLASSES = ACTION_GENOME_OBJECT_CLASSES
RELATIONSHIP_CLASSES = ACTION_GENOME_RELATIONSHIP_CLASSES_RAW
RELATIONSHIP_CLASSES_CORRECTED = ACTION_GENOME_RELATIONSHIP_CLASSES


def load_object_classes_from_file(data_path: str) -> Optional[List[str]]:
    """Load object classes from Action Genome dataset file.

    :param data_path: Path to the dataset directory
    :type data_path: str
    :return: List of object classes or None if loading fails
    :rtype: Optional[List[str]]
    """
    try:
        object_classes_path = os.path.join(
            data_path, "annotations", "object_classes.txt"
        )
        if os.path.exists(object_classes_path):
            with open(object_classes_path, "r") as f:
                classes = [line.strip() for line in f.readlines()]
                # Apply the same corrections as in the dataset class
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
    except Exception as e:
        print(f"Warning: Could not load object classes from file: {e}")

    return None


def load_relationship_classes_from_file(data_path: str) -> Optional[List[str]]:
    """Load relationship classes from Action Genome dataset file.

    :param data_path: Path to the dataset directory
    :type data_path: str
    :return: List of relationship classes (corrected format) or None if loading fails
    :rtype: Optional[List[str]]
    """
    try:
        relationship_classes_path = os.path.join(
            data_path, "annotations", "relationship_classes.txt"
        )
        if os.path.exists(relationship_classes_path):
            with open(relationship_classes_path, "r") as f:
                classes = [line.strip() for line in f.readlines()]
                # Apply the same corrections as in the dataset class
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
    except Exception as e:
        print(f"Warning: Could not load relationship classes from file: {e}")

    return None


def get_object_classes(data_path: Optional[str] = None) -> List[str]:
    """Get object classes, loading from file if available, otherwise using constants.

    :param data_path: Optional path to dataset directory
    :type data_path: Optional[str]
    :return: List of object classes
    :rtype: List[str]
    """
    if data_path:
        classes = load_object_classes_from_file(data_path)
        if classes is not None:
            return classes

    return ACTION_GENOME_OBJECT_CLASSES.copy()


def get_relationship_classes(data_path: Optional[str] = None) -> List[str]:
    """Get relationship classes, loading from file if available, otherwise using constants.

    :param data_path: Optional path to dataset directory
    :type data_path: Optional[str]
    :return: List of relationship classes (corrected format)
    :rtype: List[str]
    """
    if data_path:
        classes = load_relationship_classes_from_file(data_path)
        if classes is not None:
            return classes

    return ACTION_GENOME_RELATIONSHIP_CLASSES.copy()


def get_relationship_classes_by_category(
    category: str, data_path: Optional[str] = None
) -> List[str]:
    """Get relationship classes filtered by category.

    :param category: Relationship category ("attention", "spatial", "contact")
    :type category: str
    :param data_path: Optional path to dataset directory
    :type data_path: Optional[str]
    :return: List of relationship classes for the specified category
    :rtype: List[str]
    """
    all_classes = get_relationship_classes(data_path)

    if category == "attention":
        return all_classes[0:3]
    elif category == "spatial":
        return all_classes[3:9]
    elif category == "contact":
        return all_classes[9:]
    else:
        return all_classes
