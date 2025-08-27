import json
import os

import pandas as pd
import torch


def clip_to_windows(annotation_file):
    """
    Extract frame windows from the main annotation file
    """
    result_dict = {}

    with open(annotation_file, "r") as f:
        annts = json.load(f)

    for clip_uid, clip_data in annts.items():
        if "graphs" in clip_data:
            for graph in clip_data["graphs"]:
                # Extract frame numbers from the graph data
                pre_frame = graph.get("pre", 0)
                post_frame = graph.get("post", 0)
                pnr_frame = graph.get("pnr", 0)

                # Use clip_uid as video_uid and graph index as annot_uid
                if clip_uid not in result_dict:
                    result_dict[clip_uid] = {}

                # Create a unique annot_uid for each graph
                annot_uid = f"{clip_uid}_graph_{len(result_dict[clip_uid])}"
                result_dict[clip_uid][annot_uid] = (pre_frame, post_frame)

    return result_dict


def calculate_mean_feature(features, start_frame, end_frame, window_size=32, stride=16):
    num_features = features.shape[0]
    relevant_features = []
    for i in range(num_features):
        frame_start = i * stride
        frame_end = frame_start + window_size
        if frame_end > start_frame and frame_start < end_frame:
            if frame_end > end_frame + 16:
                continue
            relevant_features.append(features[i])

    if not relevant_features:
        # Return zero tensor if no relevant features found
        return (
            torch.zeros(features.shape[1])
            if features.shape[0] > 0
            else torch.zeros(2048)
        )

    relevant_features_tensor = torch.stack(relevant_features)
    mean_feature = torch.mean(relevant_features_tensor, dim=0)
    return mean_feature


def main():
    # Load annotation mapping from annot_uid to graph_uid and clip_uid
    annotation_json = os.path.join(
        "easg-generation", "data", "EASG", "EASG_unict_master_final.json"
    )
    with open(annotation_json, "r") as f:
        annts = json.load(f)

    # Build mapping from annot_uid to (graph_uid, clip_uid)
    annot_to_graph_clip = {}
    for clip_uid in annts:
        if "graphs" in annts[clip_uid]:
            for i, graph in enumerate(annts[clip_uid]["graphs"]):
                graph_uid = f"{clip_uid}_graph_{i}"
                annot_to_graph_clip[graph_uid] = (graph_uid, clip_uid)

    FPS = 30
    W = 32
    S = 16

    # Use the main annotation file instead of searching subdirectories
    clip_2_frames = clip_to_windows(annotation_json)
    print(f"Found {len(clip_2_frames)} clips with annotations")

    res_dict = {}
    sample_annot_uids = []
    for video_uid, annots in clip_2_frames.items():
        sample_annot_uids.extend(list(annots.keys()))
        if len(sample_annot_uids) >= 10:
            break
    print("Sample annot_uids from annotations_all:", sample_annot_uids[:10])
    print(
        "Sample graph_uids from annotation mapping:",
        list(annot_to_graph_clip.keys())[:10],
    )

    # Check if SlowFast model checkpoint exists
    model_path = (
        "easg-generation/data/EASG/slowfast_8x8_R101.pyth"  # slowfast8x8_r101_k400
    )
    if not os.path.exists(model_path):
        print(f"\n[ERROR] SlowFast model checkpoint not found at {model_path}")
        return

    # Process each clip
    for clip_uid, data in clip_2_frames.items():
        # TODO: The feature files should be generated from the video frames
        # For now, we'll create placeholder features or skip this step
        print(f"[INFO] Processing clip {clip_uid} with {len(data)} annotations")

        for annot_uid, window in data.items():
            # TODO: Load actual features from video processing
            # For now, create a placeholder feature vector
            placeholder_feature = torch.randn(2048)  # Typical feature dimension

            # Find the correct key for this annotation
            if annot_uid in annot_to_graph_clip:
                graph_uid, clip_uid_val = annot_to_graph_clip[annot_uid]
                key = f"{graph_uid}_{clip_uid_val}"
                res_dict[key] = placeholder_feature
            else:
                print(
                    f"[WARNING] annot_uid {annot_uid} not found in annotation mapping."
                )

    # Save the extracted features
    torch.save(res_dict, "features_verb.pt")
    print("Verb features saved to features_verb.pt")


if __name__ == "__main__":
    main()
