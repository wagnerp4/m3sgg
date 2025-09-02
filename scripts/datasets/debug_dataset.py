import json
import os

import torch

from datasets.EASG import EASG


def debug_dataset():
    """Debug the EASG dataset to understand why video_list is empty"""

    data_path = "easg-generation/data/EASG"

    print("=== Debugging EASG Dataset ===")
    print(f"Data path: {data_path}")

    # Check if required files exist
    required_files = ["features_verb.pt", "EASG_unict_master_final.json", "frames"]

    for file in required_files:
        path = os.path.join(data_path, file)
        if os.path.exists(path):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")

    # Check objects.txt path
    objects_path = (
        "easg-generation/data/EASG/EASG/generation/annts_in_new_format/objects.txt"
    )
    if os.path.exists(objects_path):
        print(f"✓ objects.txt exists")
    else:
        print(f"✗ objects.txt missing at {objects_path}")

    # Load features
    feats_path = os.path.join(data_path, "features_verb.pt")
    if os.path.exists(feats_path):
        feats = torch.load(feats_path)
        print(f"✓ Loaded features with {len(feats)} keys")
        print(f"  Sample keys: {list(feats.keys())[:5]}")

        # Analyze key format
        sample_keys = list(feats.keys())[:10]
        print("\n=== Feature Key Analysis ===")
        for key in sample_keys:
            parts = key.split("_")
            print(f"Key: {key}")
            print(f"  Parts: {parts}")
            if len(parts) >= 4:
                print(f"  Clip ID: {parts[0]}")
                print(f"  Graph prefix: {parts[1]}")
                print(f"  Graph index: {parts[2]}")
                print(f"  Graph UID: {parts[3]}")
            print()
    else:
        print("✗ Cannot load features")
        return

    # Load annotations
    annts_path = os.path.join(data_path, "EASG_unict_master_final.json")
    if os.path.exists(annts_path):
        with open(annts_path, "rb") as f:
            annts = json.load(f)
        print(f"✓ Loaded annotations with {len(annts)} clips")

        # Check splits
        splits = {}
        for clip_id in annts:
            split = annts[clip_id].get("split", "unknown")
            splits[split] = splits.get(split, 0) + 1
        print(f"  Splits: {splits}")

        # Check a sample clip
        sample_clip = list(annts.keys())[0]
        print(f"\n=== Sample Clip Analysis ===")
        print(f"Sample clip {sample_clip}:")
        print(f"  Split: {annts[sample_clip].get('split')}")
        print(f"  Graphs: {len(annts[sample_clip].get('graphs', []))}")

        if annts[sample_clip].get("graphs"):
            for i, graph in enumerate(
                annts[sample_clip]["graphs"][:3]
            ):  # Check first 3 graphs
                print(f"\n  Graph {i}:")
                print(f"    UID: {graph.get('graph_uid')}")
                print(f"    Triplets: {len(graph.get('triplets', []))}")
                print(f"    Groundings: {list(graph.get('groundings', {}).keys())}")

                # Try different key formats
                graph_uid = graph.get("graph_uid")
                if graph_uid:
                    # Current format (incorrect)
                    key1 = f"{graph_uid}_{sample_clip}"
                    # Suspected correct format
                    key2 = f"{sample_clip}_graph_{i}_{graph_uid}"

                    print(f"    Current key format: {key1}")
                    print(f"    Suspected key format: {key2}")
                    print(f"    Current key exists: {key1 in feats}")
                    print(f"    Suspected key exists: {key2 in feats}")

                    # Check if frame files exist
                    for t in ["pre", "pnr", "post"]:
                        frame_path = os.path.join(
                            data_path, "frames", f"{graph_uid}/{sample_clip}_{t}.jpg"
                        )
                        if os.path.exists(frame_path):
                            print(f"      ✓ Frame {t}: exists")
                        else:
                            print(f"      ✗ Frame {t}: missing at {frame_path}")
    else:
        print("✗ Cannot load annotations")
        return


if __name__ == "__main__":
    debug_dataset()
