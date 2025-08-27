import json
import os

import torch

# Paths (adjust if needed)
root_path = os.path.join("data", "EASG")
features_path = os.path.join(root_path, "verb_features.pt")
annotations_path = os.path.join(root_path, "EASG_unict_master_final.json")

# Load features
feats = torch.load(features_path)
feature_keys = set(feats.keys())

# Load annotations
with open(annotations_path, "r") as f:
    annts = json.load(f)

annotation_keys = set()
for clip_id in annts:
    for graph in annts[clip_id]["graphs"]:
        graph_uid = graph["graph_uid"]
        key = f"{graph_uid}_{clip_id}"
        annotation_keys.add(key)

# Check matches
matches = annotation_keys & feature_keys
missing = annotation_keys - feature_keys

print(f"Total annotation keys: {len(annotation_keys)}")
print(f"Total feature keys: {len(feature_keys)}")
print(f"Matches: {len(matches)}")
print(f"Missing: {len(missing)}")

if missing:
    print("Sample missing keys:")
    for k in list(missing)[:10]:
        print(k)
else:
    print("All annotation keys have corresponding features.")
