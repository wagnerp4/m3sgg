import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Add project root to path for fasterRCNN imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fasterRCNN.lib.model.utils.blob import im_list_to_blob, prep_im_for_blob


class EASG(Dataset):
    def __init__(self, split, datasize, data_path=None):
        root_path = data_path
        self.frames_path = os.path.join(root_path, "frames")

        self.obj_classes = ["__background__"]
        with open(f"{root_path}/EASG/generation/annts_in_new_format/objects.txt") as f:
            for line in f:
                self.obj_classes.append(line.strip("\n"))

        # Add missing attributes to match Action Genome interface
        self.object_classes = self.obj_classes  # Alias for compatibility
        self.relationship_classes = []  # EASG doesn't use traditional relationships
        self.attention_relationships = []  # EASG doesn't use attention relationships
        self.spatial_relationships = []  # EASG doesn't use spatial relationships
        self.contacting_relationships = []  # EASG doesn't use contacting relationships

        self.verb_feats = []
        self.verb_classes = []
        self.edge_classes = []
        self.video_list = []
        self.video_size = []
        self.gt_groundings = []

        missing_feature_count = 0

        feats = torch.load(os.path.join(root_path, "features_verb.pt"))

        with open(os.path.join(root_path, "EASG_unict_master_final.json"), "rb") as f:
            annts = json.load(f)

        for clip_id in annts:
            for graph in annts[clip_id]["graphs"]:
                for triplet in graph["triplets"]:
                    n1, e, n2 = triplet
                    if n1 == "CW":
                        assert e == "verb"
                        if n2 not in self.verb_classes:
                            self.verb_classes.append(n2)
                    else:
                        if ":" in n2:
                            n2 = n2.split(":")[0]

                        if n2 not in self.obj_classes:
                            continue
                        if e not in self.edge_classes:
                            self.edge_classes.append(e)

        for clip_id in annts:
            if annts[clip_id]["split"] != split:
                continue

            video_size = (annts[clip_id]["W"], annts[clip_id]["H"])

            num_frames = 0
            video = []
            feat = []
            gt_grounding = []
            for graph in annts[clip_id]["graphs"]:
                graph_uid = graph["graph_uid"]
                obj_to_edge = {}
                for triplet in graph["triplets"]:
                    n1, e, n2 = triplet
                    if n1 == "CW":
                        verb = n2
                    else:
                        if ":" in n2:
                            n2 = n2.split(":")[0]

                        if n2 not in self.obj_classes:
                            continue
                        if n2 not in obj_to_edge:
                            obj_to_edge[n2] = []

                        if e not in obj_to_edge[n2]:
                            obj_to_edge[n2].append(e)

                grounding_t = {}
                grounding_t["pre"] = []
                grounding_t["pnr"] = []
                grounding_t["post"] = []
                for t in ["pre", "pnr", "post"]:
                    if t not in graph["groundings"]:
                        continue

                    for n in graph["groundings"][t]:
                        if n not in obj_to_edge:
                            # Here we ignore the mismatched graphs/groundings
                            continue
                        # Extract bounding box coordinates for the object
                        g = graph["groundings"][t][n]
                        x, y, w, h = g["left"], g["top"], g["width"], g["height"]
                        bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
                        grounding_t[t].append(
                            {
                                "obj": self.obj_classes.index(n) - 1,
                                "bbox": bbox,
                                "verb": self.verb_classes.index(verb),
                                "edge": sorted(
                                    [self.edge_classes.index(e) for e in obj_to_edge[n]]
                                ),
                            }
                        )
                        # FIX
                        # Old approach (different file naming):
                        # key = f'{graph_uid}_{clip_id}'

                        # The feature keys use clip_id as both the graph UID and clip ID
                        # Format: {clip_id}_graph_{graph_index}_{clip_id}
                        # We need to find the correct graph index for this graph
                        graph_index = None
                        for i, g in enumerate(annts[clip_id]["graphs"]):
                            if g["graph_uid"] == graph_uid:
                                graph_index = i
                                break

                        if graph_index is None:
                            missing_feature_count += 1
                            continue

                        key = f"{clip_id}_graph_{graph_index}_{clip_id}"
                        if key not in feats:
                            # print(f"[WARNING] Missing feature for key: {key}")
                            missing_feature_count += 1
                            continue
                        feat.append(feats[key])

                for t in ["pre", "pnr", "post"]:
                    if not grounding_t[t]:
                        continue

                    # print('{}/{}_{}.jpg'.format(graph_uid, clip_id, t))
                    abs_path = os.path.join(
                        self.frames_path, "{}/{}_{}.jpg".format(graph_uid, clip_id, t)
                    )
                    # print(abs_path)
                    if not os.path.exists(abs_path):
                        print(f"[WARNING] Missing frame: {abs_path}")
                        continue
                    video.append(abs_path)
                    gt_grounding.append(grounding_t[t])
                    num_frames += 1

                if num_frames >= 100:
                    if feat:  # Only append if feat is not empty
                        self.video_list.append(video)
                        self.video_size.append(video_size)
                        self.verb_feats.append(torch.stack(feat))
                        self.gt_groundings.append(gt_grounding)
                    else:
                        # TODO: Log or handle cases where all features for a video are missing (video skipped)
                        pass
                    video = []
                    feat = []
                    gt_grounding = []
                    num_frames = 0

            if num_frames > 0:
                if feat:  # Only append if feat is not empty
                    self.video_list.append(video)
                    self.video_size.append(video_size)
                    self.verb_feats.append(torch.stack(feat))
                    self.gt_groundings.append(gt_grounding)
                else:
                    # TODO: Log or handle cases where all features for a video are missing (video skipped)
                    pass

        if self.video_list:
            print(
                "There are {} videos and {} maximum number of frames".format(
                    len(self.video_list), max([len(v) for v in self.video_list])
                )
            )
        else:
            print(
                "No videos found after filtering. Please check your data and features."
            )
            # TODO: Handle empty dataset case if needed

        print("--------------------finish!-------------------------")
        print(f"Total missing features: {missing_feature_count}")

    def __getitem__(self, index):
        frame_names = self.video_list[index]
        processed_ims = []
        im_scales = []

        for idx, name in enumerate(frame_names):
            img_path = name
            im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if im is None:
                print(f"[ERROR] Failed to load image: {img_path}")
                raise FileNotFoundError(f"Could not load image: {img_path}")

            im, im_scale = prep_im_for_blob(
                im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000
            )  # cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            im_scales.append(im_scale)
            processed_ims.append(im)

        blob = im_list_to_blob(processed_ims)
        im_info = np.array(
            [[blob.shape[1], blob.shape[2], im_scales[0]]], dtype=np.float32
        )
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        return img_tensor, im_info, index

    def __len__(self):
        return len(self.video_list)


def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]
