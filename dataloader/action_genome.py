import os
import pickle

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from fasterRCNN.lib.model.utils.blob import im_list_to_blob, prep_im_for_blob


class AG(Dataset):
    def __init__(
        self,
        mode,
        datasize,
        data_path=None,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
        print_stats=False,
    ):
        root_path = data_path
        self.frames_path = os.path.join(root_path, "frames/")

        print("-" * 60, "loading object classes", "-" * 60)
        self.object_classes = ["__background__"]
        with open(os.path.join(root_path, "annotations/object_classes.txt"), "r") as f:
            for line in f.readlines():
                line = line.strip("\n")
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = "closet/cabinet"
        self.object_classes[11] = "cup/glass/bottle"
        self.object_classes[23] = "paper/notebook"
        self.object_classes[24] = "phone/camera"
        self.object_classes[31] = "sofa/couch"

        print("-" * 60, "loading relationship classes", "-" * 60)
        self.relationship_classes = []
        with open(
            os.path.join(root_path, "annotations/relationship_classes.txt"), "r"
        ) as f:
            for line in f.readlines():
                line = line.strip("\n")
                self.relationship_classes.append(line)
        f.close()
        self.relationship_classes[0] = "looking_at"
        self.relationship_classes[1] = "not_looking_at"
        self.relationship_classes[5] = "in_front_of"
        self.relationship_classes[7] = "on_the_side_of"
        self.relationship_classes[10] = "covered_by"
        self.relationship_classes[11] = "drinking_from"
        self.relationship_classes[13] = "have_it_on_the_back"
        self.relationship_classes[15] = "leaning_on"
        self.relationship_classes[16] = "lying_on"
        self.relationship_classes[17] = "not_contacting"
        self.relationship_classes[18] = "other_relationship"
        self.relationship_classes[19] = "sitting_on"
        self.relationship_classes[20] = "standing_on"
        self.relationship_classes[25] = "writing_on"
        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]

        print("-" * 60, "loading annotations", "-" * 60)

        if filter_small_box:
            with open(
                os.path.join(root_path, "annotations/person_bbox.pkl"), "rb"
            ) as f:
                person_bbox = pickle.load(f)
            f.close()
            with open("data/object_bbox_and_relationship_filtersmall.pkl", "rb") as f:
                object_bbox = pickle.load(f)
        else:
            with open(
                os.path.join(root_path, "annotations/person_bbox.pkl"), "rb"
            ) as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(
                os.path.join(root_path, "annotations/object_bbox_and_relationship.pkl"),
                "rb",
            ) as f:
                object_bbox = pickle.load(f)
            f.close()

        print("-" * 60, "finish!", "-" * 60)

        if datasize == "mini":
            small_person = {}
            small_object = {}
            for i in list(person_bbox.keys())[:80000]:
                small_person[i] = person_bbox[i]
                small_object[i] = object_bbox[i]
            person_bbox = small_person
            object_bbox = small_object

        print(f"\n{'='*60} Processing {mode.upper()} Set {'='*60}")
        print("-" * 60, "collecting valid frames", "-" * 60)

        missing_frames = 0

        video_dict = {}
        for i in person_bbox.keys():
            if object_bbox[i][0]["metadata"]["set"] == mode:  # train or testing?
                frame_valid = False
                for j in object_bbox[i]:  # the frame is valid if there is visible bbox
                    if j["visible"]:
                        frame_valid = True
                if frame_valid:
                    # Check if the frame file actually exists
                    frame_path = os.path.join(self.frames_path, i)
                    if os.path.exists(frame_path):
                        video_name, frame_num = i.split("/")
                        if video_name in video_dict.keys():
                            video_dict[video_name].append(i)
                        else:
                            video_dict[video_name] = [i]
                    else:
                        missing_frames += 1

        self.video_list = []
        self.video_size = []  # (w,h)
        self.gt_annotations = []
        self.non_gt_human_nums = 0
        self.non_heatmap_nums = 0
        self.non_person_video = 0
        self.one_frame_video = 0
        self.valid_nums = 0

        """
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        """
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            for j in video_dict[i]:
                # Double-check that frame exists (should already be filtered, but being safe)
                frame_path = os.path.join(self.frames_path, j)
                if not os.path.exists(frame_path):
                    continue

                if filter_nonperson_box_frame:
                    if person_bbox[j]["bbox"].shape[0] == 0:
                        self.non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        self.valid_nums += 1

                gt_annotation_frame = [
                    {"person_bbox": person_bbox[j]["bbox"], "frame": j}
                ]
                # each frames's objects and human
                for k in object_bbox[j]:
                    if k["visible"]:
                        assert (
                            k["bbox"] is not None
                        ), "warning! The object is visible without bbox"
                        k["class"] = self.object_classes.index(k["class"])
                        k["bbox"] = np.array(
                            [
                                k["bbox"][0],
                                k["bbox"][1],
                                k["bbox"][0] + k["bbox"][2],
                                k["bbox"][1] + k["bbox"][3],
                            ]
                        )  # from xywh to xyxy
                        k["attention_relationship"] = torch.tensor(
                            [
                                self.attention_relationships.index(r)
                                for r in k["attention_relationship"]
                            ],
                            dtype=torch.long,
                        )
                        k["spatial_relationship"] = torch.tensor(
                            [
                                self.spatial_relationships.index(r)
                                for r in k["spatial_relationship"]
                            ],
                            dtype=torch.long,
                        )
                        k["contacting_relationship"] = torch.tensor(
                            [
                                self.contacting_relationships.index(r)
                                for r in k["contacting_relationship"]
                            ],
                            dtype=torch.long,
                        )
                        gt_annotation_frame.append(k)
                gt_annotation_video.append(gt_annotation_frame)

            if len(video) > 2:
                print(f"INCLUDE video {i} with {len(video)} valid frames")
                self.video_list.append(video)
                self.video_size.append(person_bbox[j]["bbox_size"])
                self.gt_annotations.append(gt_annotation_video)
            elif len(video) == 1:
                print(f"EXCLUDE video {i}: only 1 valid frame")
                self.one_frame_video += 1
            else:
                print(f"EXCLUDE video {i}: no valid frames")
                self.non_person_video += 1

        print("x" * 60)
        print(f"Filtered out {missing_frames} frames that do not exist on disk")
        if filter_nonperson_box_frame:
            print(
                f"There are {len(self.video_list)} videos and {self.valid_nums} valid frames in the {mode.upper()} set"
            )
            print(
                f"{self.non_person_video} videos are invalid (no person), remove them"
            )
            print(
                f"{self.one_frame_video} videos are invalid (only one frame), remove them"
            )
            print(
                f"{self.non_gt_human_nums} frames have no human bbox in GT, remove them!"
            )
        else:
            print(
                f"There are {len(self.video_list)} videos and {self.valid_nums} valid frames in the {mode.upper()} set"
            )
            print(f"{self.non_gt_human_nums} frames have no human bbox in GT")
            print(
                f"Removed {self.non_heatmap_nums} of them without joint heatmaps which means FasterRCNN also cannot find the human"
            )
        print("x" * 60)
        print(f"{'='*20} Finished Processing {mode.upper()} Set {'='*20}\n")

    def __getitem__(self, index):
        frame_names = self.video_list[index]
        processed_ims = []
        im_scales = []

        for idx, name in enumerate(frame_names):
            im = imageio.imread(os.path.join(self.frames_path, name))  # channel h,w,3
            im = im[:, :, ::-1]  # rgb -> bgr

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

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

        return img_tensor, im_info, gt_boxes, num_boxes, index

    def __len__(self):
        return len(self.video_list)


def cuda_collate_fn(batch):
    """
    Proper collate function that handles variable-length sequences
    and ensures consistent batch sizes for the model
    """
    # Since we're processing one video at a time, just return the first item
    # but ensure it's properly formatted
    if len(batch) == 0:
        raise ValueError("Empty batch received")

    # Get the first (and only) item
    item = batch[0]

    # Ensure all tensors are properly shaped
    img_tensor, im_info, gt_boxes, num_boxes, index = item

    # Validate tensor shapes
    if img_tensor.dim() != 4:  # [num_frames, channels, height, width]
        raise ValueError(f"Expected 4D image tensor, got {img_tensor.dim()}D")

    if im_info.dim() != 2:  # [num_frames, 3]
        raise ValueError(f"Expected 2D im_info tensor, got {im_info.dim()}D")

    if gt_boxes.dim() != 3:  # [num_frames, max_boxes, 5]
        raise ValueError(f"Expected 3D gt_boxes tensor, got {gt_boxes.dim()}D")

    if num_boxes.dim() != 1:  # [num_frames]
        raise ValueError(f"Expected 1D num_boxes tensor, got {num_boxes.dim()}D")

    # Ensure batch sizes are consistent
    num_frames = img_tensor.size(0)
    if im_info.size(0) != num_frames:
        raise ValueError(
            f"im_info batch size {im_info.size(0)} doesn't match image frames {num_frames}"
        )

    if gt_boxes.size(0) != num_frames:
        raise ValueError(
            f"gt_boxes batch size {gt_boxes.size(0)} doesn't match image frames {num_frames}"
        )

    if num_boxes.size(0) != num_frames:
        raise ValueError(
            f"num_boxes batch size {num_boxes.size(0)} doesn't match image frames {num_frames}"
        )

    return item
