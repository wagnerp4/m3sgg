"""
Spatio-Temporal Knowledge-Enhanced Transformer (STKET) for Scene Graph Generation.

This module implements the STKET model for video scene graph generation,
combining spatial and temporal reasoning with transformer architectures.
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to path for fasterRCNN imports
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from m3sgg.utils.draw_rectangles.draw_rectangles import draw_union_boxes
from m3sgg.utils.fpn.box_utils import center_size
from m3sgg.utils.word_vectors import obj_edge_vectors
from m3sgg.utils.transformer import transformer

from .transformer_stket import ensemble_decoder, spatial_encoder, temporal_decoder


class ObjectClassifier(nn.Module):
    """Module for computing object contexts and edge contexts in scene graphs.

    Handles object classification and contextual feature extraction for
    spatial-temporal transformer-based scene graph generation.

    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """

    def __init__(self, mode="sgdet", obj_classes=None):
        """Initialize the object classifier.

        :param mode: Classification mode ('predcls', 'sgcls', 'sgdet'), defaults to "sgdet"
        :type mode: str, optional
        :param obj_classes: List of object class names, defaults to None
        :type obj_classes: list, optional
        :return: None
        :rtype: None
        """
        super(ObjectClassifier, self).__init__()
        self.classes = obj_classes
        self.mode = mode

        # ----------add nms when sgdet
        self.nms_filter_duplicates = True
        self.max_per_img = 64
        self.thresh = 0.01

        # roi align
        self.RCNN_roi_align = ROIAlign((7, 7), 1.0 / 16.0, 0)

        embed_vecs = obj_edge_vectors(
            obj_classes[1:], wv_type="glove.6B", wv_dir="data", wv_dim=200
        )
        self.obj_embed = nn.Embedding(len(obj_classes) - 1, 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(
            nn.BatchNorm1d(4, momentum=0.01 / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.obj_dim = 2048
        self.decoder_lin = nn.Sequential(
            nn.Linear(self.obj_dim + 200 + 128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.decoder_lin2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, len(obj_classes) - 1),
        )

    def forward(self, entry):
        """Forward pass for object classification.

        :param entry: Dictionary containing input data
        :type entry: dict
        :return: Updated entry dictionary with predictions
        :rtype: dict
        """
        if self.mode == "predcls":
            obj_scores = entry["labels"]
            obj_boxes = entry["boxes"]
            obj_labels = entry["labels"]
        else:
            obj_scores = entry["scores"]
            obj_boxes = entry["boxes"]
            obj_labels = entry["labels"]

        if self.mode == "sgdet":
            # Apply NMS to object detections
            if self.nms_filter_duplicates:
                keep = nms(obj_boxes, obj_scores, self.thresh)
            else:
                keep = torch.arange(len(obj_boxes))
            obj_boxes = obj_boxes[keep]
            obj_scores = obj_scores[keep]
            obj_labels = obj_labels[keep]

        # Limit number of objects per image
        if len(obj_boxes) > self.max_per_img:
            keep = torch.argsort(obj_scores, descending=True)[: self.max_per_img]
            obj_boxes = obj_boxes[keep]
            obj_scores = obj_scores[keep]
            obj_labels = obj_labels[keep]

        # Extract features and compute predictions
        if "features" in entry:
            features = entry["features"]
            if self.mode == "sgdet":
                features = features[keep]
        else:
            features = entry["features"]

        # Positional encoding
        pos_feat = self.pos_embed(center_size(obj_boxes))

        # Object embeddings
        obj_embed = self.obj_embed(obj_labels - 1)

        # Combine features
        obj_feat = torch.cat([features, obj_embed, pos_feat], 1)
        obj_feat = self.decoder_lin(obj_feat)
        obj_feat = self.decoder_lin2(obj_feat)

        # Update entry
        entry["distribution"] = obj_feat
        entry["pred_labels"] = obj_labels
        if self.mode == "sgdet":
            entry["pred_scores"] = obj_scores

        return entry


class STKET(nn.Module):
    """Spatio-Temporal Knowledge-Enhanced Transformer for Scene Graph Generation.

    Implements the STKET model that combines spatial and temporal reasoning
    with transformer architectures for video scene graph generation.

    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """

    def __init__(
        self,
        mode="sgdet",
        attention_class_num=None,
        spatial_class_num=None,
        contact_class_num=None,
        obj_classes=None,
        rel_classes=None,
        N_layer_num=1,
        enc_layer_num=None,
        dec_layer_num=None,
        pred_contact_threshold=0.5,
        window_size=4,
        trainPrior=None,
        use_spatial_prior=False,
        use_temporal_prior=False,
    ):
        """Initialize the STKET model.

        :param mode: Classification mode ('sgdet', 'sgcls', 'predcls'), defaults to "sgdet"
        :type mode: str, optional
        :param attention_class_num: Number of attention relationship classes, defaults to None
        :type attention_class_num: int, optional
        :param spatial_class_num: Number of spatial relationship classes, defaults to None
        :type spatial_class_num: int, optional
        :param contact_class_num: Number of contact relationship classes, defaults to None
        :type contact_class_num: int, optional
        :param obj_classes: List of object class names, defaults to None
        :type obj_classes: list, optional
        :param rel_classes: List of relationship class names, defaults to None
        :type rel_classes: list, optional
        :param N_layer_num: Number of transformer layers, defaults to 1
        :type N_layer_num: int, optional
        :param enc_layer_num: Number of encoder layers, defaults to None
        :type enc_layer_num: int, optional
        :param dec_layer_num: Number of decoder layers, defaults to None
        :type dec_layer_num: int, optional
        :param pred_contact_threshold: Contact prediction threshold, defaults to 0.5
        :type pred_contact_threshold: float, optional
        :param window_size: Temporal window size, defaults to 4
        :type window_size: int, optional
        :param trainPrior: Training prior information, defaults to None
        :type trainPrior: dict, optional
        :param use_spatial_prior: Whether to use spatial priors, defaults to False
        :type use_spatial_prior: bool, optional
        :param use_temporal_prior: Whether to use temporal priors, defaults to False
        :type use_temporal_prior: bool, optional
        :return: None
        :rtype: None
        """
        super(STKET, self).__init__()

        assert mode in ("sgdet", "sgcls", "predcls")

        self.mode = mode

        self.obj_classes = obj_classes
        self.object_classifier = ObjectClassifier(
            mode=self.mode, obj_classes=self.obj_classes
        )

        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num

        ###################################
        self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256 // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        self.subj_fc = nn.Linear(2048, 512)
        self.obj_fc = nn.Linear(2048, 512)
        self.vr_fc = nn.Linear(256 * 7 * 7, 512)

        embed_vecs = obj_edge_vectors(
            obj_classes, wv_type="glove.6B", wv_dir="data", wv_dim=200
        )
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

        self.N_layer_num = N_layer_num
        self.pred_contact_threshold = pred_contact_threshold
        self.window_size = window_size

        self.use_spatial_prior = use_spatial_prior
        self.use_temporal_prior = use_temporal_prior

        self.enc_layer_num = enc_layer_num
        self.dec_layer_num = dec_layer_num

        if enc_layer_num > 0:
            self.spatial_encoder = spatial_encoder(
                enc_layer_num=enc_layer_num,
                embed_dim=1936,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                trainPrior=trainPrior,
                use_spatial_prior=use_spatial_prior,
                obj_class_num=len(self.obj_classes),
                attention_class_num=self.attention_class_num,
                spatial_class_num=self.spatial_class_num,
                contact_class_num=self.contact_class_num,
            )

        if dec_layer_num > 0:
            self.temporal_decoder = temporal_decoder(
                dec_layer_num=dec_layer_num,
                embed_dim=1936,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                pred_contact_threshold=pred_contact_threshold,
                trainPrior=trainPrior,
                use_temporal_prior=use_temporal_prior,
                obj_class_num=len(self.obj_classes),
                attention_class_num=self.attention_class_num,
                spatial_class_num=self.spatial_class_num,
                contact_class_num=self.contact_class_num,
            )

        if (enc_layer_num > 0) and (dec_layer_num > 0):
            self.ensemble_decoder = ensemble_decoder(
                dec_layer_num=2,
                embed_dim=1936,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                pred_contact_threshold=pred_contact_threshold,
                window_size=window_size,
                obj_class_num=len(self.obj_classes),
                attention_class_num=self.attention_class_num,
                spatial_class_num=self.spatial_class_num,
                contact_class_num=self.contact_class_num,
            )

        self.a_rel_compress = nn.Linear(1936, self.attention_class_num)
        self.s_rel_compress = nn.Linear(1936, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(1936, self.contact_class_num)

        self.spa_a_rel_compress = nn.Linear(1936, self.attention_class_num)
        self.spa_s_rel_compress = nn.Linear(1936, self.spatial_class_num)
        self.spa_c_rel_compress = nn.Linear(1936, self.contact_class_num)

        self.tem_a_rel_compress = nn.Linear(1936, self.attention_class_num)
        self.tem_s_rel_compress = nn.Linear(1936, self.spatial_class_num)
        self.tem_c_rel_compress = nn.Linear(1936, self.contact_class_num)

        self.ens_a_rel_compress = nn.Linear(1936, self.attention_class_num)
        self.ens_s_rel_compress = nn.Linear(1936, self.spatial_class_num)
        self.ens_c_rel_compress = nn.Linear(1936, self.contact_class_num)

        print(
            "[STKET] Mode: {}, Pred_contact_threshold: {:.1f}, window_size: {:d}".format(
                self.mode, self.pred_contact_threshold, self.window_size
            )
        )

    def forward(self, entry):
        """Forward pass for STKET model.

        :param entry: Dictionary containing input data
        :type entry: dict
        :return: Updated entry dictionary with predictions
        :rtype: dict
        """
        # elements in entry:
        # boxes: tensor, (all_bbox_nums, 5)
        # labels: tensor, (all_bbox_nums,)
        # scores: tensor, (all_bbox_nums,)
        # im_idx: tensor, (relation_nums,)
        # pair_idx: tensor, (pair_nums, 2)
        # human_idx: tensor, (person_nums,)
        # features: tensor, (all_bbox_nums, 2048)
        # union_feat: tensor, (pair_num, 1024, 7, 7)
        # union_box: tensor, (pair_nums, 5)
        # spatial_masks:
        # attention_gt: list, (relation_nums,). e.g., [[0], [2], [3]]
        # spatial_gt: list, (relation_nums,). e.g., [[5], [3, 4], [3]]
        # contacting_gt: list, (relation_nums,). e.g., [[8], [10, 6], [8]]

        entry = self.object_classifier(entry)

        # Add two elements in entry:
        # distribution: tensor, (all_bbox_nums, 37)
        # pred_labels: tensor, (all_bbox_nums,)

        # Add one elements in entry when testing
        # pred_scores: tensor, (all_bbox_nums,)

        # visual part
        subj_rep = entry["features"][entry["pair_idx"][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry["features"][entry["pair_idx"][:, 1]]
        obj_rep = self.obj_fc(obj_rep)
        vr = self.union_func1(entry["union_feat"]) + self.conv(entry["spatial_masks"])
        vr = self.vr_fc(vr.view(-1, 256 * 7 * 7))
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)

        # x_visual: tensor, (pair_nums, 512 + 512 + 512)

        # semantic part
        subj_class = entry["pred_labels"][entry["pair_idx"][:, 0]]
        obj_class = entry["pred_labels"][entry["pair_idx"][:, 1]]
        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        x_semantic = torch.cat(
            (subj_emb, obj_emb), 1
        )  # x_semantic: tensor, (pair_nums, 200 + 200)

        rel_features = torch.cat(
            (x_visual, x_semantic), dim=1
        )  # rel_features: tensor, (pair_nums, 512 + 512 + 512 + 200 + 200)

        # Predict by spatial features
        entry["attention_distribution"] = self.a_rel_compress(rel_features)
        entry["spatial_distribution"] = self.s_rel_compress(rel_features)
        entry["contact_distribution"] = self.c_rel_compress(rel_features)

        entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
        entry["contact_distribution"] = torch.sigmoid(entry["contact_distribution"])

        for _ in range(self.N_layer_num):
            # Spatial Encoder
            if self.enc_layer_num > 0:
                spatial_features, spatial_priors = self.spatial_encoder(
                    features=rel_features,
                    im_idx=entry["im_idx"],
                    entry=entry,
                    mode=self.mode,
                )

                # Predict by spatial features
                entry["spatial_attention_distribution"] = self.spa_a_rel_compress(
                    spatial_features
                )
                entry["spatial_spatial_distribution"] = self.spa_s_rel_compress(
                    spatial_features
                )
                entry["spatial_contact_distribution"] = self.spa_c_rel_compress(
                    spatial_features
                )

                entry["spatial_spatial_distribution"] = torch.sigmoid(
                    entry["spatial_spatial_distribution"]
                )
                entry["spatial_contact_distribution"] = torch.sigmoid(
                    entry["spatial_contact_distribution"]
                )

                # Predict by spatial priors
                if self.use_spatial_prior:
                    entry["spatial_prior_attention_distribution"] = (
                        self.spa_a_rel_compress(spatial_priors)
                    )
                    entry["spatial_prior_spatial_distribution"] = (
                        self.spa_s_rel_compress(spatial_priors)
                    )
                    entry["spatial_prior_contact_distribution"] = (
                        self.spa_c_rel_compress(spatial_priors)
                    )

                    entry["spatial_prior_spatial_distribution"] = torch.sigmoid(
                        entry["spatial_prior_spatial_distribution"]
                    )
                    entry["spatial_prior_contact_distribution"] = torch.sigmoid(
                        entry["spatial_prior_contact_distribution"]
                    )

            contact_distribution = (
                entry["spatial_contact_distribution"]
                if self.enc_layer_num > 0
                else entry["contact_distribution"]
            )

            # Temporal Decoder
            if self.dec_layer_num > 0:
                temporal_features, temporal_priors = self.temporal_decoder(
                    features=spatial_features
                    if self.enc_layer_num > 0
                    else rel_features,
                    contact_distribution=contact_distribution,
                    im_idx=entry["im_idx"],
                    entry=entry,
                    mode=self.mode,
                )

                # Predict by temporal features
                entry["temporal_attention_distribution"] = self.tem_a_rel_compress(
                    temporal_features
                )
                entry["temporal_spatial_distribution"] = self.tem_s_rel_compress(
                    temporal_features
                )
                entry["temporal_contact_distribution"] = self.tem_c_rel_compress(
                    temporal_features
                )

                entry["temporal_spatial_distribution"] = torch.sigmoid(
                    entry["temporal_spatial_distribution"]
                )
                entry["temporal_contact_distribution"] = torch.sigmoid(
                    entry["temporal_contact_distribution"]
                )

                # Predict by temporal priors
                if self.use_temporal_prior:
                    entry["temporal_prior_attention_distribution"] = (
                        self.tem_a_rel_compress(temporal_priors)
                    )
                    entry["temporal_prior_spatial_distribution"] = (
                        self.tem_s_rel_compress(temporal_priors)
                    )
                    entry["temporal_prior_contact_distribution"] = (
                        self.tem_c_rel_compress(temporal_priors)
                    )

                    entry["temporal_prior_spatial_distribution"] = torch.sigmoid(
                        entry["temporal_prior_spatial_distribution"]
                    )
                    entry["temporal_prior_contact_distribution"] = torch.sigmoid(
                        entry["temporal_prior_contact_distribution"]
                    )

            rel_features = spatial_features if self.enc_layer_num > 0 else rel_features

        if (self.enc_layer_num > 0) and (self.dec_layer_num > 0):
            contact_distribution = (
                entry["spatial_contact_distribution"]
                if self.enc_layer_num > 0
                else entry["contact_distribution"]
            )
            ensemble_features = self.ensemble_decoder(
                spatial_features=spatial_features,
                temporal_features=temporal_features,
                contact_distribution=contact_distribution,
                im_idx=entry["im_idx"],
                entry=entry,
                mode=self.mode,
            )

            # ensemble
            entry["ensemble_attention_distribution"] = self.ens_a_rel_compress(
                ensemble_features
            )
            entry["ensemble_spatial_distribution"] = self.ens_s_rel_compress(
                ensemble_features
            )
            entry["ensemble_contact_distribution"] = self.ens_c_rel_compress(
                ensemble_features
            )

            entry["ensemble_spatial_distribution"] = torch.sigmoid(
                entry["ensemble_spatial_distribution"]
            )
            entry["ensemble_contact_distribution"] = torch.sigmoid(
                entry["ensemble_contact_distribution"]
            )

        return entry
