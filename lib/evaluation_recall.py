from functools import reduce

import numpy as np
import torch
import torch.nn as nn

from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from lib.ults.pytorch_misc import argsort_desc, intersect_2d


class BasicSceneGraphEvaluator:
    """Evaluator for scene graph generation tasks.
    
    Computes recall metrics for scene graph generation across different tasks
    (predcls, sgcls, sgdet) and handles constraint evaluation modes.
    
    :param object: Base object class
    :type object: class
    """
    def __init__(
        self,
        mode,
        AG_object_classes,
        AG_all_predicates,
        AG_attention_predicates,
        AG_spatial_predicates,
        AG_contacting_predicates,
        iou_threshold=0.5,
        constraint=False,
        semithreshold=None,
        logger=None,
    ):
        """Initialize the scene graph evaluator.
        
        :param mode: Evaluation mode ('predcls', 'sgcls', or 'sgdet')
        :type mode: str
        :param AG_object_classes: List of object class names
        :type AG_object_classes: list
        :param AG_all_predicates: List of all predicate names
        :type AG_all_predicates: list
        :param AG_attention_predicates: List of attention predicate names
        :type AG_attention_predicates: list
        :param AG_spatial_predicates: List of spatial predicate names
        :type AG_spatial_predicates: list
        :param AG_contacting_predicates: List of contacting predicate names
        :type AG_contacting_predicates: list
        :param iou_threshold: IoU threshold for evaluation, defaults to 0.5
        :type iou_threshold: float, optional
        :param constraint: Whether to use constraint evaluation, defaults to False
        :type constraint: bool, optional
        :param semithreshold: Semi-constraint threshold, defaults to None
        :type semithreshold: float, optional
        :param logger: Logger instance, defaults to None
        :type logger: logging.Logger, optional
        :return: None
        :rtype: None
        """
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + "_recall"] = {10: [], 20: [], 50: [], 100: []}
        self.result_dict[self.mode + "_recall_hit"] = {
            10: [0] * len(AG_all_predicates),
            20: [0] * len(AG_all_predicates),
            50: [0] * len(AG_all_predicates),
            100: [0] * len(AG_all_predicates),
        }
        self.result_dict[self.mode + "_recall_count"] = {
            10: [0] * len(AG_all_predicates),
            20: [0] * len(AG_all_predicates),
            50: [0] * len(AG_all_predicates),
            100: [0] * len(AG_all_predicates),
        }
        self.constraint = constraint  # semi constraint if True
        self.iou_threshold = iou_threshold
        self.AG_object_classes = AG_object_classes
        self.AG_all_predicates = AG_all_predicates
        self.AG_attention_predicates = AG_attention_predicates
        self.AG_spatial_predicates = AG_spatial_predicates
        self.AG_contacting_predicates = AG_contacting_predicates
        self.semithreshold = semithreshold
        self.per_class_recall = {}
        self.logger = logger
        # Tempura specific variables
        self.tot_all_predicates = len(AG_all_predicates)
        self.gt_obj_list = []
        self.pred_obj_list = []

    def reset_result(self):
        self.result_dict[self.mode + "_recall"] = {10: [], 20: [], 50: [], 100: []}

    def calc_mrecall(self):
        for k, v in self.result_dict[self.mode + "_recall"].items():
            avg = 0
            self.per_class_recall[k] = {}
            for idx in range(self.tot_all_predicates):
                tmp_avg = float(
                    self.result_dict[self.mode + "_recall_hit"][k][idx]
                ) / float(self.result_dict[self.mode + "_recall_count"][k][idx] + 1e-10)

                avg += tmp_avg
                self.per_class_recall[k][self.AG_all_predicates[idx]] = tmp_avg
            if (self.mode + "_Mrecall") not in self.result_dict:
                self.result_dict[self.mode + "_Mrecall"] = {}
            self.result_dict[self.mode + "_Mrecall"][k] = avg / self.tot_all_predicates

        return self.result_dict[self.mode + "_Mrecall"]

    def print_stats(self):
        if self.logger is not None:
            self.logger.info(f"Evaluating {self.constraint} constraint, {self.mode}")
            for k, v in self.result_dict[self.mode + "_recall"].items():
                self.logger.info("R@%i: %f" % (k, np.mean(v)))

            if (self.mode + "_Mrecall") in self.result_dict:
                for k, v in self.result_dict[self.mode + "_Mrecall"].items():
                    self.logger.info("MR@%i: %f" % (k, v))
        else:
            print(f"Evaluating {self.constraint} constraint, {self.mode}")
            for k, v in self.result_dict[self.mode + "_recall"].items():
                print("R@%i: %f" % (k, np.mean(v)))

            if (self.mode + "_Mrecall") in self.result_dict:
                for k, v in self.result_dict[self.mode + "_Mrecall"].items():
                    print("MR@%i: %f" % (k, v))

    def evaluate_scene_graph(self, gt, pred, return_per_sample=False):
        """collect the groundtruth and prediction"""

        per_sample_metrics = [] if return_per_sample else None

        pred["attention_distribution"] = nn.functional.softmax(
            pred["attention_distribution"], dim=1
        )

        for idx, frame_gt in enumerate(gt):
            # generate the ground truth
            gt_boxes = np.zeros(
                [len(frame_gt), 4]
            )  # now there is no person box! we assume that person box index == 0
            gt_classes = np.zeros(len(frame_gt))
            gt_relations = []
            human_idx = 0
            gt_classes[human_idx] = 1
            gt_boxes[human_idx] = frame_gt[0]["person_bbox"]
            for m, n in enumerate(frame_gt[1:]):
                # each pair
                gt_boxes[m + 1, :] = n["bbox"]
                gt_classes[m + 1] = n["class"]
                gt_relations.append(
                    [
                        human_idx,
                        m + 1,
                        self.AG_all_predicates.index(
                            self.AG_attention_predicates[n["attention_relationship"]]
                        ),
                    ]
                )  # for attention triplet <human-object-predicate>_
                # spatial and contacting relationship could be multiple
                for spatial in n["spatial_relationship"].numpy().tolist():
                    gt_relations.append(
                        [
                            m + 1,
                            human_idx,
                            self.AG_all_predicates.index(
                                self.AG_spatial_predicates[spatial]
                            ),
                        ]
                    )  # for spatial triplet <object-human-predicate>
                for contact in n["contacting_relationship"].numpy().tolist():
                    gt_relations.append(
                        [
                            human_idx,
                            m + 1,
                            self.AG_all_predicates.index(
                                self.AG_contacting_predicates[contact]
                            ),
                        ]
                    )  # for contact triplet <human-object-predicate>

            gt_entry = {
                "gt_classes": gt_classes,
                "gt_relations": np.array(gt_relations),
                "gt_boxes": gt_boxes,
            }

            # first part for attention and contact, second for spatial

            rels_i = np.concatenate(
                (
                    pred["pair_idx"][pred["im_idx"] == idx]
                    .cpu()
                    .clone()
                    .numpy(),  # attention
                    pred["pair_idx"][pred["im_idx"] == idx]
                    .cpu()
                    .clone()
                    .numpy()[:, ::-1],  # spatial
                    pred["pair_idx"][pred["im_idx"] == idx].cpu().clone().numpy(),
                ),
                axis=0,
            )  # contacting

            pred_scores_1 = np.concatenate(
                (
                    pred["attention_distribution"][pred["im_idx"] == idx].cpu().numpy(),
                    np.zeros(
                        [
                            pred["pair_idx"][pred["im_idx"] == idx].shape[0],
                            pred["spatial_distribution"].shape[1],
                        ]
                    ),
                    np.zeros(
                        [
                            pred["pair_idx"][pred["im_idx"] == idx].shape[0],
                            pred["contact_distribution"].shape[1],
                        ]
                    ),
                ),
                axis=1,
            )

            pred_scores_2 = np.concatenate(
                (
                    np.zeros(
                        [
                            pred["pair_idx"][pred["im_idx"] == idx].shape[0],
                            pred["attention_distribution"].shape[1],
                        ]
                    ),
                    pred["spatial_distribution"][pred["im_idx"] == idx].cpu().numpy(),
                    np.zeros(
                        [
                            pred["pair_idx"][pred["im_idx"] == idx].shape[0],
                            pred["contact_distribution"].shape[1],
                        ]
                    ),
                ),
                axis=1,
            )

            pred_scores_3 = np.concatenate(
                (
                    np.zeros(
                        [
                            pred["pair_idx"][pred["im_idx"] == idx].shape[0],
                            pred["attention_distribution"].shape[1],
                        ]
                    ),
                    np.zeros(
                        [
                            pred["pair_idx"][pred["im_idx"] == idx].shape[0],
                            pred["spatial_distribution"].shape[1],
                        ]
                    ),
                    pred["contact_distribution"][pred["im_idx"] == idx].cpu().numpy(),
                ),
                axis=1,
            )

            if self.mode == "predcls":
                pred_entry = {
                    "pred_boxes": pred["boxes"][:, 1:].cpu().clone().numpy(),
                    "pred_classes": pred["labels"].cpu().clone().numpy(),
                    "pred_rel_inds": rels_i,
                    "obj_scores": pred["scores"].cpu().clone().numpy(),
                    "rel_scores": np.concatenate(
                        (pred_scores_1, pred_scores_2, pred_scores_3), axis=0
                    ),
                }
            else:  # FIX KeyError: 'pred_scores'
                pred_entry = {
                    "pred_boxes": pred["boxes"][:, 1:].cpu().clone().numpy(),
                    "pred_classes": pred["pred_labels"].cpu().clone().numpy(),
                    "pred_rel_inds": rels_i,
                    "obj_scores": pred["pred_scores"].cpu().clone().numpy(),
                    "rel_scores": np.concatenate(
                        (pred_scores_1, pred_scores_2, pred_scores_3), axis=0
                    ),
                }

            if return_per_sample:
                # Calculate per-sample metrics
                sample_metrics = self._evaluate_single_sample(
                    gt_entry,
                    pred_entry,
                    self.mode,
                    self.iou_threshold,
                    self.constraint,
                    self.semithreshold,
                )
                per_sample_metrics.append(sample_metrics)
            else:
                # Original behavior - accumulate global metrics
                evaluate_from_dict(
                    gt_entry,
                    pred_entry,
                    self.mode,
                    self.result_dict,
                    iou_thresh=self.iou_threshold,
                    method=self.constraint,
                    threshold=self.semithreshold,
                )

        return per_sample_metrics

    def _evaluate_single_sample(
        self, gt_entry, pred_entry, mode, iou_thresh, method, threshold
    ):
        """Evaluate a single sample and return per-sample metrics"""
        # Create a temporary result dict for this sample
        temp_result_dict = {mode + "_recall": {10: [], 20: [], 50: [], 100: []}}

        # Initialize per-class tracking for this sample
        temp_result_dict[mode + "_recall_hit"] = {
            10: [0] * self.tot_all_predicates,
            20: [0] * self.tot_all_predicates,
            50: [0] * self.tot_all_predicates,
            100: [0] * self.tot_all_predicates,
        }
        temp_result_dict[mode + "_recall_count"] = {
            10: [0] * self.tot_all_predicates,
            20: [0] * self.tot_all_predicates,
            50: [0] * self.tot_all_predicates,
            100: [0] * self.tot_all_predicates,
        }

        # Evaluate this single sample
        evaluate_from_dict(
            gt_entry,
            pred_entry,
            mode,
            temp_result_dict,
            iou_thresh=iou_thresh,
            method=method,
            threshold=threshold,
        )

        # Extract metrics for this sample
        sample_metrics = {}
        for k in [10, 20, 50, 100]:
            if temp_result_dict[mode + "_recall"][k]:
                sample_metrics[f"r{k}"] = temp_result_dict[mode + "_recall"][k][0]
            else:
                sample_metrics[f"r{k}"] = 0.0

        # Calculate mrecall for this sample
        # Compute per-class recall and average them
        per_class_recalls = []
        for pred_class in range(self.tot_all_predicates):
            if temp_result_dict[mode + "_recall_count"][20][pred_class] > 0:
                class_recall = (
                    temp_result_dict[mode + "_recall_hit"][20][pred_class]
                    / temp_result_dict[mode + "_recall_count"][20][pred_class]
                )
                per_class_recalls.append(class_recall)

        if per_class_recalls:
            sample_metrics["mrecall"] = np.mean(per_class_recalls)
        else:
            sample_metrics["mrecall"] = 0.0

        return sample_metrics


def evaluate_from_dict(
    gt_entry, pred_entry, mode, result_dict, method=None, threshold=0.9, **kwargs
):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param result_dict:
    :param kwargs:
    :return:
    """
    gt_rels = gt_entry["gt_relations"]
    gt_boxes = gt_entry["gt_boxes"].astype(float)
    gt_classes = gt_entry["gt_classes"]

    pred_rel_inds = pred_entry["pred_rel_inds"]
    rel_scores = pred_entry["rel_scores"]

    pred_boxes = pred_entry["pred_boxes"].astype(float)
    pred_classes = pred_entry["pred_classes"]
    obj_scores = pred_entry["obj_scores"]

    if method == "semi":
        pred_rels = []
        predicate_scores = []
        for i, j in enumerate(pred_rel_inds):
            if rel_scores[i, 0] + rel_scores[i, 1] > 0:
                # this is the attention distribution
                pred_rels.append(np.append(j, rel_scores[i].argmax()))
                predicate_scores.append(rel_scores[i].max())
            elif rel_scores[i, 3] + rel_scores[i, 4] > 0:
                # this is the spatial distribution
                for k in np.where(rel_scores[i] > threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i, k])
            elif rel_scores[i, 9] + rel_scores[i, 10] > 0:
                # this is the contact distribution
                for k in np.where(rel_scores[i] > threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i, k])

        pred_rels = np.array(pred_rels)
        predicate_scores = np.array(predicate_scores)
    elif method == "no":
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:, None] * rel_scores
        score_inds = argsort_desc(overall_scores)[:100]
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
        predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]

    else:
        pred_rels = np.column_stack(
            (pred_rel_inds, rel_scores.argmax(1))
        )  # 1+  dont add 1 because no dummy 'no relations'
        predicate_scores = rel_scores.max(1)

    pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
        gt_rels,
        gt_boxes,
        gt_classes,
        pred_rels,
        pred_boxes,
        pred_classes,
        predicate_scores,
        obj_scores,
        phrdet=mode == "phrdet",
        **kwargs,
    )

    # Track per-class recall statistics for mrecall calculation
    if mode + "_recall_hit" in result_dict and mode + "_recall_count" in result_dict:
        # Get unique predicate classes from ground truth
        gt_predicates = gt_rels[:, 2]
        unique_predicates = np.unique(gt_predicates)

        for k in result_dict[mode + "_recall"]:
            match = reduce(np.union1d, pred_to_gt[:k])

            # Overall recall
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            result_dict[mode + "_recall"][k].append(rec_i)

            # Per-class recall statistics
            for pred_class in unique_predicates:
                # Count how many GT relations of this class were matched
                gt_indices_of_class = np.where(gt_predicates == pred_class)[0]
                matched_indices_of_class = [
                    i for i in match if i in gt_indices_of_class
                ]

                # Update hit count for this class
                result_dict[mode + "_recall_hit"][k][pred_class] += len(
                    matched_indices_of_class
                )
                # Update total count for this class
                result_dict[mode + "_recall_count"][k][pred_class] += len(
                    gt_indices_of_class
                )
    else:
        # Fallback to original behavior if per-class tracking not initialized
        for k in result_dict[mode + "_recall"]:
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            result_dict[mode + "_recall"][k].append(rec_i)

    return pred_to_gt, pred_5ples, rel_scores


def evaluate_recall(
    gt_rels,
    gt_boxes,
    gt_classes,
    pred_rels,
    pred_boxes,
    pred_classes,
    rel_scores=None,
    cls_scores=None,
    iou_thresh=0.5,
    phrdet=False,
):
    """Evaluates recall metrics for scene graph generation.
    
    Computes recall by matching predicted relations to ground truth relations
    based on object detection IoU and relation class matching.
    
    :param gt_rels: Ground truth relations array of shape [#gt_rel, 3]
    :type gt_rels: numpy.ndarray
    :param gt_boxes: Ground truth bounding boxes of shape [#gt_box, 4]
    :type gt_boxes: numpy.ndarray
    :param gt_classes: Ground truth object classes of shape [#gt_box]
    :type gt_classes: numpy.ndarray
    :param pred_rels: Predicted relations array of shape [#pred_rel, 3] (id0, id1, rel)
    :type pred_rels: numpy.ndarray
    :param pred_boxes: Predicted bounding boxes of shape [#pred_box, 4]
    :type pred_boxes: numpy.ndarray
    :param pred_classes: Predicted object classes of shape [#pred_box]
    :type pred_classes: numpy.ndarray
    :param rel_scores: Relation scores, defaults to None
    :type rel_scores: numpy.ndarray, optional
    :param cls_scores: Classification scores, defaults to None
    :type cls_scores: numpy.ndarray, optional
    :param iou_thresh: IoU threshold for matching, defaults to 0.5
    :type iou_thresh: float, optional
    :param phrdet: Whether to use phrase detection mode, defaults to False
    :type phrdet: bool, optional
    :return: Tuple containing predicate-to-GT matching, predicted 5-tuples, and relation scores
    :rtype: tuple
    """
    if pred_rels.size == 0:
        return [[]], np.zeros((0, 5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(
        gt_rels[:, 2], gt_rels[:, :2], gt_classes, gt_boxes
    )
    num_boxes = pred_boxes.shape[0]
    assert pred_rels[:, :2].max() < pred_classes.shape[0]

    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,ĺeftright])
    # assert np.all(pred_rels[:,2] > 0)

    pred_triplets, pred_triplet_boxes, relation_scores = _triplet(
        pred_rels[:, 2],
        pred_rels[:, :2],
        pred_classes,
        pred_boxes,
        rel_scores,
        cls_scores,
    )

    sorted_scores = relation_scores.prod(1)
    pred_triplets = pred_triplets[sorted_scores.argsort()[::-1], :]
    pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1], :]
    relation_scores = relation_scores[sorted_scores.argsort()[::-1], :]
    scores_overall = relation_scores.prod(1)

    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print(
            "Somehow the relations weren't sorted properly: \n{}".format(scores_overall)
        )
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack(
        (
            pred_rels[:, :2],
            pred_triplets[:, [0, 2, 1]],
        )
    )

    return pred_to_gt, pred_5ples, relation_scores


def _triplet(
    predicates, relations, classes, boxes, predicate_scores=None, class_scores=None
):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert predicates.shape[0] == relations.shape[0]

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack(
            (
                class_scores[relations[:, 0]],
                class_scores[relations[:, 1]],
                predicate_scores,
            )
        )

    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(
    gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thresh, phrdet=False
):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(
        np.where(gt_has_match)[0],
        gt_boxes[gt_has_match],
        keeps[gt_has_match],
    ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate(
                (gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0
            )

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate(
                (box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1
            )

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
