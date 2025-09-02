import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """Hungarian algorithm-based matcher for object detection.

    Computes an assignment between targets and network predictions using the Hungarian algorithm.
    For efficiency, targets don't include no-object class. When there are more predictions than
    targets, performs 1-to-1 matching of best predictions while treating others as no-object.

    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_iou: float = 1,
    ):
        """Initialize the Hungarian matcher.

        :param cost_class: Relative weight of classification error in matching cost, defaults to 1
        :type cost_class: float, optional
        :param cost_bbox: Relative weight of L1 error of bounding box coordinates, defaults to 1
        :type cost_bbox: float, optional
        :param cost_giou: Relative weight of GIoU loss of bounding box, defaults to 1
        :type cost_giou: float, optional
        :param cost_iou: Relative weight of IoU loss of bounding box, defaults to 1
        :type cost_iou: float, optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_iou = cost_iou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Perform the matching between predictions and targets.

        :param outputs: Dictionary containing model predictions
        :type outputs: dict
        :param targets: List of targets (ground truth)
        :type targets: list
        :return: List of tuples (index_i, index_j) where index_i is indices of selected predictions and index_j is indices of corresponding selected targets
        :rtype: list
        """
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = torch.zeros_like(cost_bbox)  # Placeholder for giou cost

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def build_matcher(args=None):
    return HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1, cost_iou=0.5)
