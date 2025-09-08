import cv2
import numpy as np


def draw_union_boxes(pair_rois, spatial_scale=27):
    """Draw union boxes for pairs of ROIs to create spatial masks.

    Creates spatial masks for subject-object pairs by drawing their bounding
    boxes and union boxes on a grid. Used for spatial relationship modeling
    in scene graph generation.

    :param pair_rois: Array of ROI pairs, shape [N, 8] with format [x1_subj, y1_subj, x2_subj, y2_subj, x1_obj, y1_obj, x2_obj, y2_obj]
    :type pair_rois: numpy.ndarray
    :param spatial_scale: Scale for spatial masks, defaults to 27
    :type spatial_scale: int, optional
    :return: Spatial masks for each pair, shape [num_pairs, 2, spatial_scale, spatial_scale]
    :rtype: numpy.ndarray
    """
    num_pairs = pair_rois.shape[0]

    # Initialize the output array with correct shape: [num_pairs, 2, spatial_scale, spatial_scale]
    # Channel 0: subject mask, Channel 1: object mask
    spatial_masks = np.zeros(
        (num_pairs, 2, spatial_scale, spatial_scale), dtype=np.float32
    )

    for i in range(num_pairs):
        # Extract subject and object boxes
        subj_box = pair_rois[i, :4]  # [x1, y1, x2, y2]
        obj_box = pair_rois[i, 4:]  # [x1, y1, x2, y2]

        # Compute union box
        union_x1 = min(subj_box[0], obj_box[0])
        union_y1 = min(subj_box[1], obj_box[1])
        union_x2 = max(subj_box[2], obj_box[2])
        union_y2 = max(subj_box[3], obj_box[3])

        union_w = union_x2 - union_x1
        union_h = union_y2 - union_y1

        # Avoid division by zero
        if union_w <= 0 or union_h <= 0:
            continue

        # Normalize coordinates to the spatial scale
        subj_x1_norm = int((subj_box[0] - union_x1) / union_w * spatial_scale)
        subj_y1_norm = int((subj_box[1] - union_y1) / union_h * spatial_scale)
        subj_x2_norm = int((subj_box[2] - union_x1) / union_w * spatial_scale)
        subj_y2_norm = int((subj_box[3] - union_y1) / union_h * spatial_scale)

        obj_x1_norm = int((obj_box[0] - union_x1) / union_w * spatial_scale)
        obj_y1_norm = int((obj_box[1] - union_y1) / union_h * spatial_scale)
        obj_x2_norm = int((obj_box[2] - union_x1) / union_w * spatial_scale)
        obj_y2_norm = int((obj_box[3] - union_y1) / union_h * spatial_scale)

        # Clamp to valid range
        subj_x1_norm = max(0, min(subj_x1_norm, spatial_scale - 1))
        subj_y1_norm = max(0, min(subj_y1_norm, spatial_scale - 1))
        subj_x2_norm = max(0, min(subj_x2_norm, spatial_scale - 1))
        subj_y2_norm = max(0, min(subj_y2_norm, spatial_scale - 1))

        obj_x1_norm = max(0, min(obj_x1_norm, spatial_scale - 1))
        obj_y1_norm = max(0, min(obj_y1_norm, spatial_scale - 1))
        obj_x2_norm = max(0, min(obj_x2_norm, spatial_scale - 1))
        obj_y2_norm = max(0, min(obj_y2_norm, spatial_scale - 1))

        # Channel 0: Subject mask
        if subj_x2_norm > subj_x1_norm and subj_y2_norm > subj_y1_norm:
            spatial_masks[
                i, 0, subj_y1_norm : subj_y2_norm + 1, subj_x1_norm : subj_x2_norm + 1
            ] = 1

        # Channel 1: Object mask
        if obj_x2_norm > obj_x1_norm and obj_y2_norm > obj_y1_norm:
            spatial_masks[
                i, 1, obj_y1_norm : obj_y2_norm + 1, obj_x1_norm : obj_x2_norm + 1
            ] = 1

    return spatial_masks


# Alias for backward compatibility in case it's needed
def draw_union_boxes_cython(pair_rois, spatial_scale=27):
    """Alias for draw_union_boxes for compatibility."""
    return draw_union_boxes(pair_rois, spatial_scale)
