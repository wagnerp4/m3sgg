import numpy as np


def bbox_overlaps(boxes1, boxes2):
    # boxes1: [N, 4], boxes2: [K, 4]
    N = boxes1.shape[0]
    K = boxes2.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        box1 = boxes1[n]
        for k in range(K):
            box2 = boxes2[k]
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            iw = max(x2 - x1 + 1, 0)
            ih = max(y2 - y1 + 1, 0)
            inter = iw * ih
            area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
            area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
            union = area1 + area2 - inter
            overlaps[n, k] = inter / union if union > 0 else 0
    return overlaps


def bbox_intersections(boxes1, boxes2):
    N = boxes1.shape[0]
    K = boxes2.shape[0]
    intersections = np.zeros((N, K), dtype=np.float32)
    for n in range(N):
        box1 = boxes1[n]
        for k in range(K):
            box2 = boxes2[k]
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            iw = max(x2 - x1 + 1, 0)
            ih = max(y2 - y1 + 1, 0)
            intersections[n, k] = iw * ih
    return intersections
