import numpy as np
import cv2

def ex_box_jaccard(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    inter_x1 = np.maximum(np.min(a[:, 0]), np.min(b[:, 0]))
    inter_y1 = np.maximum(np.min(a[:, 1]), np.min(b[:, 1]))
    inter_x2 = np.minimum(np.max(a[:, 0]), np.max(b[:, 0]))
    inter_y2 = np.minimum(np.max(a[:, 1]), np.max(b[:, 1]))

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    
    x1 = np.minimum(np.min(a[:, 0]), np.min(b[:, 0]))
    y1 = np.minimum(np.min(a[:, 1]), np.min(b[:, 1]))
    x2 = np.maximum(np.max(a[:, 0]), np.max(b[:, 0]))
    y2 = np.maximum(np.max(a[:, 1]), np.max(b[:, 1]))

    mask_w = np.int(np.ceil(x2 - x1))
    mask_h = np.int(np.ceil(y2 - y1))
    mask_a = np.zeros((mask_h, mask_w), dtype=np.uint8)
    mask_b = np.zeros((mask_h, mask_w), dtype=np.uint8)

    a[:, 0] = a[:, 0] - x1
    a[:, 1] = a[:, 1] - y1
    b[:, 0] = b[:, 0] - x1
    b[:, 1] = b[:, 1] - y1

    mask_a = cv2.fillPoly(mask_a, [a.astype(np.int32)], 1)
    mask_b = cv2.fillPoly(mask_b, [b.astype(np.int32)], 1)

    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    iou = float(inter) / (float(union) + 1e-12)
    return iou