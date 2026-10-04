"""Lab 13: box IoU, non-maximum suppression, and average precision for one class.

Boxes are float arrays (N, 4) in (x1, y1, x2, y2) format with x2 > x1 and y2 > y1.
"""
import numpy as np


def box_area(boxes):
    """Areas, shape (N,)."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(a, b):
    """Pairwise IoU, shape (N, M). Subgoals: 1. intersection corners by broadcasting max/min
    2. clip widths and heights at 0  3. IoU = inter / (area_a + area_b - inter)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (box_area(a)[:, None] + box_area(b)[None, :] - inter)


def nms(boxes, scores, iou_threshold):
    """Greedy NMS: repeatedly keep the highest-scoring remaining box and drop the boxes whose IoU with it is
    > iou_threshold. Return the kept indices in descending score order (ties: lower index first)."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return keep


def average_precision(preds, gts, iou_threshold=0.5):
    """AP for one class over a dataset.
    preds: list over images of (boxes (N_i, 4), scores (N_i,));  gts: list over images of boxes (G_i, 4).

    Subgoals:
      1. Pool all predictions, sorted by descending score (stable).
      2. Walk down the list. A prediction is a true positive if its best-IoU unmatched ground truth IN THE SAME IMAGE
         has IoU >= iou_threshold; that ground truth is then marked matched. Otherwise it is a false positive.
      3. precision_k = TP_k / k, recall_k = TP_k / (total ground truths).
      4. All-point interpolated AP (PASCAL VOC 2010+): make precision monotone non-increasing from the right,
         then sum (recall_k - recall_{k-1}) * precision_k.
    Return 0.0 if there are no ground truths."""
    n_gt = sum(len(g) for g in gts)
    if n_gt == 0:
        return 0.0
    entries = [(s, img, j) for img, (bx, sc) in enumerate(preds) for j, s in enumerate(sc)]
    entries.sort(key=lambda e: -e[0])
    matched = [np.zeros(len(g), bool) for g in gts]
    tp = np.zeros(len(entries))
    for k, (_, img, j) in enumerate(entries):
        g = gts[img]
        if len(g) == 0:
            continue
        ious = box_iou(preds[img][0][j:j + 1], g)[0]
        ious[matched[img]] = -1.0
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold:
            matched[img][best] = True
            tp[k] = 1
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(entries) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
