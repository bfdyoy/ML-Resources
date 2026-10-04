# EXERCISE: generated from solution.py by scripts/make_lab_stubs.py.
# Replace each `raise NotImplementedError` with your code, then run the tests (see README.md).
"""Lab 13: box IoU, non-maximum suppression, and average precision for one class.

Boxes are float arrays (N, 4) in (x1, y1, x2, y2) format with x2 > x1 and y2 > y1.
"""
import numpy as np


def box_area(boxes):
    """Areas, shape (N,)."""
    raise NotImplementedError("your code here")


def box_iou(a, b):
    """Pairwise IoU, shape (N, M). Subgoals: 1. intersection corners by broadcasting max/min
    2. clip widths and heights at 0  3. IoU = inter / (area_a + area_b - inter)."""
    raise NotImplementedError("your code here")


def nms(boxes, scores, iou_threshold):
    """Greedy NMS: repeatedly keep the highest-scoring remaining box and drop the boxes whose IoU with it is
    > iou_threshold. Return the kept indices in descending score order (ties: lower index first)."""
    raise NotImplementedError("your code here")


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
    raise NotImplementedError("your code here")
