import numpy as np
import pytest


def test_box_iou(impl):
    a = np.array([[0, 0, 2, 2], [0, 0, 1, 1]], float)
    b = np.array([[1, 1, 3, 3], [0, 0, 2, 2], [5, 5, 6, 6]], float)
    iou = impl.box_iou(a, b)
    assert iou.shape == (2, 3)
    np.testing.assert_allclose(iou, [[1 / 7, 1, 0], [0, 0.25, 0]])


def test_nms(impl):
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [0, 0, 10, 9]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.95])
    assert impl.nms(boxes, scores, 0.5) == [3, 2]
    assert impl.nms(boxes, scores, 0.95) == [3, 0, 1, 2]
    assert impl.nms(boxes[:1], scores[:1], 0.5) == [0]


def test_ap_perfect_and_empty(impl):
    gt = [np.array([[0, 0, 10, 10]], float), np.array([[5, 5, 15, 15]], float)]
    preds = [(gt[0], np.array([0.9])), (gt[1], np.array([0.8]))]
    assert impl.average_precision(preds, gt) == 1.0
    assert impl.average_precision(preds, [np.zeros((0, 4)), np.zeros((0, 4))]) == 0.0


def test_ap_by_hand(impl):
    # One image, two ground truths. Ranked predictions: TP, FP, duplicate of the first (FP), TP.
    gt = [np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)]
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [0, 0, 10, 10.5], [20, 20, 30, 31]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    # precision 1, 1/2, 1/3, 1/2 ; recall .5, .5, .5, 1 ; interpolated precision 1, .5, .5, .5
    assert np.isclose(impl.average_precision([(boxes, scores)], gt), 0.5 * 1 + 0.5 * 0.5)


def test_ap_iou_threshold_matters(impl):
    gt = [np.array([[0, 0, 10, 10]], float)]
    loose = [(np.array([[2, 0, 12, 10]], float), np.array([0.9]))]       # IoU = 80/120 = 0.667
    assert impl.average_precision(loose, gt, 0.5) == 1.0
    assert impl.average_precision(loose, gt, 0.75) == 0.0
