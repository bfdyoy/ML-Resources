# Lab 13: IoU, NMS & average precision

[← Labs](../README.md) · Lesson: [CV-01 Object detection & segmentation](../../lessons/vision/01-object-detection-segmentation.md) · Notes: [CV-01 notes](../../notes/vision/01-object-detection-segmentation.md)

**Time** ≈ 1.5 h · **You'll practise:** broadcasting over box pairs, greedy suppression, and the exact matching rules behind "mAP", the number every detection paper reports.

| Function | Checked against |
|---|---|
| `box_iou` | hand values (overlap, containment, disjoint) |
| `nms` | hand cases at two thresholds |
| `average_precision` | perfect predictions = 1; a hand-computed case with a duplicate detection; the IoU threshold flips a match |

```bash
pytest labs/13-detection-metrics
```

**Bonus:** COCO's mAP@[.5:.95] averages AP over IoU thresholds 0.50, 0.55, …, 0.95. Compute it for a detector whose boxes are all shifted
by 10% of their width. Which thresholds does it lose, and what does that tell you about the two numbers papers report?
