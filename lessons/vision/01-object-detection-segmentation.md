# CV-01: Object Detection & Segmentation

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Computer Vision | ~9 h | L2→L3 | DL-04 |

## Why this matters
Most real vision products don't just classify an image. They find *where* things are: detection for counting, tracking, and
robotics, and segmentation for medical imaging, satellite data, and autonomous driving. The ideas here are anchors, IoU, NMS,
multi-scale features, and encoder-decoders, and they come up everywhere in vision.

## Learning goals
By the end you can:
- Explain bounding-box regression, anchors, IoU, non-maximum suppression, and mAP.
- Compare two-stage (Faster R-CNN) and one-stage (YOLO/SSD/RetinaNet) detectors, and say why focal loss helps.
- Explain semantic vs instance segmentation, FCNs, U-Net, and Mask R-CNN.
- Fine-tune a pretrained detector on a custom dataset, and evaluate it properly.
- Know where transformer detectors (DETR) and promptable segmentation (SAM) fit in.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Object Detection & Segmentation](../../notes/vision/01-object-detection-segmentation.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [Lilian Weng: Object Detection for Dummies](https://lilianweng.github.io/posts/2017-10-29-object-recognition-part-1/) | Parts [1](https://lilianweng.github.io/posts/2017-10-29-object-recognition-part-1/)–[2](https://lilianweng.github.io/posts/2017-12-15-object-recognition-part-2/) for the building blocks, [part 3](https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3/) (R-CNN family), and [part 4](https://lilianweng.github.io/posts/2018-12-27-object-recognition-part-4/) (fast detectors: YOLO, SSD, RetinaNet) | 2 h |
| 2 | **Read + Build** | [D2L](https://d2l.ai/) | Chapter "Computer Vision": the sections on bounding boxes, anchor boxes, multiscale detection, SSD, R-CNNs, and semantic segmentation/FCN. Run the code. | 2.5 h |
| 3 | **Read** | [U-Net](https://arxiv.org/abs/1505.04597) | The whole paper (short, with a clear figure) | 30 min |
| 4 | **Build** | [Ultralytics](https://github.com/ultralytics/ultralytics) or [Detectron2](https://github.com/facebookresearch/detectron2) | Fine-tune a pretrained detector on a small custom dataset. Report mAP@0.5 and mAP@0.5:0.95. | 2.5 h |
| 5 | *Read (optional)* | [HF Community CV Course](https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome) | The object detection and segmentation units (including DETR and SAM) | 1 h |

## Check your understanding
1. How is IoU computed, and why is mAP@0.5:0.95 harder than mAP@0.5?
2. What problem does NMS solve, and when does it hurt (e.g. crowded scenes)?
3. Why do one-stage detectors suffer from class imbalance, and how does focal loss fix it?
4. What do Feature Pyramid Networks add, and for which object sizes?
5. Why does U-Net need skip connections for segmentation?
6. *(debug)* Your detector finds large objects fine but misses almost all small ones. List three things to change.

## Mini-project
**Task:** Label about 200 images of one or two object classes you care about. Fine-tune a detector, report mAP, and show 10 failure
cases with a hypothesis for each. Then train a small U-Net on a public segmentation dataset and report IoU/Dice.
**Deliverable:** A notebook, a labeled-data README, and an error-analysis grid.

## Go deeper
- Papers: [Faster R-CNN](https://arxiv.org/abs/1506.01497) · [YOLO](https://arxiv.org/abs/1506.02640) · [FPN](https://arxiv.org/abs/1612.03144) · [Focal loss](https://arxiv.org/abs/1708.02002) · [Mask R-CNN](https://arxiv.org/abs/1703.06870)
- [DETR](https://arxiv.org/abs/2005.12872) and [Segment Anything](https://arxiv.org/abs/2304.02643): the transformer era of detection and segmentation.
- [Szeliski, *Computer Vision: Algorithms and Applications*](https://szeliski.org/Book/): the "Recognition" chapter, for the classical background.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 06: Computer vision](../../toolbox/06-computer-vision.md).
- **Papers:** [Computer vision](../../papers/02-computer-vision.md) (detection & segmentation). Start with the ⭐ ones.
- **Implement it yourself:** IoU, NMS, and mAP from scratch in NumPy. Check them against `torchvision.ops`.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) computer-vision problems · more in [exercises/](../../exercises/README.md).
- **Lab:** [Lab 13: IoU, NMS & average precision](../../labs/13-detection-metrics/README.md) (stubs + `pytest`).
