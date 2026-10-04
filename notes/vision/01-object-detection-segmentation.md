# CV-01 notes: Object Detection & Segmentation

[← Lesson CV-01](../../lessons/vision/01-object-detection-segmentation.md) · [All notes](../README.md) · [← DL-09 notes](../deep-learning/09-graph-neural-networks.md) · Next: [CV-02 notes →](02-vision-transformers-self-supervised.md)

> **Reading time** ≈ 60 min. **You need:** [DL-04 notes](../deep-learning/04-cnns-computer-vision.md) (convolutions, receptive fields, ResNets, fine-tuning) and [CORE-03 notes](../core-ml/03-classification-and-metrics.md) §3–4 (precision, recall, PR curves).

---

## Where we are

DL-04's classifier answers "*what* is in this image?". Detection answers "**what, and where**" (a box per object), and segmentation answers "**which pixels**". The new ingredients are:

- a way to measure overlap (IoU);
- a way to regress boxes;
- a way to remove duplicate detections (NMS);
- an evaluation metric that combines all of the above (mAP);
- fixes for two hard problems: extreme class imbalance and small objects.

---

## 1. Boxes and IoU

A box is $(x_1, y_1, x_2, y_2)$, or a centre with a width and height. **Intersection over union** measures how well two boxes overlap:

```math
\mathrm{IoU}(A, B) = \frac{\lvert A\cap B\rvert}{\lvert A\cup B\rvert} = \frac{\lvert A\cap B\rvert}{\lvert A\rvert + \lvert B\rvert - \lvert A\cap B\rvert}.
```

*Worked example:* $A = (0,0,4,4)$ and $B = (2,2,6,6)$. The intersection is the 2×2 square from (2,2) to (4,4), area 4. The union is $16 + 16 - 4 = 28$. So IoU = $4/28 \approx 0.143$.

A detection counts as a **true positive** if its IoU with an unmatched ground-truth box of the same class is at least a threshold (0.5 is classic).

## 2. How detectors predict boxes

### 2.1 Anchors and box regression

Many detectors place **anchor boxes** (several sizes and aspect ratios) at every position of a feature map. For each anchor, they predict (a) class scores and (b) **offsets** from the anchor to the object, in a scale-invariant parameterization:

```math
t_x = \frac{x - x_a}{w_a},\quad t_y = \frac{y - y_a}{h_a},\quad t_w = \log\frac{w}{w_a},\quad t_h = \log\frac{h}{h_a}.
```

Dividing by the anchor size makes a 5-pixel error mean the same thing for small and large boxes, and the log makes the width and height regression symmetric for growing and shrinking. The losses are classification cross-entropy (or focal loss, §3) plus a smooth-L1 or IoU-based box loss on the positive anchors.

### 2.2 Two-stage vs one-stage

- **Two-stage (Faster R-CNN):** a Region Proposal Network suggests about 1,000 likely boxes, then a second head classifies and refines each one, using features cropped for that region (RoIAlign). It's accurate, and slower.
- **One-stage (SSD, RetinaNet, YOLO):** it predicts classes and boxes densely, in a single pass. It's fast, and it faces a far worse class imbalance: about 100k anchors per image, only a handful containing objects.
- **Anchor-free** variants (FCOS, recent YOLOs) predict distances from each location to the box edges instead.
- **DETR** (a transformer) predicts a *set* of boxes directly with object queries, uses bipartite (Hungarian) matching to ground truth during training, and needs **no anchors and no NMS**.

### 2.3 Non-maximum suppression (NMS)

Dense prediction fires many overlapping boxes on the same object. The NMS algorithm:

1. Sort the boxes by score.
2. Keep the top box, and **remove every remaining box whose IoU with it exceeds a threshold** (say 0.5).
3. Repeat with the next remaining box.

**When it hurts:** in **crowded scenes** (people in a crowd, cars in a parking lot), true neighbouring objects overlap heavily, so NMS deletes real detections. Remedies: Soft-NMS (decay scores instead of deleting), a higher IoU threshold, or set-based detectors like DETR.

---

## 3. Focal loss: taming extreme imbalance

With about 100k easy background anchors per image, plain cross-entropy is dominated by them. Each easy negative contributes a little loss, but they're so numerous that together they **swamp** the handful of hard, informative examples. **Focal loss** down-weights the easy ones:

```math
\mathrm{FL}(p_t) = -\alpha_t\,(1 - p_t)^{\gamma}\,\log p_t,
```

where $p_t$ is the predicted probability of the *true* class. For an easy example ($p_t = 0.99$) with $\gamma = 2$, the factor is $(0.01)^2 = 10^{-4}$, so its loss shrinks 10,000×. A hard example ($p_t = 0.3$) keeps about half its loss.
The training signal concentrates on the hard cases, and that let one-stage RetinaNet match two-stage accuracy.

## 4. Multi-scale features: FPN

Deep CNN layers carry strong semantics at low resolution, while shallow layers have high resolution with weak semantics. Small objects may cover just a few cells of the deepest feature map. A **Feature Pyramid Network** builds a **top-down pathway**: upsample the deep features and add them (lateral connections) to the shallower ones.
The result is semantically strong features at *every* scale. Small objects are detected on the high-resolution levels, and large objects on the coarse ones. It helps **small objects most**.

---

## 5. Evaluating detectors: AP and mAP

For one class:

1. Sort all the detections by confidence.
2. Mark each one TP or FP (IoU ≥ threshold, at most one match per ground-truth box).
3. Walk down the list, computing precision and recall at each cut-off.
4. **AP** is the area under the precision–recall curve, after making precision monotone (each point takes the max precision at any higher recall), typically sampled at 101 recall points (the COCO convention).

**mAP** averages AP over the classes. **COCO's mAP@0.5:0.95** also averages over IoU thresholds 0.50, 0.55, …, 0.95. At 0.9 IoU, a box must be almost pixel-perfect, so this metric rewards **precise localization**, not just finding the object. It's much harder than mAP@0.5. COCO also reports AP for small, medium, and large objects.

---

## 6. Segmentation

- **Semantic segmentation:** a class for every pixel (all the cars get one "car" label). **Instance segmentation:** a separate mask per object (car #1, car #2). **Panoptic:** both, covering "stuff" (sky, road) as well as "things".
- **FCN:** replace the dense layers with convolutions so the network outputs a coarse class map, then upsample.
- **U-Net:** an encoder (downsampling, gathering context) plus a decoder (upsampling), with **skip connections** that concatenate encoder features at each resolution into the decoder.
  **Why the skips are essential:** downsampling throws away precise spatial detail (exact edges). The decoder can recover *what* from the deep features, but not exactly *where* the boundaries are. Skip connections hand it the high-resolution detail directly.
- **Mask R-CNN:** Faster R-CNN plus a small mask head per detected box (with RoIAlign's precise sampling).
- **SAM (Segment Anything):** a promptable segmenter (points, boxes, text) trained on about 1B masks. It's often the first tool to try now for labeling and zero-shot masks.
- **Metrics:** mean IoU per class (semantic); mask AP (instance).

**Debug: large objects are found, small ones are missed.** Change:

1. **input resolution:** train and infer at higher resolution, or use tiling or slicing (SAHI-style) for huge images;
2. **multi-scale features:** an FPN with a high-resolution level (P2), and smaller anchors (or an anchor-free design) matched to the small-object sizes;
3. **data:** augmentations that preserve or create small objects (copy-paste, mosaic, careful scale jitter), more small-object labels, and checking that small objects weren't dropped by a label filter or a "min box size" setting;
4. **evaluation:** report AP-small separately.

```python
import numpy as np
rng = np.random.default_rng(0)

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)
print("IoU worked example:", round(iou((0, 0, 4, 4), (2, 2, 6, 6)), 3))

def nms(boxes, scores, thr=0.5):
    order, keep = list(np.argsort(-scores)), []
    while order:
        i = order.pop(0); keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < thr]
    return [int(k) for k in keep]
boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52], [11, 9, 49, 51],      # 3 boxes on object A
                  [25, 10, 65, 50]])                                       # object B overlaps A (crowd)
scores = np.array([0.9, 0.8, 0.7, 0.85])
print("NMS keeps (thr 0.5):", nms(boxes, scores, 0.5), "| IoU(A,B) =", round(iou(boxes[0], boxes[3]), 2))
print("NMS keeps (thr 0.3):", nms(boxes, scores, 0.3), " <- too aggressive: object B (box 3) suppressed in a crowd")

# Focal loss vs cross-entropy for easy and hard examples
for pt in [0.99, 0.9, 0.5, 0.3]:
    ce = -np.log(pt); fl = -(1 - pt) ** 2 * np.log(pt)
    print(f"p_t={pt:4}: CE {ce:.4f}  focal(gamma=2) {fl:.6f}  ratio {fl/ce:.4f}")

# Share of the total loss coming from easy negatives: 100k easy (p_t=0.99) vs 50 hard (p_t=0.3)
easy, hard = 100_000, 50
for name, f in [("CE", lambda p: -np.log(p)), ("focal", lambda p: -(1 - p) ** 2 * np.log(p))]:
    share = easy * f(0.99) / (easy * f(0.99) + hard * f(0.3))
    print(f"{name:5s}: easy negatives make up {share:.1%} of the loss")
```

```python
# --- Average precision for one class (COCO-style 101-point interpolation) -----------------------------------
def average_precision(tp_flags, n_gt):
    tp = np.cumsum(tp_flags); fp = np.cumsum(1 - np.array(tp_flags))
    recall, precision = tp / n_gt, tp / (tp + fp)
    for i in range(len(precision) - 2, -1, -1):          # make precision monotone (envelope)
        precision[i] = max(precision[i], precision[i + 1])
    ap = 0
    for r in np.linspace(0, 1, 101):
        p = precision[recall >= r]; ap += (p.max() if len(p) else 0) / 101
    return ap
# detections sorted by confidence: 1 = matched a ground-truth box (TP), 0 = FP; 6 GT objects exist
print("AP =", round(average_precision([1, 1, 0, 1, 0, 1, 0, 0, 1], n_gt=6), 3))
print("same detections, but one box localized too loosely for a stricter IoU threshold:",
      round(average_precision([1, 0, 0, 1, 0, 1, 0, 0, 1], n_gt=6), 3))
```

---

## Pitfalls & misconceptions

- **Comparing mAP@0.5 to mAP@0.5:0.95.** They're different metrics.
- **The same NMS threshold for every dataset.** Crowded scenes need special care.
- **Training at low resolution and expecting small-object detection.**
- **Ignoring annotation quality.** Box precision limits the achievable high-IoU AP.
- **Augmentations that break box or mask alignment.** Transform labels together with images.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| IoU | $\lvert A\cap B\rvert / \lvert A\cup B\rvert$ |
| Box targets | $t_x = (x-x_a)/w_a$, $t_w = \log(w/w_a)$ |
| Focal loss | $-\alpha(1-p_t)^\gamma\log p_t$ |
| NMS | keep the max, drop IoU > threshold, repeat |
| AP | area under the monotone PR curve (101-point) |
| COCO mAP | mean over classes and IoU ∈ {0.5, …, 0.95} |
| U-Net | encoder–decoder + skip connections for spatial detail |

## Answer sketches for the lesson's self-check

<details>
<summary>1. How is IoU computed, and why is mAP@0.5:0.95 harder?</summary>

Intersection area over union area. mAP@0.5:0.95 averages AP over strict IoU thresholds up to 0.95, which require near-pixel-perfect boxes, so it rewards precise localization, not just finding objects.
</details>

<details>
<summary>2. What does NMS solve, and when does it hurt?</summary>

It removes duplicate overlapping detections of the same object. It hurts in crowded scenes, where distinct objects overlap above the threshold and get suppressed (the demo shows object B lost at a low threshold). Use Soft-NMS, tuned thresholds, or NMS-free detectors.
</details>

<details>
<summary>3. Why do one-stage detectors suffer from imbalance, and how does focal loss fix it?</summary>

They score around 100k anchors, nearly all easy background, whose summed loss dominates (about 94% of the CE loss in the demo, vs 0.3% with focal loss). Focal loss multiplies by $(1-p_t)^\gamma$, shrinking easy examples' loss by orders of magnitude, so the gradient comes from the hard examples.
</details>

<details>
<summary>4. What do FPNs add, and for which object sizes?</summary>

A top-down pathway with lateral connections, giving semantically strong features at multiple resolutions. It helps small objects most, which need high-resolution features with deep semantics.
</details>

<details>
<summary>5. Why does U-Net need skip connections?</summary>

Downsampling loses precise spatial information. Skip connections pass high-resolution encoder features to the decoder, so the masks get accurate boundaries, while the deep path supplies the context.
</details>

<details>
<summary>6. Large objects found, small ones missed: three changes.</summary>

Higher input resolution or tiling. Multi-scale features (an FPN with a high-resolution level) and anchors or assignment suited to small boxes. Small-object-friendly data and augmentation (copy-paste, more labels, no min-size filtering). Track AP-small.
</details>

## Where this leads

Next: [CV-02 notes](02-vision-transformers-self-supervised.md). Detection and segmentation need lots of labeled boxes and masks, which are expensive. CV-02 covers the backbones that make them work with less labeling: vision transformers, and representations learned **without labels at all**.
