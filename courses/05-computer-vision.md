# Course 5: Computer Vision, a 5-week syllabus

[← Courses](README.md) · Path: [Path 5: Computer Vision](../paths/05-computer-vision.md) · Labs: [index](../labs/README.md) · Cards: [vision deck](flashcards/README.md)

> **Pace** ≈ 8 h/week for 5 weeks. **Before you start:** DL-04 and DL-06.
> **Running project:** the [path capstone](../paths/05-computer-vision.md#capstone), option (a), (b) or (c), chosen in week 1.

The [weekly loop](README.md#32-the-weekly-loop-about-8-hours) applies every week.

## How this course is shaped

- **Pretrained first** (fast.ai). Every week begins by running a pretrained model on *your* images, and only then studies how it was trained.
- **The metric is the curriculum** (CS231n). Detection and segmentation are hard mostly because their *evaluation* is subtle. Lab 13 builds IoU, NMS and AP by hand so that mAP stops being a black box.
- **Label efficiency as the throughline.** Week 2's linear probes, week 3's zero-shot CLIP, and the capstone all ask the same question: *how many labels do you really need?*

---

## Week 1: Whole game + detection & segmentation
- **Whole game (1–2 h):** run a pretrained detector on 20 of your own photos. Count the obvious misses by eye.
- **Do:** [CV-01](../lessons/vision/01-object-detection-segmentation.md) with its [notes](../notes/vision/01-object-detection-segmentation.md).
- **Lab:** [Lab 13: IoU, NMS & average precision](../labs/13-detection-metrics/README.md).
- **Project:** choose your capstone option, and collect or label the first 100 images.

## Week 2: Vision transformers & self-supervised learning
- **Warm-up:** CV-01 + DL-04 cards. *Interleave:* "mAP@0.5 is high, but mAP@[.5:.95] is low. What does your detector do badly?"
- **Do:** [CV-02](../lessons/vision/02-vision-transformers-self-supervised.md).
- **Playbook:** [frozen embeddings + a linear probe](../playbook/05-outside-the-box.md#8-frozen-embeddings--a-linear-model-is-a-strong-baseline-for-anything).
- **Project:** a linear-probe baseline on frozen features (DINOv2 or CLIP) for your task.

## Week 3: CLIP & vision-language models
- **Warm-up:** CV-02 + DL-06 cards. *Interleave:* "Why does a ViT need more data than a CNN to train from scratch, but not to fine-tune?"
- **Do:** [CV-03](../lessons/vision/03-clip-vision-language-models.md).
- **Project:** zero-shot CLIP vs your linear probe. Where does each win?

## Week 4: 3D vision & neural rendering
- **Warm-up:** CV-03 + CV-01 cards. *Interleave:* "CLIP's contrastive loss is softmax cross-entropy over what?"
- **Do:** [CV-04](../lessons/vision/04-3d-vision-neural-rendering.md).
- **Project:** error analysis and a model card for the capstone.

## Week 5: **Capstone**
- **Interleaved quiz:** 20 cards from CV-01…04 and DL-04.
- **Capstone (rubric below).**

---

## Capstone rubric

| Criterion | What "2" looks like |
|---|---|
| Data | A labeling guide, label-quality spot checks, and a split with no near-duplicates across it |
| Baselines | A pretrained/zero-shot baseline and a linear probe, before any fine-tuning |
| Evaluation | The right metric (mAP@[.5:.95], recall@k, or a label-efficiency curve), with an uncertainty estimate |
| Error analysis | A grid of failures, grouped, each with a hypothesis |
| Responsible use | A model card with dataset-bias and failure-mode sections |

Pass: ≥ 7/10.
