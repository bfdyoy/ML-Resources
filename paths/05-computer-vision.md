# Path 5: Computer Vision

**Goal:** Go beyond image classification. Detect and segment objects, use modern self-supervised and transformer backbones, build multimodal (image + text) systems, and understand 3D vision.
**Duration:** ~34 h (≈ 4–5 weeks) · **Level:** L2→L3 · **Primary resources:** D2L and Lilian Weng's detection series, UvA DL tutorials, CLIP/VLM papers with HF explainers, Szeliski

> 📘 **Study notes:** every lesson starts with its written explanation (step 0, ~1 h): the math step by step, worked examples, runnable code, and answers to the self-check. Read in order, the notes form one continuous walkthrough: [notes index](../notes/README.md).
>
> 🗓️ **As a course:** [Course 5: Computer Vision](../courses/05-computer-vision.md) turns this path into a week-by-week plan: warm-ups, [labs](../labs/README.md), [playbook](../playbook/README.md) readings, project milestones, a midterm, and a capstone rubric.

**Prerequisites:** [Path 2](02-deep-learning.md), at least DL-03, DL-04, DL-06. CV-03 also uses [GEN-01](../lessons/llms-genai/01-how-llms-are-built.md).

| # | Lesson | What you'll be able to do | Time |
|---|---|---|---|
| 1 | [CV-01 Object Detection & Segmentation](../lessons/vision/01-object-detection-segmentation.md) | Train and evaluate detectors and segmenters on custom data | 9 h |
| 2 | [CV-02 Vision Transformers & Self-Supervised Learning](../lessons/vision/02-vision-transformers-self-supervised.md) | Pretrain without labels, and evaluate representations | 9 h |
| 3 | [CV-03 Multimodal: CLIP & Vision-Language Models](../lessons/vision/03-clip-vision-language-models.md) | Build image–text search, and use VLMs | 8 h |
| 4 | [CV-04 3D Vision & Neural Rendering](../lessons/vision/04-3d-vision-neural-rendering.md) | Camera geometry, NeRF, and Gaussian splatting | 8 h |

## Capstone
**A visual product with a real evaluation.** Pick one: (a) a custom detector for a real-world counting/inspection task, with labeled data and an error
analysis; (b) a multimodal search engine over your own photos, with CLIP/SigLIP and measured recall@k; (c) a label-efficient classifier using
self-supervised features, with a label-efficiency curve. Include a model card, and a section on failure modes and dataset bias.

**Next:** [Path 3 Part C](03-llms-genai.md) (generative media) or [electives](06-electives.md).
