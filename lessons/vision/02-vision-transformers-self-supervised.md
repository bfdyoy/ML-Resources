# CV-02: Vision Transformers & Self-Supervised Representation Learning

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Computer Vision | ~8 h | L3 | DL-04, DL-06 |

## Why this matters
Modern vision backbones are pretrained *without labels* (contrastive learning, masked image modeling, self-distillation), and often on
transformer architectures. Those pretrained features (DINO, MAE) are what you fine-tune or linear-probe in practice. The same ideas,
contrastive losses and masking, also run through NLP, audio, and multimodal models.

## Learning goals
By the end you can:
- Explain the ViT pipeline (patchify → embed → transformer → [CLS]) and why ViTs need lots of data or strong augmentation.
- Explain contrastive learning (SimCLR/MoCo): positives, negatives, the InfoNCE loss, and the role of augmentations.
- Explain non-contrastive self-supervision (BYOL/DINO) and masked image modeling (MAE).
- Evaluate representations with linear probes and k-NN evaluation.
- Choose between a supervised ImageNet backbone and a self-supervised one for a downstream task.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read** | [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) | §1–3 and Figure 1. Skim the experiments. | 1 h |
| 2 | **Build** | [UvA DL Tutorials](https://uvadlc-notebooks.readthedocs.io/) | The Vision Transformer tutorial notebook | 1.5 h |
| 3 | **Read** | [Lilian Weng: Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) | The loss functions section and the vision methods (SimCLR, MoCo, BYOL) | 1.5 h |
| 4 | **Build** | [UvA DL Tutorials](https://uvadlc-notebooks.readthedocs.io/) | The self-supervised contrastive learning (SimCLR) tutorial, with a linear probe | 2 h |
| 5 | **Read** | [A Cookbook of Self-Supervised Learning](https://arxiv.org/abs/2304.12210) | §2 (the method families) and §3 (training recipes and evaluation) | 1.5 h |
| 6 | *Read (optional)* | [MAE](https://arxiv.org/abs/2111.06377) · [DINO](https://arxiv.org/abs/2104.14294) | Abstracts, figures, and method sections | 45 min |

## Check your understanding
1. What inductive biases do CNNs have that ViTs lack, and what does that imply about data requirements?
2. In SimCLR, what are the positives and the negatives, and why does batch size matter?
3. Why don't BYOL/DINO collapse to a constant representation without negatives?
4. Why does MAE mask *75%* of the patches, while BERT masks only 15% of tokens?
5. What does a linear probe measure that fine-tuning doesn't?
6. *(debug)* Your SimCLR loss drops quickly but linear-probe accuracy stays near chance. Which augmentation mistake is the likely cause?

## Mini-project
**Task:** Pretrain SimCLR on unlabeled CIFAR-10 (or STL-10), then linear-probe with 1%, 10%, and 100% of the labels. Compare with a
supervised baseline and with frozen DINOv2 features (via `transformers`).
**Deliverable:** A label-efficiency plot (accuracy vs % of labels) for the three feature sources.

## Go deeper
- [SimCLR](https://arxiv.org/abs/2002.05709) · [MoCo](https://arxiv.org/abs/1911.05722) · [BYOL](https://arxiv.org/abs/2006.07733) · [DINOv2](https://arxiv.org/abs/2304.07193)
- [Lilian Weng: Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/): the pretext-task history.
- [Swin Transformer](https://arxiv.org/abs/2103.14030) and [ConvNeXt](https://arxiv.org/abs/2201.03545): the CNN–ViT convergence.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 06: Computer vision](../../toolbox/06-computer-vision.md) · [Toolbox 05: Architectures](../../toolbox/05-architectures.md) (autoencoders & representation learning).
- **Papers:** [Computer vision](../../papers/02-computer-vision.md) (self-supervised section).
- **Implement it yourself:** the InfoNCE/NT-Xent loss from scratch. Check it against a reference implementation on a fixed batch.
- **Drills:** more in [exercises/](../../exercises/README.md).
