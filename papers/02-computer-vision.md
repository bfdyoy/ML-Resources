# Papers: Computer Vision

[← Papers library](README.md) · Background: [DL-04 CNNs & Vision](../lessons/deep-learning/04-cnns-computer-vision.md) · [Toolbox: Computer vision](../toolbox/06-computer-vision.md)

## Classification backbones
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Very Deep Convolutional Networks (VGG)](https://arxiv.org/abs/1409.1556) | 2014 | Simple, deep stacks of 3×3 convs. Easy to read. | L1 | DL-04 |
| [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385) | 2015 | ⭐ Residual connections. | L2 | DL-04 |
| [MobileNets](https://arxiv.org/abs/1704.04861) | 2017 | Depthwise-separable convs for efficient models. | L2 | DL-04 |
| [EfficientNet](https://arxiv.org/abs/1905.11946) | 2019 | Compound scaling of depth, width, and resolution. | L2 | DL-04 |
| [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) | 2020 | ⭐ Transformers applied to image patches. | L2 | DL-06 |
| [Swin Transformer](https://arxiv.org/abs/2103.14030) | 2021 | Hierarchical ViT with shifted windows. | L3 | DL-06 |
| [A ConvNet for the 2020s (ConvNeXt)](https://arxiv.org/abs/2201.03545) | 2022 | Modernizing ResNet step by step to match ViTs. A great ablation story. | L2 | DL-04 |

## Detection & segmentation
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Faster R-CNN](https://arxiv.org/abs/1506.01497) | 2015 | ⭐ Two-stage detection, region proposal networks, anchors. | L2 | DL-04 |
| [You Only Look Once (YOLO)](https://arxiv.org/abs/1506.02640) | 2015 | ⭐ Single-stage, real-time detection. | L2 | DL-04 |
| [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) | 2016 | Multi-scale features for detection. | L2 | DL-04 |
| [Focal Loss (RetinaNet)](https://arxiv.org/abs/1708.02002) | 2017 | Handling extreme class imbalance in dense detection. | L2 | DL-04 |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870) | 2017 | Instance segmentation. | L2 | DL-04 |
| [U-Net](https://arxiv.org/abs/1505.04597) | 2015 | ⭐ The encoder-decoder with skips used in segmentation, medical imaging, and diffusion models. | L1 | DL-04 |
| [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | 2020 | Detection as set prediction, with no anchors or NMS. | L3 | DL-06 |
| [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643) | 2023 | A promptable foundation model for segmentation. | L2 | DL-06 |

## Self-supervised & representation learning
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [A Simple Framework for Contrastive Learning (SimCLR)](https://arxiv.org/abs/2002.05709) | 2020 | ⭐ Contrastive learning, clearly explained. | L2 | DL-04 |
| [Momentum Contrast (MoCo)](https://arxiv.org/abs/1911.05722) | 2019 | Queue plus momentum encoder. | L2 | DL-04 |
| [Bootstrap Your Own Latent (BYOL)](https://arxiv.org/abs/2006.07733) | 2020 | Self-supervision without negative pairs. | L3 | DL-04 |
| [Emerging Properties in Self-Supervised ViTs (DINO)](https://arxiv.org/abs/2104.14294) | 2021 | Self-distillation. Its attention maps segment objects. | L3 | DL-06 |
| [Masked Autoencoders Are Scalable Vision Learners (MAE)](https://arxiv.org/abs/2111.06377) | 2021 | ⭐ BERT-style masking for images. | L2 | DL-06 |
| [DINOv2](https://arxiv.org/abs/2304.07193) | 2023 | General-purpose visual features at scale. | L3 | DL-06 |
| [A Cookbook of Self-Supervised Learning](https://arxiv.org/abs/2304.12210) | 2023 | ⭐ A survey and practical recipe book. Read it after SimCLR. | L2 | DL-04 |

## Vision + language (multimodal)
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [CLIP](https://arxiv.org/abs/2103.00020) | 2021 | ⭐ Contrastive image–text pretraining and zero-shot classification. | L2 | DL-06 |
| [SigLIP: Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) | 2023 | A simpler loss than CLIP's. Used in many modern vision-language models (VLMs). | L3 | DL-06 |
| [Flamingo](https://arxiv.org/abs/2204.14198) | 2022 | Few-shot VLM using cross-attention into a frozen LM. | L3 | GEN-01 |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | 2023 | Connects a frozen image encoder to a frozen LLM through a Q-Former. | L3 | GEN-01 |
| [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) | 2023 | ⭐ A simple, influential open recipe for VLMs. | L2 | GEN-02 |

## 3D & neural rendering
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [NeRF](https://arxiv.org/abs/2003.08934) | 2020 | Scenes represented as neural radiance fields. | L3 | DL-04 |
| [3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) | 2023 | Real-time radiance-field rendering with explicit Gaussians. | L3 | DL-04 |
