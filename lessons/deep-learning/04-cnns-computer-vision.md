# DL-04: Convolutional Networks & Computer Vision

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Deep Learning | ~8 h | L2 | DL-03 |

## Why this matters
CNNs introduced the key idea of **inductive bias**: building assumptions (locality, translation equivariance) into the
architecture. And transfer learning from pretrained vision models is still the fastest way to solve a real
image problem with little data.

## Learning goals
By the end you can:
- Explain convolution, padding, stride, pooling, and receptive fields, and compute output shapes by hand.
- Explain why CNNs need far fewer parameters than MLPs on images.
- Describe the architecture lineage: LeNet → AlexNet → VGG → ResNet (and where ViTs fit in).
- Fine-tune a pretrained model on a small custom dataset, using augmentation.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Intuition** | [CNN Explainer](https://poloclub.github.io/cnn-explainer/) | Click through every layer. Watch the animated convolutions. | 30 min |
| 2 | **Read** | [UDL](https://udlbook.github.io/udlbook/) (Prince) | Ch. 10 "Convolutional Networks" and Ch. 11 "Residual Networks", with the notebooks | 2.5 h |
| 3 | **Read** | [CS231n notes](https://cs231n.github.io/) | "Convolutional Neural Networks: Architectures, Convolution / Pooling Layers" and "Transfer Learning" | 1.5 h |
| 4 | **Build** | [learnpytorch.io](https://www.learnpytorch.io/) | 03 Computer Vision and 06 Transfer Learning | 2.5 h |
| 5 | *Watch (optional)* | [fast.ai Practical Deep Learning](https://course.fast.ai/) | Lesson 1 (a quick, motivating demo of what fine-tuning can do in a few lines) | 1.5 h |

## Check your understanding
1. A 3×3 conv with 64 input and 128 output channels: how many parameters does it have? How many would a dense layer on a 32×32×64 input need?
2. Why do stacked 3×3 convs beat one 7×7 conv?
3. What problem do residual connections solve, and why do they make very deep networks trainable?
4. When fine-tuning, why freeze early layers first, and why use a smaller learning rate for pretrained weights?
5. *(debug)* Your fine-tuned model scores 98% on validation but fails on photos from users' phones. What's likely going on, and how do you test your guess?

## Mini-project
**Task:** Build a classifier for 3–5 classes of your own choosing, with fewer than 200 images per class. Compare training
from scratch against fine-tuning a pretrained ResNet or EfficientNet (from `torchvision.models`). Show a confusion matrix
and 10 misclassified examples.
**Dataset:** Your own images, or [Food-101 subsets as used in learnpytorch.io](https://www.learnpytorch.io/04_pytorch_custom_datasets/).
**Deliverable:** A notebook plus a short error-analysis write-up.

## Go deeper
- [D2L](https://d2l.ai/): chapters "Convolutional Neural Networks" and "Modern Convolutional Neural Networks"
- [fastbook](https://github.com/fastai/fastbook): Ch. 13 "Convolutions", Ch. 14 "ResNets", Ch. 18 "CAM" (class activation maps for interpretability)
- [Géron, *Hands-On ML*, Ch. 12 "Deep Computer Vision Using CNNs" and Ch. 16 "Vision and Multimodal Transformers"](https://github.com/ageron/handson-mlp)
