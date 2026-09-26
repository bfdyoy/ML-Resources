# Path 2: Deep Learning Foundations

**Goal:** Understand neural networks deeply enough to build, train, and debug them from scratch, then use modern architectures (CNNs, transformers) confidently.
**Duration:** ~67 h (≈ 9 weeks) · **Level:** L2→L3 · **Primary book:** *Understanding Deep Learning* (Prince, free) + Karpathy's *Zero to Hero* for building

**Prerequisites:** CORE-02, CORE-03, CORE-04 (or equivalent), comfortable Python/NumPy.

| # | Lesson | What you'll be able to do | Time |
|---|---|---|---|
| 1 | [DL-01 Neural Networks from Scratch & Backprop](../lessons/deep-learning/01-neural-networks-from-scratch.md) | Derive and implement backprop; build a tiny autograd engine | 8 h |
| 2 | [DL-02 PyTorch Fluency](../lessons/deep-learning/02-pytorch-fluency.md) | Write clean PyTorch training code from memory | 7 h |
| 3 | [DL-03 Training Deep Networks Well](../lessons/deep-learning/03-training-deep-networks.md) | Initialize, normalize, optimize, regularize, and *debug* deep nets | 8 h |
| 4 | [DL-04 Convolutional Networks & Computer Vision](../lessons/deep-learning/04-cnns-computer-vision.md) | Build CNNs and fine-tune pretrained vision models | 8 h |
| 5 | [DL-05 Embeddings, Language Modeling & Sequences](../lessons/deep-learning/05-embeddings-sequences-attention.md) | Build character-level LMs; understand why attention was needed | 7 h |
| 6 | [DL-06 Transformers](../lessons/deep-learning/06-transformers.md) | Explain every part of a transformer and build a GPT from scratch | 9 h |
| 7 | [DL-07 Making Training Fast](../lessons/deep-learning/07-performance-gpus-mixed-precision.md) | Profile and speed up training (AMP, `torch.compile`, memory) | 6 h |
| 8 | [DL-08 Modern Architectures: RoPE, GQA, MoE, SSMs](../lessons/deep-learning/08-modern-architectures-moe-ssm.md) | Read and build 2026-era LLM architectures | 7 h |
| 9 | [DL-09 Graph Neural Networks](../lessons/deep-learning/09-graph-neural-networks.md) | Learn on graphs with message passing | 7 h |

**Math, just in time:** [MATH-02](../lessons/math/02-calculus-optimization.md) block A (before DL-01) ·
[MATH-03](../lessons/math/03-probability-statistics.md) block C (with DL-01) · [MATH-01](../lessons/math/01-linear-algebra.md) block B (before DL-06)

## Capstone
**Paper replication, small scale.** Pick one: (a) ResNet on CIFAR-10 reaching >90% accuracy with your own training loop and
an ablation of 3 tricks; (b) a ViT on a small image dataset compared with a CNN of similar size; or (c) your GPT trained on a
domain corpus, with a tokenizer, evaluation (val loss / perplexity), and a small ablation study. Write it up like a short paper.
(For guidance: [learnpytorch.io 08: Paper Replicating](https://www.learnpytorch.io/08_pytorch_paper_replicating/).)

**Next:** [Path 3: LLMs & Generative AI](03-llms-genai.md) and/or [Path 5: Computer Vision](05-computer-vision.md).
