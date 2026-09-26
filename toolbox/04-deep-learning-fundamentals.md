# Toolbox 04: Deep Learning Fundamentals

[← Toolbox](README.md) · Guided version: [DL-01](../lessons/deep-learning/01-neural-networks-from-scratch.md), [DL-02](../lessons/deep-learning/02-pytorch-fluency.md), [DL-03](../lessons/deep-learning/03-training-deep-networks.md)

## Core mechanics
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Neurons, MLPs, universal approximation | [3Blue1Brown: Neural networks](https://www.3blue1brown.com/lessons/neural-networks/) · [Alammar: Visual NN basics](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/) | [UDL](https://udlbook.github.io/udlbook/) Ch. 3–4 · [Nielsen Ch. 4](http://neuralnetworksanddeeplearning.com/chap4.html) | UDL Ch. 3–4 notebooks | — |
| Backpropagation | [3B1B: Backprop calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus/) · [colah: Computational graphs](https://colah.github.io/posts/2015-08-Backprop/) | [Nielsen Ch. 2](http://neuralnetworksanddeeplearning.com/chap2.html) · [UDL](https://udlbook.github.io/udlbook/) Ch. 7 | [micrograd](https://github.com/karpathy/micrograd) · [Autodiff-Puzzles](https://github.com/srush/Autodiff-Puzzles) | — |
| Loss functions from maximum likelihood | — | [UDL](https://udlbook.github.io/udlbook/) Ch. 5 | Derive BCE and CE from Bernoulli and categorical likelihoods | — |
| Tensors & PyTorch basics | [PyTorch: Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) | [learnpytorch.io](https://www.learnpytorch.io/) 00–02 · [D2L](https://d2l.ai/) Builders' Guide | [Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles) (broadcasting mastery) | — |

## Optimization
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| SGD, momentum, Nesterov | [Why Momentum Really Works](https://distill.pub/2017/momentum/) | [UDL](https://udlbook.github.io/udlbook/) Ch. 6 · [D2L](https://d2l.ai/) Optimization Algorithms chapter | Implement each optimizer and compare on a toy loss | [Ruder overview](https://arxiv.org/abs/1609.04747) |
| Adam & AdamW | — | [UDL](https://udlbook.github.io/udlbook/) §6.4 | Reimplement `torch.optim.AdamW` and match its outputs | [Adam](https://arxiv.org/abs/1412.6980), [AdamW](https://arxiv.org/abs/1711.05101) |
| Learning-rate schedules, warmup, LR finder | — | [D2L](https://d2l.ai/) (LR scheduling section) · [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) | LR range test plus a cosine schedule on CIFAR-10 | [CLR](https://arxiv.org/abs/1506.01186), [SGDR](https://arxiv.org/abs/1608.03983), [1-cycle](https://arxiv.org/abs/1803.09820) |
| Loss landscapes & generalization | [MLU-Explain: Double Descent](https://mlu-explain.github.io/double-descent/) | [UDL](https://udlbook.github.io/udlbook/) Ch. 20 · [Lilian Weng: Are DNNs dramatically overfitted?](https://lilianweng.github.io/posts/2019-03-14-overfit/) | Reproduce grokking on modular addition | [Rethinking generalization](https://arxiv.org/abs/1611.03530), [Loss landscape](https://arxiv.org/abs/1712.09913) |

## Making deep nets train
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Initialization (Xavier, He) | — | [UDL](https://udlbook.github.io/udlbook/) Ch. 7 · [CS231n notes](https://cs231n.github.io/) (NN Part 2) | Plot activation stats per layer for different inits | [He init](https://arxiv.org/abs/1502.01852) |
| Activations (ReLU, GELU, SwiGLU) | — | [UDL](https://udlbook.github.io/udlbook/) Ch. 3 | Dead-ReLU experiment | [GELU](https://arxiv.org/abs/1606.08415), [SwiGLU](https://arxiv.org/abs/2002.05202) |
| Normalization (Batch/Layer/Group/RMS) | [Karpathy Lecture 4](https://youtu.be/P6sfmUTpUmc) | [UDL](https://udlbook.github.io/udlbook/) Ch. 11 (§ on BatchNorm) · [D2L](https://d2l.ai/) (batch normalization section) | Implement BatchNorm forward/backward by hand ([Karpathy Lecture 5](https://github.com/karpathy/nn-zero-to-hero)) | [BN](https://arxiv.org/abs/1502.03167), [LN](https://arxiv.org/abs/1607.06450), [RMSNorm](https://arxiv.org/abs/1910.07467) |
| Regularization: weight decay, dropout, augmentation, label smoothing, mixup, early stopping | — | [UDL](https://udlbook.github.io/udlbook/) Ch. 9 | Ablate each on a small CNN | [Dropout](https://arxiv.org/abs/1207.0580), [mixup](https://arxiv.org/abs/1710.09412) |
| The debugging recipe | [Karpathy: A Recipe for Training NNs](http://karpathy.github.io/2019/04/25/recipe/) | [CS231n notes](https://cs231n.github.io/) (NN Part 3: babysitting the learning process) · [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) | Overfit one batch; check the loss at init | — |
| Hyperparameter tuning strategy | — | [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) | Run a structured sweep and write down conclusions | [1-cycle](https://arxiv.org/abs/1803.09820) |

## Performance & precision
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Why GPUs are fast / slow (compute vs memory vs overhead) | [Horace He: Making DL Go Brrrr](https://horace.io/brrr_intro.html) | [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (rooflines chapter) | Profile a training step ([PyTorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)) | — |
| Mixed precision (FP16/BF16) | — | [PyTorch performance tuning guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | Enable `torch.autocast` and measure the speed-up | [Mixed precision](https://arxiv.org/abs/1710.03740) |
| `torch.compile` & kernel fusion | [Horace He: Making DL Go Brrrr](https://horace.io/brrr_intro.html) (operator fusion) | [PyTorch: torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | Compile a model and compare throughput | — |
| Gradient checkpointing & memory | — | [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (memory section) | Measure peak memory with and without checkpointing | [Sublinear memory](https://arxiv.org/abs/1604.06174) |
| Hardware choices | [Tim Dettmers: Which GPU for Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) | — | — | — |

## All-in-one alternatives
- [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/): notebook tutorials (init, optimizers, activations, and more) with PyTorch and JAX versions. Excellent.
- [NYU Deep Learning (LeCun & Canziani)](https://github.com/Atcold/NYU-DLSP21): lectures and notebooks.
- [Deep Learning (Goodfellow et al.)](https://www.deeplearningbook.org/) Part II: the classic treatment of regularization and optimization.
- [The Little Book of Deep Learning](https://fleuret.org/francois/lbdl.html): the whole field in 170 small pages, for revision.
