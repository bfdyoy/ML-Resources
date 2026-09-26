# Toolbox 05: Architectures

[← Toolbox](README.md) · Guided version: [DL-04](../lessons/deep-learning/04-cnns-computer-vision.md), [DL-05](../lessons/deep-learning/05-embeddings-sequences-attention.md), [DL-06](../lessons/deep-learning/06-transformers.md)

> **The unifying idea:** architectures encode *inductive biases*: locality (CNNs), order (RNNs), permutation symmetry (GNNs,
> attention), sparsity (MoE). For the deep version of this view, read [Geometric Deep Learning](https://arxiv.org/abs/2104.13478).

## Convolutional networks
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Convolution, padding, stride, pooling, receptive field | [CNN Explainer](https://poloclub.github.io/cnn-explainer/) | [UDL](https://udlbook.github.io/udlbook/) Ch. 10 · [CS231n: ConvNets](https://cs231n.github.io/) | Implement conv2d with loops, then with `unfold` | — |
| Residual networks | — | [UDL](https://udlbook.github.io/udlbook/) Ch. 11 · [D2L](https://d2l.ai/) Modern CNNs | ResNet-18 on CIFAR-10 from scratch | [ResNet](https://arxiv.org/abs/1512.03385) |
| Modern CNN design | — | [D2L](https://d2l.ai/) Modern CNNs chapter | Reproduce 2–3 ConvNeXt changes | [ConvNeXt](https://arxiv.org/abs/2201.03545), [EfficientNet](https://arxiv.org/abs/1905.11946) |

## Sequence models
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| RNNs | [Karpathy: The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) | [D2L](https://d2l.ai/) Recurrent Neural Networks | char-RNN in PyTorch | — |
| LSTM / GRU | [colah: Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | [D2L](https://d2l.ai/) Modern RNNs | Implement an LSTM cell and check it against `nn.LSTM` | — |
| Seq2seq + attention | [Alammar: Visualizing Neural Machine Translation](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) · [Distill: Attention & Augmented RNNs](https://distill.pub/2016/augmented-rnns/) | [Lilian Weng: Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) | A small translation model with Bahdanau attention | [Bahdanau](https://arxiv.org/abs/1409.0473) |

## Transformers
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Self-attention, multi-head attention | [3B1B: Attention](https://www.3blue1brown.com/lessons/attention/) · [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | [UDL](https://udlbook.github.io/udlbook/) Ch. 12 · [Raschka: Coding self-attention](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) | [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) · [Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| The full block (residual stream, LN, MLP) | [3B1B: Transformers](https://www.3blue1brown.com/lessons/gpt/) | [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) · [transformer circuits framework](https://transformer-circuits.pub/2021/framework/index.html) | [nanoGPT](https://github.com/karpathy/nanoGPT) · [microgpt](https://karpathy.github.io/2026/02/12/microgpt/) | [Pre-LN](https://arxiv.org/abs/2002.04745) |
| Positional encodings (sinusoidal, learned, RoPE, ALiBi) | [HF: Designing positional encoding](https://huggingface.co/blog/designing-positional-encoding) | [Lilian Weng: The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) | Implement RoPE and verify its relative-position property | [RoFormer](https://arxiv.org/abs/2104.09864), [YaRN](https://arxiv.org/abs/2309.00071) |
| Encoder vs decoder vs encoder-decoder | [Illustrated BERT](https://jalammar.github.io/illustrated-bert/) · [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) | [SLP3](https://web.stanford.edu/~jurafsky/slp3/) (transformer and masked-LM chapters) | Fine-tune BERT vs GPT-2 on the same classification task | [BERT](https://arxiv.org/abs/1810.04805), [T5](https://arxiv.org/abs/1910.10683) |
| Efficient attention variants (MQA/GQA, sliding window, linear) | — | [Lilian Weng: Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) · [Raschka: The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) | Swap MHA → GQA in nanoGPT and measure the KV-cache size | [GQA](https://arxiv.org/abs/2305.13245), [Efficient Transformers survey](https://arxiv.org/abs/2009.06732) |
| Mixture of Experts | [HF: Mixture of Experts Explained](https://huggingface.co/blog/moe) | [Raschka: LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) | Add a top-2 MoE MLP to nanoGPT | [Sparsely-gated MoE](https://arxiv.org/abs/1701.06538), [Switch](https://arxiv.org/abs/2101.03961), [Mixtral](https://arxiv.org/abs/2401.04088) |
| State-space models (S4, Mamba) | — | State-space-model appendix in [Géron's notebooks](https://github.com/ageron/handson-mlp) | Implement a minimal selective scan | [S4](https://arxiv.org/abs/2111.00396), [Mamba](https://arxiv.org/abs/2312.00752), [Mamba-2](https://arxiv.org/abs/2405.21060) |
| Vision transformers | — | [D2L](https://d2l.ai/) (vision transformers section) | ViT on CIFAR-10 from scratch vs a pretrained one | [ViT](https://arxiv.org/abs/2010.11929) |

## Graph neural networks
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Message passing, GCN, GAT, GraphSAGE | [Distill: A Gentle Introduction to GNNs](https://distill.pub/2021/gnn-intro/) → [Understanding Convolutions on Graphs](https://distill.pub/2021/understanding-gnns/) | [Hamilton: Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/) Ch. 5–7 · [CS224W notes](https://snap-stanford.github.io/cs224w-notes/) | [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) node classification on Cora · [UvA GNN tutorial](https://uvadlc-notebooks.readthedocs.io/) | [GCN](https://arxiv.org/abs/1609.02907), [GAT](https://arxiv.org/abs/1710.10903), [GraphSAGE](https://arxiv.org/abs/1706.02216), [GIN](https://arxiv.org/abs/1810.00826) |
| Graph ML applications (links, molecules, recsys) | [HF: Intro to Graph ML](https://huggingface.co/blog/intro-graphml) | [CS224W](https://cs224w.stanford.edu/) | Link prediction with PyG | [MPNN](https://arxiv.org/abs/1704.01212) |

## Autoencoders & representation learning
| Concept | Start here | Go deeper | Practice | Paper |
|---|---|---|---|---|
| Autoencoders, denoising AEs | — | [UDL](https://udlbook.github.io/udlbook/) Ch. 14, 17 · [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/) | Train a denoising AE on MNIST | [VAE](https://arxiv.org/abs/1312.6114) |
| Contrastive / self-supervised learning | [Lilian Weng: Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/) | [Lilian Weng: Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/) · [SSL Cookbook](https://arxiv.org/abs/2304.12210) | SimCLR on CIFAR-10 ([UvA tutorial](https://uvadlc-notebooks.readthedocs.io/)) | [SimCLR](https://arxiv.org/abs/2002.05709), [MAE](https://arxiv.org/abs/2111.06377) |

## Implementations of everything
- [labml.ai annotated paper implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations): transformers, GANs, diffusion, RL, optimizers, normalization layers… each with line-by-line notes.
- [rasbt/deeplearning-models](https://github.com/rasbt/deeplearning-models): a big collection of architectures as notebooks.
- [Hugging Face Transformers docs](https://github.com/huggingface/transformers): every model page links its paper, a quick way to go from paper to runnable code.
