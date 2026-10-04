# Papers: Optimization, Training & Generalization

[← Papers library](README.md) · Background: [DL-03 Training Deep Networks Well](../lessons/deep-learning/03-training-deep-networks.md) · [Toolbox: DL fundamentals](../toolbox/04-deep-learning-fundamentals.md)

## Optimizers & learning-rate schedules
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747) | 2016 | ⭐ A survey that explains SGD, momentum, Adagrad, RMSProp, and Adam in one place. Start here. | L1 | CORE-02 |
| [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | 2014 | ⭐ The default optimizer. Read §2 (the algorithm) and §3 (bias correction). | L2 | DL-03 |
| [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) | 2017 | Why L2 regularization ≠ weight decay under Adam. AdamW is what everyone actually uses. | L2 | DL-03 |
| [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) | 2015 | Introduces the LR range test ("LR finder"), which is still extremely useful. | L1 | DL-03 |
| [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) | 2016 | Cosine annealing, the default LR schedule today. | L2 | DL-03 |
| [A disciplined approach to neural network hyper-parameters: Part 1](https://arxiv.org/abs/1803.09820) | 2018 | One-cycle policy, and how LR, batch size, momentum, and weight decay interact. Very practical. | L2 | DL-03 |
| [Practical recommendations for gradient-based training of deep architectures](https://arxiv.org/abs/1206.5533) | 2012 | Bengio's classic practitioner's guide. Still sound advice. | L2 | DL-03 |

## Initialization, activations & normalization
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Delving Deep into Rectifiers (He init, PReLU)](https://arxiv.org/abs/1502.01852) | 2015 | ⭐ Derives the initialization used for ReLU nets. | L2 | DL-03 |
| [Batch Normalization](https://arxiv.org/abs/1502.03167) | 2015 | ⭐ One of the most influential training tricks. Its "internal covariate shift" explanation has since been disputed. | L2 | DL-03 |
| [Layer Normalization](https://arxiv.org/abs/1607.06450) | 2016 | The normalization transformers use. | L2 | DL-06 |
| [Group Normalization](https://arxiv.org/abs/1803.08494) | 2018 | Normalization that doesn't depend on batch size (useful for detection and segmentation). | L2 | DL-04 |
| [Root Mean Square Layer Normalization (RMSNorm)](https://arxiv.org/abs/1910.07467) | 2019 | The simpler LayerNorm used in Llama-style models. | L2 | DL-06 |
| [On Layer Normalization in the Transformer Architecture (Pre-LN)](https://arxiv.org/abs/2002.04745) | 2020 | Why modern transformers put the norm *before* attention/MLP, so warmup matters less. | L3 | DL-06 |
| [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) | 2016 | The activation used in BERT and GPT. | L1 | DL-06 |
| [GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202) | 2020 | A very short paper. SwiGLU is the MLP of modern LLMs. | L2 | DL-06 |

## Regularization & training tricks
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Improving neural networks by preventing co-adaptation of feature detectors (Dropout)](https://arxiv.org/abs/1207.0580) | 2012 | The original dropout paper. | L1 | DL-03 |
| [Rethinking the Inception Architecture (label smoothing, §7)](https://arxiv.org/abs/1512.00567) | 2015 | Read §7 for label smoothing. | L2 | DL-04 |
| [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) | 2017 | A simple data augmentation that also regularizes. | L2 | DL-04 |
| [Bag of Tricks for Image Classification with CNNs](https://arxiv.org/abs/1812.01187) | 2018 | ⭐ A dozen small tricks with measured gains. A model of how to do ablations. | L1 | DL-04 |
| [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | 2017 | FP16/BF16 training with loss scaling, which every large model uses. | L2 | DL-03 |
| [Training Deep Nets with Sublinear Memory Cost (gradient checkpointing)](https://arxiv.org/abs/1604.06174) | 2016 | Trade compute for memory. | L2 | DL-07 |
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | 2015 | ⭐ Knowledge distillation: soft targets and temperature. | L2 | DL-03 |
| [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) | 2017 | The linear LR-scaling rule plus gradual warmup, and zero-init of the last BN in each residual block. See [playbook 3 §2](../playbook/03-deep-learning-tricks.md#2-warm-up-then-decay-one-cycle-or-cosine). | L2 | DL-03 |
| [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) | 2019 | Why smoothing improves calibration, and why smoothed teachers distill worse ([playbook 3 §5](../playbook/03-deep-learning-tricks.md#5-label-smoothing)). | L2 | DL-04 |
| [CutMix](https://arxiv.org/abs/1905.04899) | 2019 | Paste patches between images and mix the labels by area: mixup's stronger sibling for vision. | L2 | DL-04 |
| [Averaging Weights Leads to Wider Optima (SWA)](https://arxiv.org/abs/1803.05407) | 2018 | Averaging the tail of SGD's iterates finds flatter, better-generalizing solutions, almost for free ([playbook 3 §7](../playbook/03-deep-learning-tricks.md#7-average-the-weights-ema-swa-model-soups)). | L2 | DL-03 |
| [Model soups](https://arxiv.org/abs/2203.05482) | 2022 | Average the *weights* of several fine-tunes from one pretrained model: ensemble-like gains at single-model cost. | L2 | CV-02 |
| [Sharpness-Aware Minimization (SAM)](https://arxiv.org/abs/2010.01412) | 2020 | Descend from the worst nearby point to seek flat minima, at about 2× the compute per step. | L3 | DL-03 |

## Generalization: what's actually going on
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530) | 2016 | ⭐ Networks can memorize random labels, which shattered the classical story. | L2 | CORE-04 |
| [Deep Double Descent](https://arxiv.org/abs/1912.02292) | 2019 | ⭐ Test error vs model size and vs epochs: descent, ascent, then descent again. | L2 | CORE-04 |
| [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) | 2017 | Great pictures. Shows why skip connections make optimization easier. | L2 | DL-03 |
| [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) | 2018 | Sparse subnetworks that train on their own. | L2 | DL-03 |
| [Neural Tangent Kernel](https://arxiv.org/abs/1806.07572) | 2018 | Infinite-width theory. Math-heavy. | L3 | — |
| [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) | 2022 | Delayed generalization long after overfitting. A fun thing to reproduce. | L2 | DL-03 |
| [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) *(guide, not a paper)* | 2023 | Google's systematic approach to hyperparameter tuning. | L2 | DL-03 |
