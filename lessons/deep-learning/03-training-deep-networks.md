# DL-03: Training Deep Networks Well

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Deep Learning | ~9 h | L2 | DL-02, CORE-04 |

## Why this matters
Getting a network to train *well* is a craft. It takes the right initialization, normalization, optimizer, learning-rate
schedule, and regularization, plus a disciplined debugging process. This lesson is the difference between
"it doesn't converge" and "I know exactly what to try next".

## Learning goals
By the end you can:
- Explain vanishing/exploding gradients, and how initialization (Xavier/He) and residual connections address them.
- Compare SGD+momentum, RMSProp, and Adam/AdamW, and pick learning-rate schedules (warmup, cosine, one-cycle).
- Use BatchNorm/LayerNorm, dropout, weight decay, and data augmentation, knowing *why* each one helps.
- Follow a systematic debugging recipe: overfit one batch, check the loss at init, and visualize activations and gradients.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Training Deep Networks Well](../../notes/deep-learning/03-training-deep-networks.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [UDL](https://udlbook.github.io/udlbook/) (Prince) | Ch. 6 "Fitting Models", Ch. 7 "Gradients and Initialization", Ch. 9 "Regularization". Run the notebooks for 6, 7 and 9. | 3 h |
| 2 | **Intuition** | [Why Momentum Really Works](https://distill.pub/2017/momentum/) (Distill) | Play with the interactive panels. The maths is optional. | 30 min |
| 3 | **Read** | [Karpathy: A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) | The whole post. Print it and keep it next to you. | 45 min |
| 4 | **Build** | [Karpathy, Lecture 4: Activations & Gradients, BatchNorm](https://youtu.be/P6sfmUTpUmc) | Code along. Plot activation and gradient histograms per layer. | 2 h |
| 5 | **Read** | [CS231n notes](https://cs231n.github.io/) | "Neural Networks Part 2" (data preprocessing, weight init, batch norm, regularization) and "Part 3" (gradient checks, sanity checks, babysitting the learning process, hyperparameter optimization) | 1.5 h |

## Check your understanding
1. Why does He initialization scale by `sqrt(2/fan_in)` for ReLU networks?
2. What problem does BatchNorm solve, and why does it behave differently in train and eval mode?
3. AdamW vs Adam + L2: what's the difference, and why does it matter?
4. Why use learning-rate warmup with Adam and large batches?
5. What should the loss be at initialization for a 10-class softmax classifier, and why check it?
6. *(debug)* You can't overfit a single batch of 32 examples. What does that tell you, and what do you check first?

## Mini-project
**Task:** Take your DL-02 Fashion-MNIST model and make it 10 layers deep. First show that it trains badly with naive init and no
normalization. Then fix it step by step (He init → BatchNorm → residual connections → AdamW + cosine schedule),
logging the effect of each change.
**Deliverable:** An ablation table and gradient-norm plots for each step.

## Go deeper
- [Nielsen Ch. 3 "Improving the way neural networks learn"](http://neuralnetworksanddeeplearning.com/chap3.html) and [Ch. 5 "Why are deep neural networks hard to train?"](http://neuralnetworksanddeeplearning.com/chap5.html)
- [Géron, *Hands-On ML*, Ch. 11 "Training Deep Neural Networks"](https://github.com/ageron/handson-mlp): a practitioner checklist with code.
- [UDL](https://udlbook.github.io/udlbook/) Ch. 11 "Residual Networks" and Ch. 20 "Why Does Deep Learning Work?"
- [MLU-Explain: Double Descent](https://mlu-explain.github.io/double-descent/): revisit it now that you train big models.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 04: Deep learning fundamentals](../../toolbox/04-deep-learning-fundamentals.md), for every concept in this lesson, with alternatives.
- **Papers:** [Optimization, training & generalization](../../papers/01-optimization-training-generalization.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 20–22.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
- **Playbook:** [DL tricks §1–§4 and §7](../../playbook/03-deep-learning-tricks.md) · [SGD vs AdamW](../../playbook/01-choosing-algorithms.md#7-optimizers-sgd--momentum-vs-adamw) · [debugging training](../../playbook/06-debugging-playbook.md#b-deep-learning-training).
