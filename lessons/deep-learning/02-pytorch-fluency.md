# DL-02: PyTorch Fluency

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Deep Learning | ~7 h | L1→L2 | DL-01 |

## Why this matters
You now know what autograd does. This lesson makes you *fast* with the tool everyone uses: tensors, `nn.Module`,
`Dataset`/`DataLoader`, the training loop, devices, and saving/loading. Every later lesson assumes this fluency.

## Learning goals
By the end you can:
- Manipulate tensors (shapes, broadcasting, indexing, dtype/device) without checking the docs every line.
- Write a clean training/eval loop from memory: `model.train()` / `model.eval()`, `zero_grad`, `backward`, `step`, `torch.no_grad()`.
- Build a custom `Dataset` and use a `DataLoader` with batching and shuffling.
- Structure a project as modules (data, model, train, utils), and track experiments.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read + Build** | [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) (online book) | [00 Fundamentals](https://www.learnpytorch.io/00_pytorch_fundamentals/), [01 Workflow](https://www.learnpytorch.io/01_pytorch_workflow/), 02 Classification. Do the exercises at the end of each. | 3.5 h |
| 2 | **Read + Build** | [learnpytorch.io](https://www.learnpytorch.io/) | 04 Custom Datasets, 05 Going Modular | 2 h |
| 3 | **Read** | [D2L](https://d2l.ai/) | Chapter "Builders' Guide" (layers and modules, parameter management, custom layers, file I/O, GPUs) | 1 h |
| 4 | **Build** | Re-type from memory | Write the full train/eval loop in an empty file, with no references. Repeat until it works first time. | 30 min |

**Notes for the learner:** The learnpytorch.io early sections move slowly for an intermediate learner, so speed-read the
prose and do *all* the exercises. The exercises are where the fluency comes from.

## Check your understanding
1. What's the difference between `tensor.view` and `tensor.reshape`? When does `view` fail?
2. Why do you need `optimizer.zero_grad()` every step?
3. What do `model.eval()` and `torch.no_grad()` each do? Why do you usually want both at inference?
4. What happens if your model is on GPU and your batch is on CPU?
5. *(debug)* Training loss drops but validation accuracy stays at chance level. Name three PyTorch-specific bugs that cause this.

## Mini-project
**Task:** Train an MLP on Fashion-MNIST (`torchvision.datasets.FashionMNIST`) using a modular project layout
(`data.py`, `model.py`, `train.py`), with command-line arguments, and log metrics to TensorBoard or a CSV.
**Deliverable:** A repo folder that trains with `python train.py --epochs 5 --lr 1e-3`.

## Go deeper
- [learnpytorch.io 07: Experiment Tracking](https://www.learnpytorch.io/07_pytorch_experiment_tracking/)
- [fastbook](https://github.com/fastai/fastbook) Ch. 4 "MNIST Basics": builds SGD and a learner from scratch. A great bridge from DL-01.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 04: Deep learning fundamentals](../../toolbox/04-deep-learning-fundamentals.md), for every concept in this lesson, with alternatives.
- **Papers:** [Optimization, training & generalization](../../papers/01-optimization-training-generalization.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rung 20.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
