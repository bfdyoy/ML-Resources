# CORE-09: Kernel Methods, SVMs, Nearest Neighbours & Gaussian Processes

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Core ML | ~8 h | L2 | CORE-03, CORE-04 · math: [MATH-01](../math/01-linear-algebra.md) block B, [MATH-02](../math/02-calculus-optimization.md) block B |

## Why this matters
Before deep learning, kernel methods were *the* way to learn non-linear functions with clean guarantees. They still matter.
SVMs are strong on small, high-dimensional data. kNN is the baseline behind every vector search and RAG system. Gaussian processes
give calibrated uncertainty and power Bayesian optimization (CORE-12). The "kernel trick" is also a useful lens on
why wide neural networks behave the way they do.

## Learning goals
By the end you can:
- Explain kNN, the role of the distance metric, and the curse of dimensionality.
- Explain the maximal-margin classifier, soft margins (`C`), and the hinge loss.
- Explain the kernel trick, and choose between linear, polynomial, and RBF kernels (and tune `gamma`).
- Explain a Gaussian process as a distribution over functions, and read its predictive mean and variance.
- Scale kernel methods with random Fourier features / kernel approximation.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Kernel Methods, SVMs, Nearest Neighbours & Gaussian Processes](../../notes/core-ml/09-kernels-svms-nearest-neighbours.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [ISLP](https://www.statlearning.com/) | §2.2.3 (the Bayes classifier and KNN) and §3.5 (KNN regression vs linear regression) | 45 min |
| 2 | **Intuition** | [StatQuest: Support Vector Machines, Part 1](https://www.youtube.com/watch?v=efR1C6CvhmE) | Main ideas: margins, soft margins, and kernels (~20 min) | 20 min |
| 3 | **Read** | [ISLP](https://www.statlearning.com/) Ch. 9 "Support Vector Machines" | All of §9.1–9.5, plus the Python lab | 2 h |
| 4 | **Read** | [CS229 lecture notes](https://cs229.stanford.edu/main_notes.pdf) | The "Kernel methods" and "Support vector machines" chapters, for the dual and the kernel derivation | 1.5 h |
| 5 | **Intuition** | [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/) (Distill) | The whole article. Play with the kernel and the observations. | 45 min |
| 6 | **Build** | scikit-learn: [SVM](https://scikit-learn.org/stable/modules/svm.html), [Nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html), [Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html), [Kernel approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html) | Grid-search `C`×`gamma` and plot the heatmap; fit a GP regressor and plot its uncertainty band; compare exact RBF-SVM with `RBFSampler` + a linear SVM for speed | 1.5 h |

**Notes for the learner:** The dual form of the SVM is where the kernel trick comes from, so don't skip CS229's derivation
even if it feels heavy. You only need to follow it once.

## Check your understanding
1. Why does kNN degrade in high dimensions? What happens to the ratio of nearest to farthest distances?
2. Which training points determine an SVM's decision boundary, and why?
3. What do large vs small `C` do to the margin and to bias/variance?
4. What does the RBF `gamma` control geometrically? What does overfitting look like at very large `gamma`?
5. Why can a kernel compute an inner product in an infinite-dimensional space cheaply?
6. What does a GP's predictive variance tell you that a point estimate doesn't? Where is it largest?
7. *(debug)* Your RBF-SVM gets 100% train accuracy and 60% test accuracy, and you never scaled the features. What two things do you fix first?

## Mini-project
**Task:** On a small, high-dimensional dataset, compare kNN, a linear SVM, an RBF-SVM, and a GBM with proper nested CV. Then fit a
GP regressor to a 1-D noisy function and show how the uncertainty shrinks as you add points.
**Dataset:** `sklearn.datasets.load_breast_cancer` (classification). A synthetic sine plus noise (GP).
**Deliverable:** A comparison table, a `C`×`gamma` heatmap, and a GP uncertainty plot.

## Go deeper
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) Ch. 12 (SVMs and flexible discriminants).
- [Probabilistic Machine Learning](https://probml.github.io/pml-book/) (Book 1): the "Kernel methods" chapter (GPs and SVMs in one probabilistic frame).
- [Bishop, PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) Ch. 6–7 (kernel methods, sparse kernel machines).

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 02: Classical ML](../../toolbox/02-classical-ml.md) (instance, kernel & tree methods).
- **Papers:** the kernel section of this lesson is textbook material. For GPs used as a tuner, see [Practical Bayesian Optimization](https://arxiv.org/abs/1206.2944) (CORE-12).
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rung 8 (vectorized kNN).
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) (kNN, kernel, and SVM problems) · more in [exercises/](../../exercises/README.md).
