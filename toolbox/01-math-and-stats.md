# Toolbox 01: Math & Statistics

[← Toolbox](README.md) · **Taught in:** [MATH-01](../lessons/math/01-linear-algebra.md) · [MATH-02](../lessons/math/02-calculus-optimization.md) · [MATH-03](../lessons/math/03-probability-statistics.md)

> Learn math **just in time**. When a lesson uses a concept you're shaky on, look it up here, fix it, and go back.

## Linear algebra
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| Vectors, matrices, linear maps | [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) Ch. 1–4 | [MML](https://mml-book.github.io/) Ch. 2 | [numpy-100](https://github.com/rougier/numpy-100) (exercises 1–40) |
| Dot products, norms, projections | 3B1B "Dot products and duality" | [MML](https://mml-book.github.io/) Ch. 3 | [Deep-ML](https://www.deep-ml.com/problems) linear-algebra problems |
| Eigenvectors, eigendecomposition | 3B1B "Eigenvectors and eigenvalues" | [MML](https://mml-book.github.io/) §4.2–4.4 | Implement power iteration in NumPy |
| SVD & low-rank approximation | [PCA Explained Visually](https://setosa.io/ev/principal-component-analysis/) | [MML](https://mml-book.github.io/) §4.5–4.6 | Compress an image with truncated SVD |
| Visual NumPy intuition | [Jay Alammar: A Visual Intro to NumPy](https://jalammar.github.io/visual-numpy/) | [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) (NumPy chapter) | [numpy-100](https://github.com/rougier/numpy-100) |

## Calculus, matrix calculus & optimization
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| Derivatives, chain rule | [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) Ch. 1–4 | [MML](https://mml-book.github.io/) §5.1–5.2 | Differentiate by hand, then check with `torch.autograd` |
| Gradients of vectors and matrices | [The Matrix Calculus You Need for Deep Learning](https://explained.ai/matrix-calculus/) (Parr & Howard) | [MML](https://mml-book.github.io/) §5.3–5.5 | Derive ∂L/∂W for a linear layer, then verify numerically |
| Backprop as reverse-mode autodiff | [colah: Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/) | [MML](https://mml-book.github.io/) §5.6 | [micrograd](https://github.com/karpathy/micrograd), [Autodiff-Puzzles](https://github.com/srush/Autodiff-Puzzles) |
| Gradient descent, momentum | [Why Momentum Really Works](https://distill.pub/2017/momentum/) | [MML](https://mml-book.github.io/) §7.1 | Implement GD, momentum, and Adam on the Rosenbrock function |
| Convexity, Lagrange multipliers | — | [MML](https://mml-book.github.io/) §7.2–7.3 · [Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) Ch. 2–5 (deep) | Derive the SVM dual (see [classical ML](02-classical-ml.md)) |

## Probability & statistics
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| Probability basics, Bayes' rule | [Seeing Theory](https://seeing-theory.brown.edu/) ch. 1–2, 5 | [MML](https://mml-book.github.io/) §6.1–6.3 · [Stat 110](https://stat110.hsites.harvard.edu/) | [Think Bayes](https://greenteapress.com/wp/think-bayes/) Ch. 1–4 |
| Distributions, expectation, variance | [Seeing Theory](https://seeing-theory.brown.edu/) ch. 3 | [MML](https://mml-book.github.io/) §6.4–6.5 (the Gaussian) | Simulate the CLT in NumPy |
| Estimation: MLE, MAP | — | [MML](https://mml-book.github.io/) §8.3 · [CS229 notes](https://cs229.stanford.edu/main_notes.pdf) (Part I: linear regression → probabilistic interpretation) | Derive MLE for a Bernoulli and a Gaussian |
| Hypothesis testing, confidence intervals, bootstrap | [Seeing Theory](https://seeing-theory.brown.edu/) ch. 4 | [Think Stats](https://greenteapress.com/wp/think-stats-3e/) · [ISLP](https://www.statlearning.com/) §5.2 (bootstrap), Ch. 13 (multiple testing) | Bootstrap a CI for a model's accuracy |
| Bayesian inference | [Seeing Theory](https://seeing-theory.brown.edu/) ch. 5 | [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) · [Statistical Rethinking 2026](https://github.com/rmcelreath/stat_rethinking_2026) | See [EL-02](../lessons/electives/02-bayesian-probabilistic-ml.md) |
| A/B testing & experiment design | [Seeing Theory](https://seeing-theory.brown.edu/) ch. 4 | [Think Stats](https://greenteapress.com/wp/think-stats-3e/) (hypothesis testing chapters) | A Bayesian A/B test (EL-02 mini-project) |

## Information theory
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| Entropy, cross-entropy, KL divergence | [UDL](https://udlbook.github.io/udlbook/) Ch. 5 (loss functions from likelihood) | [MacKay, ITILA](http://www.inference.org.uk/mackay/itila/book.html) Ch. 1–2, 8 · [PML](https://probml.github.io/pml-book/) info-theory chapter | Show that cross-entropy = entropy + KL, numerically |
| Mutual information (feature selection) | [Kaggle Feature Engineering](https://www.kaggle.com/learn/feature-engineering) (MI lesson) | [MacKay, ITILA](http://www.inference.org.uk/mackay/itila/book.html) Ch. 8 | `mutual_info_classif` in scikit-learn |

## Reference cards
- [MML book PDF](https://mml-book.github.io/book/mml-book.pdf): keep it open while reading papers.
- [CS229 lecture notes](https://cs229.stanford.edu/main_notes.pdf): compact derivations of most classical ML.
