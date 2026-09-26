# Toolbox 02: Classical ML Algorithms

[← Toolbox](README.md) · Guided version: [Path 1 Core ML](../paths/01-core-ml-practitioner.md)

> **Rule of thumb:** for every algorithm, (1) get the picture, (2) read the ISLP section, (3) implement a toy version from scratch
> (see the [from-scratch ladder](../exercises/from-scratch-ladder.md)), (4) use the scikit-learn version on real data.

## Supervised: linear & probabilistic models
| Concept | Start here | Go deeper | Practice | Paper / reference |
|---|---|---|---|---|
| Linear regression, OLS, R² | [MLU-Explain: Linear Regression](https://mlu-explain.github.io/linear-regression/) | [ISLP](https://www.statlearning.com/) Ch. 3 | [sklearn: Linear models](https://scikit-learn.org/stable/modules/linear_model.html) | [ESL](https://hastie.su.domains/ElemStatLearn/) Ch. 3 |
| Ridge, lasso, elastic net | [MLU-Explain: Bias-Variance](https://mlu-explain.github.io/bias-variance/) | [ISLP](https://www.statlearning.com/) §6.2 | Coefficient paths with `lasso_path` | [ESL](https://hastie.su.domains/ElemStatLearn/) §3.4 |
| Logistic regression | [MLU-Explain: Logistic Regression](https://mlu-explain.github.io/logistic-regression/) | [ISLP](https://www.statlearning.com/) §4.3 · [CS229 notes](https://cs229.stanford.edu/main_notes.pdf) (classification) | Logistic regression in NumPy with GD | — |
| GLMs (Poisson, etc.) | — | [ISLP](https://www.statlearning.com/) §4.6 · [CS229 notes](https://cs229.stanford.edu/main_notes.pdf) (GLMs) | Poisson regression on count data (ISLP `Bikeshare`) | — |
| Generative classifiers: LDA, QDA, naive Bayes | — | [ISLP](https://www.statlearning.com/) §4.4–4.5 | [sklearn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html), a text classifier with `MultinomialNB` | — |
| Non-linear regression: splines, GAMs | — | [ISLP](https://www.statlearning.com/) Ch. 7 | `SplineTransformer` + ridge | — |

## Supervised: instance, kernel & tree methods
| Concept | Start here | Go deeper | Practice | Paper / reference |
|---|---|---|---|---|
| k-nearest neighbours, curse of dimensionality | — | [ISLP](https://www.statlearning.com/) §2.2.3, §3.5 | [sklearn: Nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html); code kNN from scratch | — |
| Support vector machines | — | [ISLP](https://www.statlearning.com/) Ch. 9 · [CS229 notes](https://cs229.stanford.edu/main_notes.pdf) (kernel methods, SVMs) | [sklearn: SVM](https://scikit-learn.org/stable/modules/svm.html): tune `C` and `gamma` on a grid, and visualise | [ESL](https://hastie.su.domains/ElemStatLearn/) Ch. 12 |
| The kernel trick & kernel approximation | — | [CS229 notes](https://cs229.stanford.edu/main_notes.pdf) (kernels) · [PML](https://probml.github.io/pml-book/) (kernel methods chapter) | [sklearn: Kernel approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html) (random Fourier features) | — |
| Gaussian processes | — | [PML](https://probml.github.io/pml-book/) (GP chapter) · [Bishop PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) §6.4 | [sklearn: Gaussian processes](https://scikit-learn.org/stable/modules/gaussian_process.html) | — |
| Decision trees | [MLU-Explain: Decision Trees](https://mlu-explain.github.io/decision-tree/) | [ISLP](https://www.statlearning.com/) §8.1 | [sklearn: Trees](https://scikit-learn.org/stable/modules/tree.html); build CART from scratch | — |
| Bagging & random forests | [MLU-Explain: Random Forest](https://mlu-explain.github.io/random-forest/) | [ISLP](https://www.statlearning.com/) §8.2.1–8.2.2 | [sklearn: Ensembles](https://scikit-learn.org/stable/modules/ensemble.html) | — |
| Gradient boosting | [explained.ai: How to explain gradient boosting](https://explained.ai/gradient-boosting/) + [StatQuest playlist](https://www.youtube.com/playlist?list=PLZ5DHV9_5h9vQwAImmNi1RfoTtSuOUjwM) | [ISLP](https://www.statlearning.com/) §8.2.3 · [ESL](https://hastie.su.domains/ElemStatLearn/) Ch. 10 | Boosting from scratch with sklearn trees; then [Kaggle Intermediate ML: XGBoost](https://www.kaggle.com/learn/intermediate-machine-learning) | [XGBoost](https://arxiv.org/abs/1603.02754), [CatBoost](https://arxiv.org/abs/1706.09516) |
| Stacking & blending | — | [Géron, *Hands-On ML*, Ch. 6](https://github.com/ageron/handson-mlp) | `StackingClassifier` | — |
| Trees vs deep learning on tabular data | — | — | Reproduce one result from the [tabular benchmark](https://github.com/LeoGrin/tabular-benchmark) | [Grinsztajn et al.](https://arxiv.org/abs/2207.08815) |

## Unsupervised
| Concept | Start here | Go deeper | Practice | Paper / reference |
|---|---|---|---|---|
| k-means & hierarchical clustering | — | [ISLP](https://www.statlearning.com/) §12.4 | [sklearn: Clustering](https://scikit-learn.org/stable/modules/clustering.html) (read the comparison figure first) | — |
| DBSCAN / HDBSCAN | — | [Géron, *Hands-On ML*, Ch. 8](https://github.com/ageron/handson-mlp) | [sklearn: Clustering](https://scikit-learn.org/stable/modules/clustering.html) | — |
| Gaussian mixtures & EM | — | [MML](https://mml-book.github.io/) Ch. 11 · [Bishop PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) Ch. 9 | [sklearn: Mixtures](https://scikit-learn.org/stable/modules/mixture.html); implement EM for 1-D GMM | — |
| PCA | [PCA Explained Visually](https://setosa.io/ev/principal-component-analysis/) | [ISLP](https://www.statlearning.com/) §12.2 · [MML](https://mml-book.github.io/) Ch. 10 | [sklearn: Decomposition](https://scikit-learn.org/stable/modules/decomposition.html) | — |
| t-SNE & UMAP | [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) · [Understanding UMAP](https://pair-code.github.io/understanding-umap/) | [sklearn: Manifold learning](https://scikit-learn.org/stable/modules/manifold.html) | Embed MNIST with both. Vary the hyperparameters. | [UMAP](https://arxiv.org/abs/1802.03426) |
| Density estimation (KDE) | — | [sklearn: Density estimation](https://scikit-learn.org/stable/modules/density.html) | KDE with bandwidth chosen by CV | — |
| Semi-supervised learning | — | [Lilian Weng: Learning with not enough data (semi-supervised)](https://lilianweng.github.io/posts/2021-12-05-semi-supervised/) | [sklearn: Semi-supervised](https://scikit-learn.org/stable/modules/semi_supervised.html) (self-training) | [FixMatch](https://arxiv.org/abs/2001.07685) |
| Active learning | — | [Lilian Weng: Active learning](https://lilianweng.github.io/posts/2022-02-20-active-learning/) | Uncertainty sampling loop on a small labeled set | — |

## Whole-field overviews
- [CS229 lecture notes](https://cs229.stanford.edu/main_notes.pdf): the derivations behind most of the above, compactly.
- [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/): ISLP's rigorous companion. Read the matching chapter after ISLP.
- [Hands-On ML notebooks (Géron)](https://github.com/ageron/handson-mlp): runnable versions of almost everything on this page.
- [ML-For-Beginners (Microsoft)](https://github.com/microsoft/ML-For-Beginners): a gentler 12-week curriculum, useful if you want lighter review lessons.
