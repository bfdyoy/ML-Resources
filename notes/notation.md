# Notation & symbols used in the notes

[All notes](README.md)

Every note uses the same symbols, so you only learn them once. When a book uses different notation, translate it with this table.

## Data

| Symbol | Meaning | Shape / type |
|---|---|---|
| $N$ | number of examples | integer |
| $d$ | number of features (input dimension) | integer |
| $x_i$ | the $i$-th example's features | vector in $\mathbb{R}^d$ |
| $y_i$ | the $i$-th target (label) | number or class index |
| $X$ | design matrix: one row per example | $N \times d$ |
| $y$ | all targets stacked | $N$-vector |
| $\mathcal{D}$ | the dataset $\lbrace(x_i, y_i)\rbrace_{i=1}^N$ | — |
| $K$ | number of classes | integer |

## Models and learning

| Symbol | Meaning |
|---|---|
| $\theta$ | all parameters of a model (general) |
| $w, b$ | weights and bias of a linear model or layer |
| $W^{(l)}$ | weight matrix of layer $l$, shape $d_\text{out}\times d_\text{in}$ |
| $f_\theta(x)$ | model output for input $x$ |
| $\hat y$ | a prediction |
| $z$ | a pre-activation, or *logit* (a score before the sigmoid/softmax) |
| $\sigma(z)$ | sigmoid $1/(1+e^{-z})$ (in statistics, $\sigma$ can also be a standard deviation; context makes it clear) |
| $\operatorname{softmax}(z)_k$ | $e^{z_k}/\sum_j e^{z_j}$ |
| $\ell(\hat y, y)$ | loss on one example |
| $L(\theta)$ or $\mathcal{L}$ | total or average loss (the objective) |
| $\nabla_\theta L$ | gradient: same shape as $\theta$ |
| $\eta$ | learning rate |
| $\lambda$ | regularization strength (or an eigenvalue, from context) |
| $t$ | an optimization step, or time/position in a sequence |

## Linear algebra

| Symbol | Meaning |
|---|---|
| $a^\top b$ | dot product |
| $\lVert x\rVert_2,\ \lVert x\rVert_1$ | L2 and L1 norms |
| $\lVert A\rVert_F$ | Frobenius norm (the root of the sum of squared entries) |
| $A^\top$, $A^{-1}$ | transpose, inverse |
| $I$ | identity matrix |
| $\operatorname{diag}(v)$ | diagonal matrix with $v$ on the diagonal |
| $U\Sigma V^\top$ | SVD |
| $\kappa$ | condition number |
| $\odot$ | element-wise (Hadamard) product |

## Probability

| Symbol | Meaning |
|---|---|
| $p(x)$ | probability (mass or density) |
| $p(y\mid x)$ | conditional probability |
| $\mathbb{E}[X]$, $\operatorname{Var}(X)$ | expectation, variance |
| $\mathcal{N}(\mu, \sigma^2)$ | Gaussian (normal) distribution |
| $x \sim p$ | $x$ is drawn from $p$ |
| $H(p)$, $H(p,q)$ | entropy, cross-entropy |
| $\mathrm{KL}(p\Vert q)$ | Kullback–Leibler divergence |
| $\hat\theta$ | an estimate of $\theta$ |

## Sequences & transformers

| Symbol | Meaning |
|---|---|
| $T$ | sequence length (number of tokens) |
| $d_\text{model}$ | width of the residual stream |
| $Q, K, V$ | queries, keys, values, each $T \times d_k$ |
| $h$ | number of attention heads |
| $V_\text{vocab}$ or $\lvert\mathcal{V}\rvert$ | vocabulary size |

## Conventions

- Vectors are columns. A batch of examples is stored as rows, so code writes `X @ W.T` where math writes $Wx$.
- $\log$ is the natural log unless the text says "bits" (then it's $\log_2$).
- "≈" in a worked example means rounded to the digits shown. The code cell prints the exact value.
