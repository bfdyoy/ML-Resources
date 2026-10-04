# CORE-09 notes: Kernel Methods, SVMs, Nearest Neighbours & Gaussian Processes

[← Lesson CORE-09](../../lessons/core-ml/09-kernels-svms-nearest-neighbours.md) · [All notes](../README.md) · [← CORE-08 notes](08-interpretability-and-responsible-ml.md) · Next: [CORE-10 notes →](10-data-centric-ml.md)

> **Reading time** ≈ 70 min. This is the most mathematical note in Path 1, so take it in two sittings (§1–3, then §4–6).
> **You need:** [MATH-01 notes](../math/01-linear-algebra.md) §B (dot products, distances), [MATH-02 notes](../math/02-calculus-optimization.md) §B7 (Lagrange multipliers), [MATH-03 notes](../math/03-probability-statistics.md) §A4 (the multivariate Gaussian).

---

## Where we are

Every model so far learned **one weight per feature** (linear models) or **splits on features** (trees). This lesson's models think in terms of **similarity between examples**:

- *predict like the training points that look like this one* (kNN);
- *use only the hard, borderline examples* (SVMs);
- *measure similarity in a richer space without ever building that space* (kernels);
- *put a probability distribution over whole functions* (Gaussian processes).

The common thread is the **kernel**, a similarity function, which comes back in attention (DL-06) and in retrieval (GEN-05).

---

## 1. k-nearest neighbours

To predict at $x$, find the $k$ closest training points and average their targets (regression) or take a vote (classification). There's no training, only memory.

- **$k$ is the complexity knob.** $k = 1$ follows every noisy point (low bias, high variance). A large $k$ smooths everything (high bias). Choose it by CV.
- **The metric is the model.** Euclidean distance on unscaled features means "income in dollars" decides everything (as in CORE-06). Scale first, and choose a metric that suits the data: cosine for text and embeddings.

### 1.1 The curse of dimensionality

Two facts explain why kNN (and any purely distance-based method) degrades as $d$ grows:

1. **Neighbourhoods aren't local.** In a unit hypercube with uniform data, a sub-cube that contains a fraction $r$ of the points has edge length $r^{1/d}$.
   To capture 10% of the data in $d = 10$ dimensions, you need an edge of $0.1^{1/10} \approx 0.79$, which is **79% of the range on every axis**. Your "nearest" neighbours are far away.
2. **Distances concentrate.** For random points, the gap between the nearest and the farthest neighbour, relative to the nearest, $\frac{d_{\max} - d_{\min}}{d_{\min}}$, shrinks toward 0 as $d$ grows.
   When everything is about equally far away, "nearest" stops meaning anything.

Real data often lies near a **low-dimensional manifold** inside the high-dimensional space, and that rescues kNN on images or embeddings. Learned embeddings exist to create spaces where distance *is* meaningful.

---

## 2. Support vector machines

### 2.1 The geometry of the margin

A linear classifier predicts $\operatorname{sign}(w^\top x + b)$. The **signed distance** from a point to the hyperplane $w^\top x + b = 0$ is $\frac{w^\top x + b}{\lVert w\rVert}$.
Scale $w$ and $b$ so that the closest points satisfy $y_i(w^\top x_i + b) = 1$. The **margin** (the distance from the boundary to the nearest point) is then $1/\lVert w\rVert$.
Maximizing the margin means:

```math
\min_{w,b}\ \tfrac12\lVert w\rVert^2 \quad\text{s.t.}\quad y_i(w^\top x_i + b) \ge 1\ \ \forall i .
```

Why a big margin? It's the boundary that is **most robust**: small perturbations of the data can't flip predictions. That's a form of regularization.

### 2.2 Soft margin and the hinge loss

Real data overlaps, so allow violations with slack variables $\xi_i \ge 0$:

```math
\min_{w,b,\xi}\ \tfrac12\lVert w\rVert^2 + C\sum_i \xi_i \quad\text{s.t.}\quad y_i(w^\top x_i + b) \ge 1 - \xi_i .
```

At the optimum $\xi_i = \max(0,\ 1 - y_i f(x_i))$. Substituting that in gives an unconstrained problem: **hinge loss plus L2 regularization**:

```math
\min_{w,b}\ \sum_i \max\big(0,\ 1 - y_i(w^\top x_i + b)\big) + \frac{1}{2C}\lVert w\rVert^2 .
```

So an SVM is "logistic regression with a different loss". The hinge is **exactly zero** for points beyond the margin, while the log loss is never exactly zero. That's why only some points matter.
**$C$ is the inverse of the regularization strength:** a large $C$ means few violations are tolerated, a narrow margin, low bias and high variance. A small $C$ means a wide margin, more violations, and a smoother boundary.

### 2.3 The dual, support vectors, and where kernels come from

Apply Lagrange multipliers $\alpha_i \ge 0$ to the constraints (the KKT conditions) and eliminate $w$ and $b$. You get the **dual problem**:

```math
\max_{\alpha}\ \sum_i \alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j\, \boxed{x_i^\top x_j} \quad\text{s.t.}\quad 0 \le \alpha_i \le C,\ \ \sum_i \alpha_i y_i = 0,
```

with $w = \sum_i \alpha_i y_i x_i$, so the prediction is $f(x) = \sum_i \alpha_i y_i\, x_i^\top x + b$. Two facts drop out:

- **Support vectors.** Complementary slackness forces $\alpha_i = 0$ for every point strictly outside the margin. **Only the points on or inside the margin (the support vectors) determine the boundary.**
  Delete any other point and the solution doesn't change.
- **The data appear only through dot products $x_i^\top x_j$.** Replace each one with a kernel $k(x_i, x_j)$ and you have a non-linear SVM, *with the same optimization problem*.

---

## 3. The kernel trick

### 3.1 A kernel is a dot product in a feature space

$k(x, z) = \phi(x)^\top\phi(z)$ for some feature map $\phi$. *Worked example* in 2-D:

```math
(x^\top z)^2 = (x_1 z_1 + x_2 z_2)^2 = x_1^2 z_1^2 + 2x_1x_2z_1z_2 + x_2^2z_2^2 = \phi(x)^\top\phi(z),\quad \phi(x) = (x_1^2,\ \sqrt2\,x_1x_2,\ x_2^2).
```

Computing $(x^\top z)^2$ costs one dot product and a square. Building $\phi$ explicitly costs more, and for degree-$p$ polynomials in $d$ dimensions, $\phi$ has about $d^p$ coordinates.
**The kernel computes the dot product in the big space without ever going there.**

### 3.2 The RBF kernel: an infinite-dimensional feature space

```math
k(x, z) = \exp\big(-\gamma\lVert x - z\rVert^2\big).
```

Expand $\exp(2\gamma x^\top z)$ as a Taylor series and you find polynomial features of **every** degree. The feature space is infinite-dimensional, yet $k$ costs one distance computation.

**What $\gamma$ does geometrically:** $1/\sqrt{\gamma}$ is the **radius of influence** of each training point.

- Small $\gamma$: wide bumps and a smooth, nearly linear boundary.
- Large $\gamma$: each point influences only a tiny neighbourhood. The boundary wraps around individual points, giving islands around every training example, 100% training accuracy, and poor test accuracy. It's a kNN-like memorizer.

Tune $C$ and $\gamma$ **together** on a log grid, *after scaling the features*. The RBF distance is meaningless on unscaled data.

**Which kernels are valid?** Any $k$ whose Gram matrix $K_{ij} = k(x_i, x_j)$ is always positive semi-definite (Mercer's condition). Sums and products of valid kernels are valid.

### 3.3 Scaling kernels up: random Fourier features

Kernel methods need the $N \times N$ Gram matrix, so $O(N^2)$ memory and up to $O(N^3)$ time. **Random Fourier features** approximate the RBF kernel with an explicit, finite map:

```math
z(x) = \sqrt{\tfrac{2}{D}}\cos(Wx + b),\quad W_{ij}\sim\mathcal N(0, 2\gamma),\ b_i\sim U[0, 2\pi] \quad\Rightarrow\quad z(x)^\top z(x') \approx k(x, x').
```

Then train a *linear* model on $z(x)$, which costs $O(ND)$. That's scikit-learn's `RBFSampler`. The same "random features" idea powered the double-descent demo in CORE-04.

---

## 4. Gaussian processes: a distribution over functions

### 4.1 The prior

A GP says: for any set of inputs, the function values are jointly Gaussian, with covariance given by the kernel:

```math
f \sim \mathcal{GP}(0, k) \quad\Longleftrightarrow\quad \big(f(x_1),\dots,f(x_n)\big) \sim \mathcal N(0, K),\ \ K_{ij} = k(x_i, x_j).
```

The kernel encodes your beliefs: with RBF, nearby inputs have strongly correlated outputs, so the functions are **smooth**. The length scale ($1/\sqrt{2\gamma}$) sets how quickly that correlation fades.

### 4.2 The posterior, from Gaussian conditioning

With noisy observations $y = f(X) + \varepsilon$, $\varepsilon \sim \mathcal N(0, \sigma^2 I)$, condition the joint Gaussian of $(y, f_*)$ at a new point $x_*$. Write $k_* = k(X, x_*)$:

```math
\mu_* = k_*^\top (K + \sigma^2 I)^{-1} y, \qquad \sigma_*^2 = k(x_*, x_*) - k_*^\top (K + \sigma^2 I)^{-1} k_* .
```

- **The mean** is a weighted combination of the training targets, with weights driven by similarity. It's identical to **kernel ridge regression** with $\lambda = \sigma^2$.
- **The variance** starts at the prior variance $k(x_*, x_*)$ and is *reduced* by how much the training data "explains" $x_*$. **It's largest far from the data** and smallest near observed points. It doesn't depend on $y$ at all, only on *where* you've looked.
- That's what a point estimate can't give you: **calibrated "I don't know"** away from the data. It's the basis of Bayesian optimization (CORE-12), which samples where the mean is good *or* the uncertainty is high.

Kernel hyperparameters (length scale, noise) are fitted by maximizing the **marginal likelihood**, a built-in Occam's razor. The cost is $O(N^3)$ for the matrix solve, so plain GPs suit thousands of points, not millions.

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.datasets import make_moons
rng = np.random.default_rng(0)

# Curse of dimensionality: distance concentration and the 10%-neighbourhood edge
for d in [2, 10, 100, 1000]:
    P = rng.random((1000, d)); q = rng.random(d)
    dist = np.linalg.norm(P - q, axis=1)
    print(f"d={d:4d}  (max-min)/min = {(dist.max()-dist.min())/dist.min():6.2f}   edge for 10% of data = {0.1**(1/d):.2f}")

# The polynomial kernel equals a dot product of explicit features
x, z = np.array([1.0, 2.0]), np.array([3.0, 1.0])
phi = lambda v: np.array([v[0]**2, np.sqrt(2) * v[0] * v[1], v[1]**2])
print("(x.z)^2 =", (x @ z) ** 2, "  phi(x).phi(z) =", phi(x) @ phi(z))
```

```python
# C and gamma on two moons: support vectors, and train vs test accuracy
X, y = make_moons(400, noise=0.25, random_state=0)
Xtr, ytr, Xte, yte = X[:200], y[:200], X[200:], y[200:]
for C, g in [(0.1, 1), (10, 1), (10, 100), (1000, 1000)]:
    m = SVC(C=C, gamma=g).fit(Xtr, ytr)
    print(f"C={C:<5} gamma={g:<5} support vectors={m.n_support_.sum():3d}  train={m.score(Xtr,ytr):.2f}  test={m.score(Xte,yte):.2f}")

# Random Fourier features approximate the RBF kernel
gamma = 0.5
A = rng.normal(size=(5, 3))
K_true = rbf_kernel(A, gamma=gamma)
for D in [10, 100, 10_000]:
    Z = RBFSampler(gamma=gamma, n_components=D, random_state=0).fit_transform(A)
    print(f"D={D:6d}  max |K - Z Z^T| = {np.abs(K_true - Z @ Z.T).max():.3f}")

# GP posterior by hand vs sklearn; variance is largest far from the data
Xg = np.array([[-2.0], [-1.0], [0.0], [1.5]]); yg = np.sin(Xg[:, 0])
ls, noise = 1.0, 0.01
k = lambda a, b: np.exp(-((a - b.T) ** 2) / (2 * ls ** 2))
Xs = np.array([[0.5], [4.0]])
Kinv = np.linalg.inv(k(Xg, Xg) + noise * np.eye(len(Xg)))
mu = k(Xs, Xg) @ Kinv @ yg
var = 1 - np.sum(k(Xs, Xg) @ Kinv * k(Xs, Xg), axis=1)
gp = GaussianProcessRegressor(RBF(ls, "fixed"), alpha=noise, optimizer=None).fit(Xg, yg)
m_sk, s_sk = gp.predict(Xs, return_std=True)
print("by hand  mean", mu.round(3), " std", np.sqrt(var).round(3))
print("sklearn  mean", m_sk.round(3), " std", s_sk.round(3), "  <- x=4 is far from the data: big std")
```

---

## Pitfalls & misconceptions

- **RBF-SVM or kNN on unscaled features.** Distances are dominated by the features with the largest units.
- **Tuning $C$ and $\gamma$ separately, or on a linear grid.** Use a joint log grid.
- **Treating SVM scores as probabilities.** `SVC(probability=True)` bolts on Platt scaling with internal CV. It's better to calibrate explicitly (CORE-11).
- **Plain GPs or kernel SVMs on a million rows.** Use random features, inducing points, or a different model.
- **"kNN has no hyperparameters."** $k$, the metric, the scaling, and the weighting all matter.

## Cheat sheet

| Item | Formula |
|---|---|
| Margin | $1/\lVert w\rVert$ |
| Soft-margin SVM | hinge loss + $\frac{1}{2C}\lVert w\rVert^2$ |
| Dual prediction | $f(x) = \sum_i\alpha_i y_i k(x_i, x) + b$; $\alpha_i > 0$ only for support vectors |
| RBF kernel | $\exp(-\gamma\lVert x - z\rVert^2)$; radius of influence $\sim 1/\sqrt\gamma$ |
| Random Fourier features | $\sqrt{2/D}\cos(Wx + b)$, $W\sim\mathcal N(0, 2\gamma)$ |
| GP mean | $k_*^\top(K + \sigma^2 I)^{-1}y$ (= kernel ridge) |
| GP variance | $k_{**} - k_*^\top(K+\sigma^2I)^{-1}k_*$ |
| Neighbourhood edge in $d$ dims | $r^{1/d}$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why does kNN degrade in high dimensions?</summary>

Distances concentrate: $(d_{\max}-d_{\min})/d_{\min} \to 0$, so the nearest and farthest points are almost equally far, and "neighbourhoods" span most of the range of every feature ($r^{1/d}$). The demo shows the ratio collapsing as $d$ grows.
</details>

<details>
<summary>2. Which points determine an SVM's boundary?</summary>

Only the support vectors, those on or inside the margin, which have $\alpha_i > 0$. By complementary slackness, every other point has $\alpha_i = 0$ and doesn't enter $w = \sum\alpha_i y_i x_i$.
</details>

<details>
<summary>3. Large vs small C.</summary>

A large $C$ penalizes violations heavily: a narrow margin, fewer support vectors, low bias, high variance. A small $C$ gives a wide margin with more violations allowed: more regularization, higher bias, lower variance.
</details>

<details>
<summary>4. What does RBF gamma control? What does overfitting at large gamma look like?</summary>

The radius of influence of each training point, about $1/\sqrt\gamma$. At very large $\gamma$, each point influences only itself. The boundary becomes islands around training points, with about 100% training accuracy and a poor test score (see the demo row with gamma = 1000).
</details>

<details>
<summary>5. How can a kernel compute an infinite-dimensional inner product cheaply?</summary>

The kernel is a closed-form function equal to $\phi(x)^\top\phi(z)$. For RBF, the Taylor expansion of the exponential implicitly contains polynomial features of every degree, yet evaluating $\exp(-\gamma\lVert x-z\rVert^2)$ costs $O(d)$.
</details>

<details>
<summary>6. What does a GP's predictive variance tell you, and where is it largest?</summary>

How uncertain the model is about $f(x_*)$, given where it has seen data. It is largest far from the training inputs (it reverts to the prior variance) and smallest near them. A point estimate can't express this.
</details>

<details>
<summary>7. RBF-SVM: 100% train, 60% test, unscaled features.</summary>

First scale the features (inside a pipeline). Then re-tune $C$ and $\gamma$ jointly on a log grid with CV. The current $\gamma$ is almost certainly too large relative to the feature scales, which produces memorization.
</details>

## Where this leads

Next: [CORE-10 notes](10-data-centric-ml.md). We've spent nine lessons improving *models*. CORE-10 flips the view: the fastest gains often come from improving the *data*:
fixing labels, handling imbalance, and choosing which examples to label next.
