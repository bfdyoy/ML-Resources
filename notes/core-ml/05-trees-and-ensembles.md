# CORE-05 notes: Trees, Random Forests & Gradient Boosting

[← Lesson CORE-05](../../lessons/core-ml/05-trees-and-ensembles.md) · [All notes](../README.md) · [← CORE-04 notes](04-generalization-validation-regularization.md) · Next: [CORE-06 notes →](06-unsupervised-learning.md)

> **Reading time** ≈ 60 min. **You need:** [CORE-04 notes](04-generalization-validation-regularization.md) §1 (bias and variance) and [MATH-03 notes](../math/03-probability-statistics.md) §A3 (variance of an average).

---

## Where we are

Linear models are high-bias: they can only draw straight boundaries unless you hand-craft features. Trees are the opposite extreme. They can carve out any shape, which makes them low-bias and very high-variance.
Two ensemble ideas then fix the variance in different ways: **bagging** (average many trees) and **boosting** (add small trees one at a time to correct mistakes).
Gradient-boosted trees are still the strongest default for tabular data (EL-10 explains why).

---

## 1. A decision tree

### 1.1 What it is

A tree splits the feature space into axis-aligned boxes with questions like "is $x_3 \le 2.5$?". It predicts a constant in each box: the mean target (regression) or the class proportions (classification).

### 1.2 How it chooses a split: greedy impurity reduction

At a node, try every feature and every threshold, and pick the split that makes the children **purest**. For classification with class proportions $p_k$ in a node:

```math
\text{Gini}(\text{node}) = 1 - \sum_k p_k^2, \qquad \text{Entropy}(\text{node}) = -\sum_k p_k \log_2 p_k .
```

Both are 0 for a pure node and largest at 50/50. A split's quality is the **impurity decrease**: the parent's impurity minus the size-weighted average of the children's impurities.
For regression, impurity is the variance (equivalently the RSS), so the split minimizes $\text{RSS}_\text{left} + \text{RSS}_\text{right}$.

*Worked example.* A node has 10 examples, 5 of each class, so Gini = $1 - (0.5^2 + 0.5^2) = 0.5$. Two candidate splits:

| Split | Left child | Right child | Weighted Gini | Decrease |
|---|---|---|---|---|
| A | 4+ / 1− (Gini 0.32) | 1+ / 4− (Gini 0.32) | 0.32 | **0.18** |
| B | 5+ / 3− (Gini 0.469) | 0+ / 2− (Gini 0) | 0.8 × 0.469 = 0.375 | 0.125 |

Split A wins. Here's the Gini of 4/1: $1 - (0.8^2 + 0.2^2) = 0.32$. The code below checks it.

### 1.3 Why deep trees overfit, and what to do

Keep splitting and every leaf ends up pure, possibly holding one training example. Training error is zero, but each leaf's prediction is based on one or two noisy points: **high variance**.
A different sample would produce a very different tree, because an early split choice changes everything below it.

Controls: `max_depth`, `min_samples_leaf`, and **cost-complexity pruning**, which grows a big tree and then cuts back branches by minimizing $\text{RSS} + \alpha\lvert T\rvert$ (where $\lvert T\rvert$ is the number of leaves).
$\alpha$ plays the same role as $\lambda$ in ridge: it trades fit for simplicity, and you choose it by CV.

### 1.4 Two consequences of "constant in each box"

- **Insensitive to monotonic transforms.** A split only asks "which side of the threshold?". Taking the log of a feature preserves the order, so it preserves the possible partitions. You don't need scaling, and outliers in the *features* matter little.
- **No extrapolation.** The prediction is always the average of some training targets. Beyond the training range, the tree predicts the value of the edge leaf, a flat line. For trends (prices rising over time), add a linear component or model the differences.

---

## 2. Bagging and random forests: averaging away variance

### 2.1 The variance of an average of correlated things

Average $B$ models, each with variance $\sigma^2$ and pairwise correlation $\rho$:

```math
\operatorname{Var}\Big(\frac1B\sum_{b=1}^B \hat f_b\Big) = \frac{1}{B^2}\Big(B\sigma^2 + B(B-1)\rho\sigma^2\Big) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 .
```

Read it carefully. Adding more trees ($B \to \infty$) kills the second term, but **the first term, $\rho\sigma^2$, stays**. Averaging can't remove variance the models share.
So, to get more out of averaging, **make the models less correlated**.

### 2.2 Bagging

**Bootstrap aggregating:** train each tree on a bootstrap resample (drawn with replacement, same size), and average the predictions (regression) or take a vote (classification). Each tree is deep (low bias), and averaging reduces the variance.

### 2.3 Random forests: decorrelate the trees

Bagged trees are still highly correlated: if one feature is very strong, every tree splits on it first. A **random forest** considers only a random subset of `max_features` features at
**each split** (typically $\sqrt{d}$ for classification and $d/3$ for regression). Strong features can't dominate every tree, so $\rho$ drops, and by the formula above, so does the ensemble variance.
Each tree gets a little worse; the forest gets better.

### 2.4 Out-of-bag (OOB) error: free validation

The chance that a given example is *not* drawn into a bootstrap sample of size $N$ is

```math
\Big(1 - \frac1N\Big)^N \;\xrightarrow{N\to\infty}\; e^{-1} \approx 0.368 .
```

So each tree never sees about 37% of the data. To get each example's OOB prediction, average only the trees that didn't train on it. The resulting error is a CV-like estimate at no extra cost.

---

## 3. Gradient boosting: gradient descent in function space

### 3.1 The idea

Bagging builds big trees **in parallel** and averages them. Boosting builds **small** trees **in sequence**, each one fixing what the ensemble so far gets wrong:

```math
F_0(x) = \text{const}, \qquad F_m(x) = F_{m-1}(x) + \nu\, h_m(x),
```

where $h_m$ is a shallow tree and $\nu$ is the **learning rate** (shrinkage), typically between 0.01 and 0.1.

### 3.2 What each tree fits: the negative gradient

Treat the vector of predictions $F = (F(x_1), \dots, F(x_N))$ as the thing you optimize. Gradient descent on the loss $\sum_i \ell(y_i, F_i)$ would step each prediction by
$-\partial \ell/\partial F_i$. We can't step individual training predictions directly (we need a function that also works on new $x$), so **fit a tree to the negative gradients** and step along that tree.
These targets are called **pseudo-residuals**:

```math
r_{im} = -\frac{\partial \ell(y_i, F)}{\partial F}\Big|_{F = F_{m-1}(x_i)} .
```

- Squared error, $\ell = \tfrac12(y - F)^2$: $r = y - F$. These are **literally the residuals**, so "fit the residuals" is GD on MSE.
- Log loss with $F$ as the log-odds: $r = y - \sigma(F) = y - p$, the same "label minus probability" as in logistic regression (CORE-03 §1.4).
- Absolute error: $r = \operatorname{sign}(y - F)$, which is robust to outliers.

**So gradient boosting is gradient descent, where each "step" is a tree.** It works for any differentiable loss.

### 3.3 Why a small learning rate plus many trees works better

A full step ($\nu = 1$) makes each tree fit the current residuals as hard as it can, *including their noise*, and later trees build on that noise. Small steps take many gentle, slightly different
corrections. That averages out noise across trees, a regularization effect much like early stopping in GD. The cost is more trees (more compute), and the gain is almost always better validation error.
Standard practice: fix $\nu$ small, then choose the number of trees by **early stopping** on a validation set.

### 3.4 XGBoost/LightGBM in one formula

Modern libraries use a **second-order** expansion of the loss. They take gradients $g_i$ and Hessians $h_i$ at the current predictions, and add an L2 penalty $\lambda$ on the leaf values.
For a leaf containing a set of examples $I$, the optimal value and the score of that leaf are:

```math
w^* = -\frac{\sum_{i\in I} g_i}{\sum_{i\in I} h_i + \lambda}, \qquad \text{score} = -\frac12\frac{\big(\sum_{i\in I} g_i\big)^2}{\sum_{i\in I} h_i + \lambda} .
```

A split's **gain** is the children's score improvement over the parent, minus a penalty $\gamma$ per extra leaf. Each leaf value is one Newton step, with ridge shrinkage built in.
Add **row subsampling** and **column subsampling** (borrowed from random forests), plus histogram-based split finding, and you have the modern GBM.

### 3.5 The hyperparameters that matter (in order)

1. `learning_rate` together with `n_estimators` (set by early stopping);
2. tree size: `max_depth` or `num_leaves`, and `min_child_samples`;
3. `subsample` and `colsample_bytree` (each around 0.7–0.9);
4. regularization: `reg_lambda`, `min_split_gain`.

| | Random forest | Gradient boosting |
|---|---|---|
| Trees | Deep, independent | Shallow, sequential |
| Fixes | Variance (by averaging) | Bias (by adding corrections) |
| More trees… | Never hurts (it only plateaus) | Eventually overfits, so use early stopping |
| Tuning effort | Low | Moderate, with a higher ceiling |

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
rng = np.random.default_rng(0)

gini = lambda counts: 1 - np.sum((np.array(counts) / np.sum(counts)) ** 2)
def weighted(children):
    n = sum(sum(c) for c in children)
    return sum(sum(c) / n * gini(c) for c in children)
print("parent Gini:", gini([5, 5]))
print("split A weighted Gini:", round(weighted([[4, 1], [1, 4]]), 3), " split B:", round(weighted([[5, 3], [0, 2]]), 3))

# Variance of an average of correlated predictors: rho*s2 + (1-rho)*s2/B
s2, rho, B = 1.0, 0.3, 50
cov = s2 * (rho * np.ones((B, B)) + (1 - rho) * np.eye(B))
draws = rng.multivariate_normal(np.zeros(B), cov, size=20000).mean(1)
print(f"simulated var {draws.var():.3f}  formula {rho*s2 + (1-rho)*s2/B:.3f}")

# OOB fraction
N = 1000
print("fraction never drawn in a bootstrap:", np.mean([len(set(range(N)) - set(rng.integers(0, N, N))) / N for _ in range(200)]).round(3), "  1/e =", round(np.exp(-1), 3))
```

```python
# Gradient boosting from scratch (squared error): fit trees to residuals
X = rng.uniform(-3, 3, size=(500, 1)); y = np.sin(X[:, 0]) + 0.3 * rng.normal(size=500)
nu, M = 0.1, 100
F = np.full(len(y), y.mean()); trees = []
for m in range(M):
    residual = y - F                                   # negative gradient of 1/2 (y - F)^2
    t = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X, residual)
    F += nu * t.predict(X); trees.append(t)
sk = GradientBoostingRegressor(n_estimators=M, learning_rate=nu, max_depth=2, random_state=0).fit(X, y)
print("our training MSE:", np.mean((y - F) ** 2).round(4), "  sklearn's:", np.mean((y - sk.predict(X)) ** 2).round(4))

# Trees can't extrapolate: train on x in [0, 10] with y = 2x, predict x = 20
Xl = rng.uniform(0, 10, size=(300, 1)); yl = 2 * Xl[:, 0]
rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(Xl, yl)
print("forest prediction at x=20:", rf.predict([[20.0]]).round(1), " (true 40; a linear model would get it)")
```

---

## Pitfalls & misconceptions

- **Impurity-based feature importance is biased** toward high-cardinality and continuous features (it can even rank a random ID column highly). Use permutation importance (CORE-08).
- **Treating `n_estimators` in boosting like in a forest.** In boosting, more trees eventually overfit. Use early stopping.
- **Expecting trees to extrapolate trends.** They predict flat beyond the training range.
- **One-hot encoding very high-cardinality categoricals for trees.** It produces sparse, weak splits. Use native categorical support (LightGBM, CatBoost) or careful target encoding (CORE-07).

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Gini | $1 - \sum_k p_k^2$ |
| Split gain | parent impurity − weighted child impurity |
| Variance of an average | $\rho\sigma^2 + (1-\rho)\sigma^2/B$ |
| OOB fraction | $\approx 1/e \approx 36.8\%$ |
| Boosting update | $F_m = F_{m-1} + \nu h_m$, with $h_m$ fitted to $-\partial\ell/\partial F$ |
| MSE / log-loss pseudo-residual | $y - F$ / $y - p$ |
| XGBoost leaf value | $-G/(H + \lambda)$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why is a deep tree high-variance, and how does averaging fix it?</summary>

Its leaves are fitted to very few points, and early splits cascade, so small changes in the data change the tree a lot. Averaging $B$ trees cuts the variance to $\rho\sigma^2 + (1-\rho)\sigma^2/B$, keeping the low bias.
</details>

<details>
<summary>2. What does max_features add on top of bagging?</summary>

It decorrelates the trees: strong features can't be chosen first in every tree. A lower $\rho$ means lower ensemble variance, because $\rho\sigma^2$ is the part averaging can't remove.
</details>

<details>
<summary>3. What does each boosting tree fit, and why a small learning rate?</summary>

The negative gradient of the loss with respect to the current predictions: the residuals for MSE, $y-p$ for log loss. A small $\nu$ with many trees takes gentle steps that don't lock in noise, which acts as a regularizer. Choose the number of trees by early stopping.
</details>

<details>
<summary>4. Why are trees insensitive to monotonic transformations?</summary>

Splits depend only on the order of the values. A monotone transform keeps the order, so it gives the same set of possible partitions, and the same tree.
</details>

<details>
<summary>5. Why can't trees extrapolate?</summary>

Every prediction is an average of training targets in some leaf. Outside the training range, you land in an edge leaf, so the prediction is flat at the edge value.
</details>

<details>
<summary>6. Validation loss gets worse after 300 rounds while training loss keeps falling.</summary>

That's overfitting from too many rounds. Use early stopping (keep the round-300 model), lower the learning rate, and add regularization: smaller trees, `min_child_samples`, subsampling, `reg_lambda`.
</details>

## Where this leads

Next: [CORE-06 notes](06-unsupervised-learning.md). So far every example came with a label. Next, the labels disappear and we look for structure (directions, clusters)
in $X$ alone. The linear algebra from MATH-01 Block C finally pays off.
