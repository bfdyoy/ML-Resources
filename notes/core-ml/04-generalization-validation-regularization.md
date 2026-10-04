# CORE-04 notes: Generalization, Validation & Regularization

[← Lesson CORE-04](../../lessons/core-ml/04-generalization-validation-regularization.md) · [All notes](../README.md) · [← CORE-03 notes](03-classification-and-metrics.md) · Next: [CORE-05 notes →](05-trees-and-ensembles.md)

> **Reading time** ≈ 60 min. **You need:** [CORE-02 notes](02-linear-models-gradient-descent.md), [MATH-01 notes](../math/01-linear-algebra.md) §C4 (SVD), [MATH-03 notes](../math/03-probability-statistics.md) §A3 and §B4 (variance, MAP).

---

## Where we are

We can fit models (CORE-02, CORE-03), and CORE-01 told us that training error lies. This lesson answers three questions:
**why** models fail to generalize (bias and variance), **how** to measure generalization reliably (cross-validation), and **how** to control it on purpose (regularization).

---

## 1. The bias-variance decomposition

### 1.1 Set-up

Assume $y = f(x) + \varepsilon$, with noise $\mathbb{E}[\varepsilon] = 0$ and $\operatorname{Var}(\varepsilon) = \sigma^2$. Our fitted model $\hat f$ depends on the
**random training set** $\mathcal{D}$: draw a different sample and you get a different $\hat f$. Fix a test point $x$ and ask for the expected squared error, averaged
over both the noise and the training sets.

### 1.2 Derivation (three lines)

Write $\bar f(x) = \mathbb{E}_\mathcal{D}[\hat f(x)]$ for the *average* model. Then:

```math
\begin{aligned}
\mathbb{E}\big[(y - \hat f)^2\big]
&= \mathbb{E}\big[(f + \varepsilon - \hat f)^2\big]
 = \sigma^2 + \mathbb{E}\big[(f - \hat f)^2\big] \qquad (\varepsilon \text{ is independent of } \hat f \text{ and has mean } 0)\\
&= \sigma^2 + \mathbb{E}\big[(f - \bar f + \bar f - \hat f)^2\big]
 = \sigma^2 + (f - \bar f)^2 + \mathbb{E}\big[(\hat f - \bar f)^2\big] \qquad (\text{the cross term has mean } 0)\\
&= \underbrace{\sigma^2}_{\text{irreducible noise}} + \underbrace{\big(f(x) - \bar f(x)\big)^2}_{\text{bias}^2} + \underbrace{\operatorname{Var}_\mathcal{D}\big(\hat f(x)\big)}_{\text{variance}} .
\end{aligned}
```

### 1.3 Reading it

- **Bias:** how wrong the *average* model is. It's large when the model family can't represent $f$ (a line fitted to a curve).
- **Variance:** how much the model *jumps around* between training sets. It's large when the model is flexible enough to fit noise.
- **Noise:** nothing can remove it.

As flexibility grows, bias falls and variance rises, which gives the classic **U-shaped test-error curve**. Training error, meanwhile, keeps falling.
The code below estimates all three terms by refitting polynomials on 500 fresh training sets.

---

## 2. Cross-validation

### 2.1 Why not one validation split?

With a small data set, a single validation split is a **noisy** estimate (MATH-03 §D1). It also wastes data you'd like to train on.

### 2.2 k-fold CV

Split the data into $k$ folds. For each fold, train on the other $k-1$ and evaluate on it. Average the $k$ scores. Every example is used for validation exactly once.

- $k = 5$ or $10$ is the usual choice. Leave-one-out ($k = N$) has low bias but is expensive, and its estimate can have high variance.
- The **spread** of the fold scores is a rough error bar. (The folds share training data, so it underestimates the true uncertainty. Treat it as a guide.)

### 2.3 Variants: match the split to reality

| Situation | Use | Why |
|---|---|---|
| Imbalanced classes | `StratifiedKFold` | Keeps class ratios in every fold |
| Several rows per entity (patient, user) | `GroupKFold` | Otherwise the model memorizes the *entity*, and validation measures "seen patient" accuracy |
| Time-ordered data | `TimeSeriesSplit` (expanding window, optionally with a gap) | Training must come before validation, as in deployment |

The grouped case is demonstrated in code below: identical models, very different CV scores, and only the grouped score is honest.

### 2.4 Nested CV: honest estimates when you also tune

If you tune hyperparameters with CV and then report **that same CV score**, you've picked the configuration that was luckiest on those folds. That's the selection bias from CORE-01 §2.3.
**Nested CV** fixes this with two loops:

- an **inner** loop chooses hyperparameters using only the outer-training data;
- an **outer** loop scores the *whole tuning procedure* on data the inner loop never saw.

The outer score estimates "how well does *my procedure* (tuning included) generalize?" In scikit-learn, it's `cross_val_score(GridSearchCV(...), X, y)`.

---

## 3. Regularization: controlling variance on purpose

Add a penalty on the size of the weights:

```math
\textbf{Ridge: } \min_w \lVert y - Xw\rVert^2 + \lambda\lVert w\rVert_2^2 \qquad
\textbf{Lasso: } \min_w \lVert y - Xw\rVert^2 + \lambda\lVert w\rVert_1 .
```

Remember from MATH-03 §B4 that these are MAP estimates with Gaussian and Laplace priors. **Always standardize the features first**, because the penalty treats all coefficients
equally, so their units must be comparable. Don't penalize the intercept.

### 3.1 Ridge: closed form and what it shrinks

Set the gradient to zero: $-2X^\top(y - Xw) + 2\lambda w = 0$, which gives

```math
\hat w_\text{ridge} = (X^\top X + \lambda I)^{-1}X^\top y .
```

Adding $\lambda I$ raises every eigenvalue of $X^\top X$ by $\lambda$, so the matrix is always invertible and well conditioned. That fixes collinearity (CORE-02 §6).
With the SVD $X = U\Sigma V^\top$, the solution becomes:

```math
\hat w_\text{ridge} = \sum_i \underbrace{\frac{\sigma_i^2}{\sigma_i^2 + \lambda}}_{\text{shrink factor}}\;\frac{u_i^\top y}{\sigma_i}\, v_i .
```

OLS is the case $\lambda = 0$, where every shrink factor is 1. Ridge **shrinks most along the directions with small singular values**, the directions where the data
has little spread and the OLS estimate is mostly noise. That's variance reduction, targeted exactly where the variance lives.

### 3.2 Lasso: why it produces exact zeros

**The 1-D calculation.** With one standardized feature (an orthonormal design), the problem becomes $\min_w (w - z)^2 + \lambda\lvert w\rvert$, where $z$ is the OLS estimate.
For $w > 0$, the derivative $2(w - z) + \lambda = 0$ gives $w = z - \lambda/2$, valid only if $z > \lambda/2$. Symmetrically for $w < 0$. Otherwise the minimum sits at the kink, $w = 0$:

```math
\hat w_\text{lasso} = \operatorname{sign}(z)\,\max\big(\lvert z\rvert - \tfrac{\lambda}{2},\, 0\big) \qquad\text{vs}\qquad \hat w_\text{ridge} = \frac{z}{1 + \lambda}.
```

This is **soft-thresholding**. Small coefficients are set **exactly** to 0, and large ones are shifted toward 0 by a constant. Ridge only rescales: it never reaches zero.

**The geometric picture.** Equivalently, minimize the RSS subject to $\lVert w\rVert_1 \le s$. The RSS contours are ellipses. The L1 ball is a diamond with **corners on the axes**,
and expanding ellipses usually touch the diamond first at a corner, where some coordinates are 0. The L2 ball is round, with no corners, so the contact point is generic and not sparse.

**Elastic net** mixes the two penalties. It keeps lasso's sparsity, but behaves better with groups of correlated features: lasso tends to pick one of them arbitrarily.

### 3.3 Choosing λ

$\lambda$ is a hyperparameter, so choose it by CV (`RidgeCV`, `LassoCV`). Plotting validation error against $\log\lambda$ gives the bias-variance U-curve again: small $\lambda$ means
low bias and high variance, and large $\lambda$ means the reverse.

---

## 4. Double descent: when bigger stops meaning worse

The U-curve isn't the whole story. Push a model past the **interpolation threshold** (where it has just enough parameters to fit the training data exactly) and test error can
**fall again**:

- **At the threshold** there is essentially *one* way to fit the data perfectly. That fit is wild, because it has to bend through every noisy point, so variance explodes.
- **Beyond it** there are *infinitely many* perfect fits. GD started from zero (or a pseudo-inverse) picks the **minimum-norm** one, the smoothest interpolator, and
  that implicit regularization makes it generalize better as parameters are added.

This is why huge neural networks don't overfit the way the U-curve predicts. The demo below reproduces the peak with random-feature regression. ([MLU-Explain: Double Descent](https://mlu-explain.github.io/double-descent/) animates it.)

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor
rng = np.random.default_rng(0)

# --- Bias^2 and variance by simulation (polynomials fitted to a sine) ------
f = lambda x: np.sin(2 * np.pi * x)
x_test, sigma, n = np.linspace(0.05, 0.95, 50), 0.3, 30
for deg in [1, 3, 9]:
    preds = []
    for _ in range(500):
        x = rng.random(n); y = f(x) + rng.normal(0, sigma, n)
        preds.append(np.polyval(np.polyfit(x, y, deg), x_test))
    preds = np.array(preds)
    bias2 = np.mean((preds.mean(0) - f(x_test)) ** 2)
    var = np.mean(preds.var(0))
    print(f"degree {deg}: bias^2={bias2:.3f}  variance={var:.3f}  expected test MSE≈{bias2+var+sigma**2:.3f}")
```

```python
# --- Ridge = SVD shrinkage; lasso = soft-thresholding ------------------------
X = rng.normal(size=(100, 5)); y = X @ np.array([3, 0, 0, 1.5, 0.2]) + rng.normal(size=100)
lam = 10.0
w_closed = np.linalg.solve(X.T @ X + lam * np.eye(5), X.T @ y)
U, s, Vt = np.linalg.svd(X, full_matrices=False)
w_svd = Vt.T @ ((s / (s**2 + lam)) * (U.T @ y))
w_sk = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
print("ridge closed form :", w_closed.round(4))
print("ridge via SVD     :", w_svd.round(4))
print("sklearn Ridge     :", w_sk.round(4))

for alpha in [0.01, 0.1, 0.5]:
    nz_l = np.sum(np.abs(Lasso(alpha=alpha).fit(X, y).coef_) > 1e-8)
    nz_r = np.sum(np.abs(Ridge(alpha=alpha * 100).fit(X, y).coef_) > 1e-8)
    print(f"alpha={alpha}: lasso non-zero coefs = {nz_l}/5, ridge non-zero coefs = {nz_r}/5")

# --- Grouped data: plain KFold vs GroupKFold ------------------------------------
patients = np.repeat(np.arange(60), 5)                  # 60 patients, 5 rows each
fingerprint = rng.normal(size=(60, 3))[patients]          # per-patient traits (e.g. age, height, baseline lab)
Xg = fingerprint + 0.05 * rng.normal(size=(300, 3))      # rows of one patient look almost identical
yg = rng.normal(size=60)[patients] + 0.3 * rng.normal(size=300)   # outcome: patient-specific, NOT predictable from traits
model = RandomForestRegressor(n_estimators=100, random_state=0)
plain = cross_val_score(model, Xg, yg, cv=KFold(5, shuffle=True, random_state=0)).mean()
group = cross_val_score(model, Xg, yg, cv=GroupKFold(5), groups=patients).mean()
print(f"R^2 with plain KFold: {plain:.2f}   with GroupKFold: {group:.2f}")
```

```python
# --- Double descent with random ReLU features, minimum-norm least squares ----
n_train, d = 40, 5
Xtr, Xte = rng.normal(size=(n_train, d)), rng.normal(size=(1000, d))
w_true = rng.normal(size=d)
ytr = Xtr @ w_true + 0.5 * rng.normal(size=n_train); yte = Xte @ w_true
for p in [5, 20, 35, 40, 45, 80, 400, 2000]:
    errs = []
    for s in range(20):
        W = np.random.default_rng(s).normal(size=(d, p)) / np.sqrt(d)
        Ftr, Fte = np.maximum(Xtr @ W, 0), np.maximum(Xte @ W, 0)
        beta = np.linalg.pinv(Ftr) @ ytr                   # min-norm solution when p > n
        errs.append(np.mean((Fte @ beta - yte) ** 2))
    print(f"features p={p:5d}  median test MSE={np.median(errs):8.2f}" + ("   <- interpolation threshold (p = n)" if p == n_train else ""))
```

---

## Pitfalls & misconceptions

- **Regularizing unscaled features.** The penalty then hits features differently depending on their units.
- **Reporting the tuned CV score as the final estimate.** It's optimistic. Use nested CV or a held-out test set.
- **k-fold on grouped or temporal data.** It produces leaky, inflated scores.
- **"Lasso picks the true features."** With correlated features its choice is unstable. Check it with bootstrap resampling.
- **"More parameters always means more overfitting."** Not past the interpolation threshold, with implicit or explicit regularization.

## Cheat sheet

| Idea | Formula / rule |
|---|---|
| Decomposition | test MSE = $\sigma^2$ + bias² + variance |
| Ridge | $(X^\top X + \lambda I)^{-1}X^\top y$; shrink factor $\sigma_i^2/(\sigma_i^2+\lambda)$ |
| Lasso (orthonormal design) | soft-threshold: $\operatorname{sign}(z)\max(\lvert z\rvert - \lambda/2, 0)$ |
| CV choice | stratified / grouped / time-series, to mimic deployment |
| Nested CV | inner loop tunes, outer loop scores the procedure |
| Golden rule | every data-driven decision happens inside the CV loop |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Training and test error against complexity.</summary>

Training error decreases monotonically. Test error is U-shaped: on the left (simple models) it's high because of bias, which is under-fitting, and both errors are high and close together. On the right (complex models) it's high because of variance, which is over-fitting, with a big gap between them. The minimum of the test-error curve is the sweet spot.
</details>

<details>
<summary>2. Why does lasso give exact zeros and ridge doesn't?</summary>

The L1 constraint region is a diamond with corners on the axes, and the elliptical RSS contours tend to touch it at a corner, where coordinates are 0. Algebraically, soft-thresholding sets $\lvert z\rvert \le \lambda/2$ to exactly 0, while ridge only rescales, $z/(1+\lambda)$.
</details>

<details>
<summary>3. Five measurements per patient: why is plain k-fold wrong?</summary>

The same patient lands in both training and validation folds. The model learns patient-specific patterns and looks good on "new" rows from known patients, but deployment means new patients. Use `GroupKFold` with patient IDs. The demo shows a large gap between the two scores.
</details>

<details>
<summary>4. Why TimeSeriesSplit for forecasting?</summary>

A random split puts future data in training, so the model peeks at information it won't have. A time-ordered split (train on the past, validate on the future, ideally with a gap) mimics deployment.
</details>

<details>
<summary>5. What does nested CV solve?</summary>

The optimistic bias of reporting the score of the configuration selected on the same folds. The outer loop evaluates the entire tuning procedure on data it never touched.
</details>

<details>
<summary>6. 400 configs, best CV much better than fresh data.</summary>

Selection bias, or overfitting to the validation folds. With many configs, the best one is partly lucky on those folds. Fixes: nested CV or an untouched test set, fewer and more sensible configurations, repeated CV, and treating the "winner" as within noise of the runners-up.
</details>

## Where this leads

Next: [CORE-05 notes](05-trees-and-ensembles.md). Trees are a model family with very low bias and very high variance. Seeing how ensembles tame that variance
(bagging) or chip away at the bias (boosting) is the bias-variance decomposition put to work.
