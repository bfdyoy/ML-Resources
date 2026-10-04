# CORE-02 notes: Linear Models & Gradient Descent

[← Lesson CORE-02](../../lessons/core-ml/02-linear-models-gradient-descent.md) · [All notes](../README.md) · [← CORE-01 notes](01-ml-workflow-end-to-end.md) · Next: [CORE-03 notes →](03-classification-and-metrics.md)

> **Reading time** ≈ 50 min. **You need:** [MATH-01 notes](../math/01-linear-algebra.md) §B4 (least squares as projection) and [MATH-02 notes](../math/02-calculus-optimization.md) §A4, §B1–B3 (gradients, GD, conditioning).

---

## Where we are

CORE-01 gave us the frame: minimize the loss on training data, measure honestly on held-out data. Now we fill the frame with the simplest real model.
Linear regression is worth learning thoroughly because **every idea in it returns later**: a neural network's last layer *is* a linear model, and its training loop *is*
the mini-batch gradient descent you'll write here.

---

## 1. The model

```math
\hat y = w_0 + w_1 x_1 + \dots + w_d x_d = w^\top x \quad(\text{with a constant } x_0 = 1 \text{ absorbed into } x).
```

For a whole data set, $\hat y = Xw$. "Linear" means linear **in the parameters $w$**, not in the inputs. $\hat y = w_0 + w_1 x + w_2 x^2$ is still a linear model.
That's why polynomial regression (§5) is "just" linear regression on extra columns.

### Reading a coefficient

In **multiple** regression, $w_j$ is the expected change in $y$ when $x_j$ increases by 1 **and every other feature is held fixed**. In **simple** regression (one feature),
$w_1$ is the change in $y$ per unit of $x_1$ *with everything else free to vary along with it*. The two numbers can differ a lot, even in sign.

*Example.* Ice-cream sales and drownings are positively related in simple regression. Add temperature as a feature and the ice-cream coefficient drops to about zero:
heat drives both. The code in §6 shows a variable whose simple-regression slope is about +1.4 but whose multiple-regression slope is about −1.0. The difference is called **omitted-variable bias**.

---

## 2. Fitting: three routes to one answer

The loss is the **mean squared error**:

```math
L(w) = \frac{1}{N}\lVert Xw - y\rVert^2 = \frac1N\sum_{i=1}^N (w^\top x_i - y_i)^2 .
```

1. **Geometry** (MATH-01 §B4): $X\hat w$ is the projection of $y$ onto the column space of $X$, so the residual is perpendicular to every column, and $X^\top X\hat w = X^\top y$.
2. **Calculus** (MATH-02 §A4): $\nabla L = \frac2N X^\top(Xw - y) = 0$ gives the same normal equations.
3. **Probability** (MATH-03 §B2): if $y = w^\top x + \varepsilon$ with Gaussian noise, maximum likelihood = least squares.

So $\hat w = (X^\top X)^{-1}X^\top y$ **when $X^\top X$ is invertible**.

### How good is the fit? $R^2$

```math
R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}.
```

TSS is the error of the dumbest baseline, "always predict the mean" (CORE-01 §4). $R^2$ is the **fraction of that baseline error the model removes**.
$R^2 = 0.7$ means the model explains 70% of the variance around the mean. On *test* data, $R^2$ can be negative: the model is worse than predicting the mean.

### How sure are we about each coefficient? Standard errors

If the noise has variance $\sigma^2$, then

```math
\operatorname{Var}(\hat w) = \sigma^2 (X^\top X)^{-1}, \qquad \mathrm{SE}(\hat w_j) = \hat\sigma\sqrt{\big[(X^\top X)^{-1}\big]_{jj}} .
```

$\hat w_j / \mathrm{SE}(\hat w_j)$ is the **t-statistic** that `statsmodels` prints. A rough reading: $\lvert t\rvert > 2$ means "unlikely to be zero". Note the inverse of $X^\top X$ in the formula:
when features are nearly collinear, $X^\top X$ is nearly singular, its inverse explodes, and so do the standard errors (§6).

---

## 3. Why bother with gradient descent?

The closed form costs about $O(Nd^2 + d^3)$ and needs all of $X$ in memory. It is also **numerically fragile** when $X^\top X$ is badly conditioned
(forming $X^\top X$ *squares* the condition number). Gradient descent:

- costs $O(Nd)$ per pass and works on mini-batches streamed from disk;
- generalizes to models with **no** closed form (logistic regression, every neural net);
- can be stopped early, which acts as regularization.

The trade: you now have a learning rate to choose and convergence to watch.

### The update, three flavours

```math
w \leftarrow w - \eta\, \frac{2}{B} X_B^\top (X_B w - y_B)
```

| Variant | $B$ (batch size) | Behaviour |
|---|---|---|
| Batch GD | $N$ | Smooth, exact gradient; one step per pass over the data |
| Stochastic GD | 1 | Very noisy; many cheap steps; needs a decaying $\eta$ |
| Mini-batch | 32–1024 | The standard: vectorized and reasonably smooth |

---

## 4. Feature scaling: why it matters for GD (and not for OLS)

The Hessian of the MSE is $H = \frac2N X^\top X$. Its eigenvalues set the shape of the loss bowl (MATH-02 §B3).
If one feature is in metres (values around 1) and another in millimetres (values around 1000), the curvature along the second is roughly $10^6$ times larger. The bowl is then a
**long, thin valley**, with condition number $\kappa \sim 10^6$. The stable learning rate is capped by the steep direction ($\eta < 2/\lambda_{\max}$), so progress
along the flat direction is glacial: GD **zig-zags across the valley** and barely moves along it.

**Standardizing** each feature (subtract the mean, divide by the standard deviation) makes the bowl much rounder. Then one learning rate suits all directions.
OLS doesn't care, because the closed form solves the system exactly in any units (the coefficients simply rescale). GD cares a lot.

---

## 5. Polynomial features, under- and over-fitting, learning curves

Add the columns $x^2, x^3, \dots$ (or interactions $x_1 x_2$) and a linear model can fit curves. Degree is a **complexity knob**:

- **Too low** → **high bias** (under-fitting): training and validation error are both high, *and close together*. More data won't help. A richer model will.
- **Too high** → **high variance** (over-fitting): training error is low, validation error is high, and there's a big gap. More data or regularization helps.

A **learning curve** plots training and validation error against training-set size. Reading the gap and the level is the fastest diagnosis in ML.
(CORE-04 makes this precise with the bias-variance decomposition.)

---

## 6. Collinearity

When two features are almost copies of each other, many combinations of their weights fit equally well: $w_1 x_1 + w_2 x_2$ barely changes along the line
$w_1 + w_2 = \text{const}$. So:

- the **coefficients** become unstable (huge standard errors, flipping signs between resamples);
- the **predictions** stay fine, as long as new data has the same correlation.

Fixes: drop or combine features, or use **ridge** regression, which adds $\lambda I$ to $X^\top X$, lifts every eigenvalue, and makes the solution unique and stable (CORE-04).

```python
import numpy as np
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(0)

# --- Mini-batch GD from scratch vs sklearn ---------------------------------
N, d = 1000, 3
X = rng.normal(size=(N, d))
true_w, true_b = np.array([2.0, -1.0, 0.5]), 3.0
y = X @ true_w + true_b + rng.normal(scale=0.5, size=N)

def fit_gd(X, y, lr=0.05, epochs=200, batch_size=32, seed=0):
    r = np.random.default_rng(seed)
    Xb = np.c_[np.ones(len(X)), X]                 # absorb the intercept
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        idx = r.permutation(len(Xb))
        for s in range(0, len(Xb), batch_size):
            B = idx[s:s + batch_size]
            grad = 2 / len(B) * Xb[B].T @ (Xb[B] @ w - y[B])
            w -= lr * grad
        lr *= 0.98                                  # gentle decay so SGD noise settles
    return w

w_gd = fit_gd(X, y)
sk = LinearRegression().fit(X, y)
print("GD      :", np.round(w_gd, 4))
print("sklearn :", np.round(np.r_[sk.intercept_, sk.coef_], 4))
# Mini-batch noise leaves you within ~1e-3. For a 4-decimal match, use batch_size=len(X) and more epochs without decay.

# --- Scaling vs number of full-batch GD steps --------------------------------
def steps_to_converge(X, y, tol=1e-6, max_steps=200_000):
    Xb = np.c_[np.ones(len(X)), X]
    H = 2 / len(Xb) * Xb.T @ Xb
    lr = 1.0 / np.linalg.eigvalsh(H).max()            # safe step: below 2 / lambda_max
    w_star = np.linalg.lstsq(Xb, y, rcond=None)[0]
    w = np.zeros(Xb.shape[1])
    for t in range(max_steps):
        w -= lr * 2 / len(Xb) * Xb.T @ (Xb @ w - y)
        if np.linalg.norm(w - w_star) < tol:
            return t + 1, np.linalg.cond(H)
    return f">{max_steps}", np.linalg.cond(H)

X_unscaled = X * np.array([1.0, 30.0, 0.05])        # same information, different units
X_scaled = (X_unscaled - X_unscaled.mean(0)) / X_unscaled.std(0)
for name, Z in [("unscaled", X_unscaled), ("standardized", X_scaled)]:
    steps, cond = steps_to_converge(Z, y)
    print(f"{name:13s} condition number {cond:12.1f}  GD steps: {steps}")
```

```python
# --- Omitted-variable bias: simple vs multiple regression ----------------
n = 5000
temp = rng.normal(size=n)
x1 = 0.8 * temp + 0.6 * rng.normal(size=n)          # x1 is correlated with temperature
yy = -1.0 * x1 + 3.0 * temp + rng.normal(size=n)     # x1's true effect is NEGATIVE
simple = LinearRegression().fit(x1[:, None], yy).coef_[0]
multi = LinearRegression().fit(np.c_[x1, temp], yy).coef_[0]
print(f"slope of x1 alone: {simple:+.2f}   slope of x1 holding temp fixed: {multi:+.2f}")

# --- Collinearity: coefficients wobble, predictions don't ------------------
coefs, preds = [], []
x_new = np.array([[1.0, 1.0]])
for s in range(200):
    r = np.random.default_rng(s)
    a = r.normal(size=100)
    b = a + 0.01 * r.normal(size=100)                # near-duplicate feature
    t = a + b + r.normal(scale=0.5, size=100)
    m = LinearRegression().fit(np.c_[a, b], t)
    coefs.append(m.coef_); preds.append(m.predict(x_new)[0])
coefs = np.array(coefs)
print(f"coef std across resamples: {coefs.std(0).round(2)}   prediction std: {np.std(preds):.3f}")
```

---

## Pitfalls & misconceptions

- **"A coefficient is the effect of the feature."** It's the association *holding the other features fixed*, and that's not causal unless you've controlled for every confounder (EL-05).
- **Comparing coefficient sizes across features in different units.** Standardize first, or compare effect sizes in meaningful units.
- **GD loss goes to `nan`.** The learning rate is too high for the largest curvature, often because of unscaled features. Data with `inf` or `nan` values does it too.
- **Trusting training $R^2$.** Report $R^2$ on held-out data.

## Cheat sheet

| Item | Formula |
|---|---|
| Model | $\hat y = Xw$ |
| MSE gradient | $\frac{2}{N}X^\top(Xw - y)$ |
| Closed form | $\hat w = (X^\top X)^{-1}X^\top y$ (use `lstsq`) |
| $R^2$ | $1 - \text{RSS}/\text{TSS}$ |
| Coefficient covariance | $\sigma^2 (X^\top X)^{-1}$ |
| Stable step | $\eta < 2/\lambda_{\max}\big(\tfrac2N X^\top X\big)$ |
| Ridge fix | $(X^\top X + \lambda I)^{-1}X^\top y$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. What does a coefficient mean in multiple vs simple regression?</summary>

Multiple: the change in $\hat y$ per unit of $x_j$ holding the other features fixed. Simple: the total association, including everything correlated with $x_j$. They differ whenever $x_j$ correlates with other predictors of $y$ (omitted-variable bias; see the demo: about +1.4 alone vs −1.0 controlled).
</details>

<details>
<summary>2. Why can the normal equation be slow or unstable, and what does GD trade?</summary>

It costs $O(Nd^2 + d^3)$, needs everything in memory, and squares the condition number when forming $X^\top X$, so near-collinear features make it unstable. GD trades exactness for cheap iterative steps that stream data and generalize to models without a closed form, at the price of tuning $\eta$ and monitoring convergence.
</details>

<details>
<summary>3. Loss path of GD on unscaled vs scaled features.</summary>

Unscaled: elongated elliptical contours. The path zig-zags across the narrow direction and crawls along the long one, taking many steps. Scaled: near-circular contours, so the path heads almost straight to the minimum. The cause is the Hessian's condition number (§4). In the demo, the scaled version converges in a small fraction of the steps.
</details>

<details>
<summary>4. Training and validation error both high and close together.</summary>

High bias, or under-fitting. The model is too simple, and more data won't help. Add features or polynomial terms, use a more flexible model, or reduce regularization.
</details>

<details>
<summary>5. What does collinearity do to coefficients and to predictions?</summary>

Coefficients get large variances and can flip sign between samples, because many combinations fit equally well. Predictions stay stable as long as new data has the same correlation structure (see the demo). Ridge or dropping features stabilizes the coefficients.
</details>

<details>
<summary>6. GD loss is nan after 3 epochs: two likely causes.</summary>

(1) The learning rate is too high for the largest curvature, often because features are unscaled, so the iterates diverge. (2) Bad data: `nan` or `inf` in the inputs or targets, or a division by zero in preprocessing. Also check that the gradient is averaged rather than summed over a large batch, which multiplies the effective step size.
</details>

## Where this leads

Next: [CORE-03 notes](03-classification-and-metrics.md). Same recipe (a linear score $w^\top x$, a loss from maximum likelihood, gradient descent), but now
the target is a **class**, so we squash the score into a probability and need new ways to judge success.
