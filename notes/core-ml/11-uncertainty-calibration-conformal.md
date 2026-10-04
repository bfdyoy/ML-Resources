# CORE-11 notes: Uncertainty: Calibration & Conformal Prediction

[← Lesson CORE-11](../../lessons/core-ml/11-uncertainty-calibration-conformal.md) · [All notes](../README.md) · [← CORE-10 notes](10-data-centric-ml.md) · Next: [CORE-12 notes →](12-hyperparameter-optimization.md)

> **Reading time** ≈ 60 min. **You need:** [CORE-03 notes](03-classification-and-metrics.md) §1.5, §2, §6 (softmax, thresholds, calibration) and [MATH-03 notes](../math/03-probability-statistics.md) §D (quantiles, error bars).

---

## Where we are

A model's output is only as useful as **how much you can trust it**. This lesson gives you two complementary tools:

1. **Calibration** makes a predicted probability mean what it says: "0.8" should be right 80% of the time.
2. **Conformal prediction** turns *any* model into one that outputs **sets** or **intervals** with a guaranteed coverage rate, with no distributional assumptions beyond exchangeability.

---

## 1. Calibration

### 1.1 Definition and measurement

A classifier is calibrated if $P(Y = 1 \mid \hat p(X) = p) = p$ for every $p$ (and, for multiclass, the confidence of the top class matches its accuracy).

- **Reliability diagram:** bin the predictions by confidence and plot the average confidence against the observed accuracy in each bin. Perfect calibration is the diagonal. Points below the diagonal mean **over**confidence.
- **Expected calibration error (ECE):** the weighted average gap across $B$ bins:

```math
\mathrm{ECE} = \sum_{b=1}^B \frac{n_b}{N}\,\big\lvert \mathrm{acc}(b) - \mathrm{conf}(b)\big\rvert .
```

  ECE depends on the binning, and a model can have low ECE while being useless (always predicting the base rate is perfectly calibrated). So also report a **proper scoring rule**: log loss or the Brier score, $\frac1N\sum (\hat p_i - y_i)^2$.
  Proper scoring rules reward calibration *and* sharpness together.

### 1.2 Why modern nets are overconfident

The log loss keeps rewarding larger logits on correctly classified training examples **even after accuracy has saturated**. With high capacity (and little weight decay), the network keeps
inflating its logits, so its confidence outruns its accuracy. Guo et al. (2017) showed that deeper, wider, batch-normalized nets are worse calibrated than older, smaller ones, even when they're more accurate.

### 1.3 Recalibration methods (all fitted on held-out data)

| Method | Maps a score $s$ to | Parameters | Notes |
|---|---|---|---|
| **Platt scaling** | $\sigma(a s + b)$ | 2 | Logistic regression on the score. Good for SVMs and small data. |
| **Isotonic regression** | Any monotone step function | Many | Flexible, but needs more data, or it overfits |
| **Temperature scaling** | $\operatorname{softmax}(z/T)$ | 1 | For neural nets. $T > 1$ softens the distribution. Doesn't change the argmax, so accuracy is unchanged. |

**Why held-out data?** On the training set, the model is overconfident *and* mostly right. A calibrator fitted there learns "trust the model", which is exactly the wrong lesson.
Use a separate calibration split, or cross-validation (`CalibratedClassifierCV`).

Temperature scaling finds $T$ by minimizing the NLL on the calibration set. That's a one-dimensional convex problem.

---

## 2. Conformal prediction

### 2.1 The goal

Given *any* trained model, produce a set $C(x)$ such that

```math
P\big(Y_\text{test} \in C(X_\text{test})\big) \ge 1 - \alpha
```

(for example 90%), **without assuming the model is good or the data is Gaussian**.

### 2.2 Split conformal, step by step (classification)

1. Train the model on a training split. Keep a **calibration split** of $n$ examples it never saw.
2. Define a **conformity score** that is large when the model is wrong, for example $s(x, y) = 1 - \hat p(y\mid x)$.
3. Compute $s_i = s(x_i, y_i)$ on the calibration set.
4. Take the quantile $\hat q$ = the $\lceil (n+1)(1-\alpha)\rceil$-th smallest of the $s_i$. That's slightly above the plain $(1-\alpha)$ quantile.
5. For a new $x$, include **every label $y$ with $s(x, y) \le \hat q$**: $C(x) = \lbrace y : \hat p(y\mid x) \ge 1 - \hat q\rbrace$.

### 2.3 Why the guarantee holds (the whole proof)

Assume the calibration points and the test point are **exchangeable** (i.i.d. is enough). Then the test score $s_\text{test}$ is equally likely to fall at any rank among the $n + 1$ scores.
So the probability that it's **at or below** the $\lceil (n+1)(1-\alpha)\rceil$-th smallest calibration score is at least $\frac{\lceil (n+1)(1-\alpha)\rceil}{n+1} \ge 1 - \alpha$.
And "$s_\text{test} \le \hat q$" is exactly "$y_\text{test} \in C(x_\text{test})$". With continuous scores there's also an upper bound:

```math
1 - \alpha \;\le\; P(Y \in C(X)) \;\le\; 1 - \alpha + \frac{1}{n+1}.
```

That's all. No model assumptions, and finite-sample validity. The **probability is over the random draw of the calibration set and the test point together**. It's an average over many deployments, not a promise about any single $x$.

### 2.4 What the sets tell you

The set **grows when the model is unsure**. For an easy image, the set is {cat}. For an ambiguous one, it's {cat, fox, dog}. That's a feature: set size is an honest, per-example uncertainty signal.
A better model gives smaller sets at the same coverage. The scores used to rank models are therefore **coverage** (it should be about $1 - \alpha$) **and average set size**.
The **APS** score (adaptive prediction sets) uses cumulative probability mass, so the set sizes adapt better to example difficulty.

### 2.5 Regression intervals

- **Absolute-residual scores:** with $s = \lvert y - \hat y(x)\rvert$, the interval is $\hat y(x) \pm \hat q$. It's valid, but **the same width everywhere**, even where the noise is small.
- **Conformalized quantile regression (CQR):** fit lower and upper quantile models $\hat q_\text{lo}, \hat q_\text{hi}$ (for example with a GBM using the quantile loss). Use the score
  $s = \max\big(\hat q_\text{lo}(x) - y,\ y - \hat q_\text{hi}(x)\big)$, and output $[\hat q_\text{lo}(x) - \hat q,\ \hat q_\text{hi}(x) + \hat q]$.
  The intervals are **adaptive** (wide where the data is noisy) and still carry the guarantee.

### 2.6 The limits

- **Marginal, not conditional.** 90% overall can be 97% for an easy group and 70% for a hard one (the demo below shows this). The fix is **group-conditional (Mondrian)** conformal: calibrate separately per group, which needs enough calibration data in each group.
- **Exchangeability breaks under distribution shift.** If production data drifts, coverage silently degrades. Monitor the empirical coverage on labeled production samples, recalibrate on recent data, or use weighted or adaptive conformal methods for shift.

```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)

def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e

# An overconfident "network": the logits are the true ones times 3
K, n = 5, 20000
true_logits = rng.normal(size=(n, K)) * 1.5
y = np.array([rng.choice(K, p=p) for p in softmax(true_logits, axis=1)])
logits = 3.0 * true_logits
cal, test = slice(0, 10000), slice(10000, None)
nll = lambda T, sl: -log_softmax(logits[sl] / T, axis=1)[np.arange(len(y[sl])), y[sl]].mean()
T = minimize_scalar(lambda T: nll(T, cal), bounds=(0.05, 20), method="bounded").x
for name, Tval in [("before", 1.0), ("after", T)]:
    P = softmax(logits[test] / Tval, axis=1)
    print(f"{name:6s} T={Tval:.2f}  accuracy={np.mean(P.argmax(1) == y[test]):.3f}  ECE={ece(P.max(1), P.argmax(1) == y[test]):.3f}  NLL={nll(Tval, test):.3f}")
```

```python
# Split conformal for classification: coverage over many random calibration/test splits
X, yc = make_classification(n_samples=6000, n_features=20, n_informative=8, n_classes=4,
                            n_clusters_per_class=1, class_sep=0.8, random_state=0)
clf = LogisticRegression(max_iter=2000).fit(X[:2000], yc[:2000])
P_rest, y_rest = clf.predict_proba(X[2000:]), yc[2000:]
alpha, n_cal, covs, sizes = 0.1, 500, [], []
for s in range(300):
    idx = np.random.default_rng(s).permutation(len(y_rest)); c, t = idx[:n_cal], idx[n_cal:]
    scores = 1 - P_rest[c, y_rest[c]]
    qhat = np.sort(scores)[int(np.ceil((n_cal + 1) * (1 - alpha))) - 1]
    sets = P_rest[t] >= 1 - qhat
    covs.append(sets[np.arange(len(t)), y_rest[t]].mean()); sizes.append(sets.sum(1).mean())
print(f"target coverage {1-alpha:.2f}; mean coverage {np.mean(covs):.3f} (bound <= {1-alpha+1/(n_cal+1):.3f}); mean set size {np.mean(sizes):.2f}; base accuracy {clf.score(X[2000:], y_rest):.2f}")
```

```python
# Regression: constant-width conformal vs CQR, on heteroscedastic data, and per-group coverage
N = 6000
x = rng.uniform(0, 10, N); noise_sd = np.where(x > 5, 3.0, 0.3)          # the "hard" group is x > 5
yr = np.sin(x) * 3 + rng.normal(0, noise_sd)
Xr = x[:, None]; tr, ca, te = slice(0, 2000), slice(2000, 4000), slice(4000, None)
alpha = 0.1
k = int(np.ceil((2000 + 1) * (1 - alpha))) - 1

mean_m = GradientBoostingRegressor(random_state=0).fit(Xr[tr], yr[tr])
q_abs = np.sort(np.abs(yr[ca] - mean_m.predict(Xr[ca])))[k]
lo_m = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, random_state=0).fit(Xr[tr], yr[tr])
hi_m = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, random_state=0).fit(Xr[tr], yr[tr])
s_cqr = np.maximum(lo_m.predict(Xr[ca]) - yr[ca], yr[ca] - hi_m.predict(Xr[ca]))
q_cqr = np.sort(s_cqr)[k]

pred = mean_m.predict(Xr[te]); lo, hi = lo_m.predict(Xr[te]) - q_cqr, hi_m.predict(Xr[te]) + q_cqr
hard = x[te] > 5
for name, L, H in [("absolute residual", pred - q_abs, pred + q_abs), ("CQR", lo, hi)]:
    cov = (yr[te] >= L) & (yr[te] <= H)
    print(f"{name:17s} overall {cov.mean():.2f} | easy group {cov[~hard].mean():.2f} | hard group {cov[hard].mean():.2f} "
          f"| width easy {np.mean((H-L)[~hard]):.1f}, hard {np.mean((H-L)[hard]):.1f}")
# (One split, so coverage wobbles around 0.90 by a point or two. The guarantee is on average over splits.)
```

---

## Pitfalls & misconceptions

- **Calibrating on the training data.**
- **Reading conformal coverage as "90% for this patient".** It's marginal: an average over the population.
- **Ignoring set size.** Trivial sets that contain every label always cover. Coverage without efficiency means nothing.
- **Reusing the calibration set for model selection.** That breaks exchangeability. Keep it clean.
- **Assuming the guarantee survives drift.** It doesn't. Monitor coverage in production (PROD-03).

## Cheat sheet

| Item | Formula |
|---|---|
| ECE | $\sum_b \frac{n_b}{N}\lvert\mathrm{acc}_b - \mathrm{conf}_b\rvert$ |
| Brier score | $\frac1N\sum(\hat p_i - y_i)^2$ |
| Temperature scaling | $\operatorname{softmax}(z/T)$, $T$ = argmin NLL on calibration data |
| Conformal quantile | $\hat q$ = the $\lceil (n+1)(1-\alpha)\rceil$-th smallest calibration score |
| Classification set | $\lbrace y: 1 - \hat p(y\mid x) \le \hat q\rbrace$ |
| Coverage bounds | $1-\alpha \le P(Y\in C) \le 1-\alpha + \frac{1}{n+1}$ |
| CQR score | $\max(\hat q_\text{lo} - y,\ y - \hat q_\text{hi})$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why are modern deep nets usually overconfident?</summary>

The log loss keeps pushing logits up on training examples that are already correct. With large capacity, little regularization, and BatchNorm, the networks inflate their logits beyond what their accuracy justifies. Temperature scaling ($T > 1$) undoes most of this.
</details>

<details>
<summary>2. Why calibrate on data the model wasn't trained on?</summary>

On training data the model is overconfident and right, so a calibrator fitted there learns to trust that overconfidence. Calibration must reflect performance on unseen data.
</details>

<details>
<summary>3. What exactly does 90% coverage guarantee?</summary>

That, averaged over the random draw of the calibration set and a new test point from the same (exchangeable) distribution, the true label is in the set at least 90% of the time. It's marginal over the population, not conditional on a particular $x$ or group.
</details>

<details>
<summary>4. Why are sets bigger for hard examples?</summary>

For hard examples the model spreads its probability, so more labels pass the threshold $\hat p(y\mid x) \ge 1 - \hat q$. That's a feature: set size is an honest per-example uncertainty signal.
</details>

<details>
<summary>5. 90% overall but 70% for one group.</summary>

Coverage is marginal: the easy majority groups are over-covered, which compensates for the under-covered hard group. Use group-conditional (Mondrian) conformal calibration, adaptive scores (CQR, APS), or better features for that group. The regression demo shows constant-width intervals failing the hard group, and CQR repairing it.
</details>

<details>
<summary>6. Deployed coverage drops from 90% to 75%.</summary>

Distribution shift breaks exchangeability between the calibration data and production data. Recalibrate on recent labeled data, use weighted or adaptive conformal methods designed for shift, and monitor coverage continuously.
</details>

## Where this leads

Next: [CORE-12 notes](12-hyperparameter-optimization.md), the last lesson of Path 1. Every lesson so far has had knobs ($\lambda$, $C$, $\gamma$, depth, learning rate).
CORE-12 is about turning those knobs efficiently, without fooling yourself. The GP from CORE-09 comes back as the engine of Bayesian optimization.
