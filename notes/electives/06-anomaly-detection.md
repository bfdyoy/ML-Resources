# EL-06 notes: Anomaly Detection

[← Lesson EL-06](../../lessons/electives/06-anomaly-detection.md) · [All notes](../README.md) · [← EL-05 notes](05-causal-inference-uplift.md) · Next: [EL-07 notes →](07-mechanistic-interpretability.md)

> **Reading time** ≈ 50 min. **You need:** [CORE-06 notes](../core-ml/06-unsupervised-learning.md) (PCA, density, k-means), [CORE-09 notes](../core-ml/09-kernels-svms-nearest-neighbours.md) §1 (distances, curse of dimensionality), and [CORE-03 notes](../core-ml/03-classification-and-metrics.md) §4 (PR vs ROC on rare classes).

---

## Where we are

Fraud, intrusions, defective parts, failing sensors: the interesting cases are **rare**, often **unlabeled**, and **changing**. Anomaly detection models "normal" and flags what doesn't fit. The model matters less than three framing decisions:

- what "unusual" means **in context**;
- how you **evaluate** with almost no labels;
- how you set **thresholds** that humans can act on.

---

## 1. Two settings

- **Outlier detection:** the training data *contains* anomalies (contaminated). The method must be robust to them. It's typical of exploratory analysis and of logs.
- **Novelty detection:** the training data is **clean** (only normal examples), and you flag new points unlike it. It's typical with a curated normal set: manufacturing, or a known-good period.

Many methods do both. The difference matters for robustness, and for reconstruction methods (§4).

---

## 2. Isolation forest

Build random trees: pick a random feature and a random split value between its min and max, and recurse until every point is alone. **Anomalies are few and different, so they get isolated in very few splits**, while normal points sit in dense regions and need many splits to separate.
The anomaly score uses the average path length $E[h(x)]$, normalized by $c(n)$ (the average path length of an unsuccessful search in a binary search tree with $n$ points):

```math
s(x) = 2^{-E[h(x)]/c(n)},\qquad c(n) = 2H(n-1) - \frac{2(n-1)}{n},\ H(i)\approx\ln i + 0.5772 .
```

$s$ close to 1 means an anomaly, and around 0.5 or below means normal. It's fast, it scales, it handles many features, and it needs no distances. It's weaker on **local** anomalies: points unusual relative to their own dense cluster, but not globally isolated.

## 3. Density and distance methods

- **LOF (local outlier factor):** compare a point's local density (from its $k$-NN distances) with its neighbours' densities. An LOF much greater than 1 means it's less dense than its neighbours. **The local view fixes a global-method failure:** with one tight cluster and one diffuse cluster, a point just outside the tight cluster is anomalous *for that cluster*, even though it's closer to everything than typical diffuse-cluster points are. A global distance threshold either misses it or flags the whole diffuse cluster.
- **Elliptic envelope / Mahalanobis distance:** for roughly Gaussian data, $d^2(x) = (x - \mu)^\top\Sigma^{-1}(x - \mu)$, which follows a $\chi^2_d$ distribution under normality, so it gives principled thresholds.
  **Masking:** outliers inflate the estimated $\Sigma$ and shift $\mu$, which makes them look *less* extreme. **Robust covariance** (the minimum covariance determinant) fits $\mu$ and $\Sigma$ on the most concentrated subset, so it unmasks them.
- **One-class SVM:** learns a boundary enclosing most of the data in a kernel feature space (CORE-09). It's sensitive to scaling and to its $\nu$/$\gamma$ hyperparameters.

---

## 4. Reconstruction-based detection

Fit a model that compresses and reconstructs the data (PCA with $k$ components, or an autoencoder). Points that reconstruct poorly don't follow the normal structure: the score is $\lVert x - \hat x\rVert^2$.

**When it fails:**

- **Contaminated training data:** an autoencoder trained on data that includes anomalies (especially recurring ones) **learns to reconstruct them too**. Their error drops and they go unnoticed. That's why reconstruction methods prefer novelty detection on clean data.
- **Simple anomalies:** a blank or constant input can reconstruct *better* than complex normal inputs.
- **Too much capacity:** an overly powerful autoencoder approaches the identity map and reconstructs everything well.

---

## 5. Evaluation and thresholds

### 5.1 Metrics

With 0.1% anomalies, ROC-AUC is dominated by the huge negative class: a model can have an FPR of 0.01 (a ROC-AUC near 0.99) and still raise 10× more false alarms than true ones (CORE-03 §4). Report:

- **PR-AUC** (its baseline is the anomaly rate);
- **precision@k**, at the k that analysts can actually review;
- **recall at a fixed alert budget**;
- for time series, *event*-level detection and time-to-detect.

### 5.2 Thresholds from an alert budget

The operational question isn't "what's the optimal threshold?" but "**analysts can review 50 alerts a day**". So set the threshold at the score quantile that yields about 50 alerts per day on recent traffic:

```math
\text{threshold} = \text{quantile}_{1 - 50/N_\text{day}}\big(\text{scores}\big).
```

Monitor precision among the reviewed alerts (that's your label stream), and re-tune as traffic changes. Combine the scores with **rules** (known fraud patterns) and **human feedback**. The reviewed alerts become labels for a supervised model later.

---

## 6. Context: unusual for *what*?

**Debug: the detector mostly flags weekend transactions.** The features (timestamps, volumes, merchant mixes) differ on weekends, and weekends are a minority of the data, so the detector learns that "weekend" is unusual.
Those are **contextual differences, not anomalies**. Fixes:

- add **context features** (day of week, hour, holidays), so the model knows a weekend is normal;
- **normalize per context:** compare each transaction with the same user's usual behaviour, or with the same weekday-and-hour baseline (z-scores within the context);
- fit **separate models**, or conditional models, per context;
- check what drives the alerts (feature contributions), and validate with a few labeled cases.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.metrics import roc_auc_score, average_precision_score
rng = np.random.default_rng(0)

# --- Global vs local anomalies: one tight cluster, one diffuse cluster -----------------------------------
tight = rng.normal([0, 0], 0.2, (500, 2)); diffuse = rng.normal([6, 6], 2.0, (500, 2))
local_anoms = rng.normal([0, 1.2], 0.05, (10, 2))       # just outside the tight cluster
X = np.vstack([tight, diffuse, local_anoms]); y = np.r_[np.zeros(1000), np.ones(10)]
iso = -IsolationForest(random_state=0).fit(X).score_samples(X)
lof = -LocalOutlierFactor(n_neighbors=20).fit(X).negative_outlier_factor_
for name, s in [("isolation forest", iso), ("LOF", lof)]:
    print(f"{name:16s}: PR-AUC {average_precision_score(y, s):.2f}  ROC-AUC {roc_auc_score(y, s):.2f}")

# --- Masking: outliers inflate the classical covariance; robust MCD unmasks them ------------------------------------
normal = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 950)
outl = rng.multivariate_normal([3, -3], [[0.3, 0], [0, 0.3]], 50)     # a cluster against the correlation
Z = np.vstack([normal, outl]); yz = np.r_[np.zeros(950), np.ones(50)]
for name, est in [("classical", EmpiricalCovariance()), ("robust MCD", MinCovDet(random_state=0))]:
    d2 = est.fit(Z).mahalanobis(Z)
    print(f"{name:10s} Mahalanobis: mean d^2 of outliers {d2[yz == 1].mean():6.1f}, of normals {d2[yz == 0].mean():.1f}")
```

```python
# --- Alert budget thresholds and precision@k --------------------------------------------------------------------------
N_day, budget = 20_000, 50
scores = np.r_[rng.normal(0, 1, N_day - 20), rng.normal(3.5, 1, 20)]       # 20 true anomalies today
labels = np.r_[np.zeros(N_day - 20), np.ones(20)]
thr = np.quantile(scores, 1 - budget / N_day)
alerts = scores >= thr
print(f"threshold {thr:.2f} -> {alerts.sum()} alerts; precision@{budget} {labels[alerts].mean():.2f}, recall {labels[alerts].sum() / 20:.2f}")
print(f"ROC-AUC {roc_auc_score(labels, scores):.3f} looks great; PR-AUC {average_precision_score(labels, scores):.3f} tells the operational truth")

# --- Context: weekends look 'anomalous' until you normalize per context -----------------------------------------------------
days = rng.integers(0, 7, 7000); weekend = days >= 5
amount = np.where(weekend, rng.normal(80, 15, 7000), rng.normal(40, 10, 7000))   # weekends: bigger baskets
amount[:20] = np.where(weekend[:20], 160, 90)                                   # 20 true anomalies (about 2x usual)
truth = np.zeros(7000); truth[:20] = 1
raw = -IsolationForest(random_state=0).fit(amount[:, None]).score_samples(amount[:, None])
z = np.empty(7000)
for w in (False, True):
    m = weekend == w; z[m] = np.abs(amount[m] - np.median(amount[m])) / amount[m].std()
top_raw, top_z = np.argsort(-raw)[:50], np.argsort(-z)[:50]
print(f"raw detector: {weekend[top_raw].mean():.0%} of the top-50 alerts are weekend, true anomalies caught {truth[top_raw].sum():.0f}/20")
print(f"per-context z-score: {weekend[top_z].mean():.0%} weekend, true anomalies caught {truth[top_z].sum():.0f}/20")
```

---

## Pitfalls & misconceptions

- **Reporting ROC-AUC on very rare anomalies.**
- **Fitting reconstruction models on contaminated data** and expecting them to flag what they've learned to reconstruct.
- **Context-free features** (time, user, location baselines missing).
- **Unscaled features** in distance or density methods.
- **Thresholds chosen in a vacuum** instead of from the review capacity.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Isolation-forest score | $2^{-E[h(x)]/c(n)}$; short paths mean anomalies |
| LOF | local density relative to the neighbours' (≫ 1 is anomalous) |
| Mahalanobis | $(x-\mu)^\top\Sigma^{-1}(x-\mu)\sim\chi^2_d$; use robust MCD against masking |
| Reconstruction | $\lVert x - \hat x\rVert^2$; prefer clean (novelty) training data |
| Metrics | PR-AUC, precision@k, recall at the budget |
| Threshold | quantile $1 - \text{budget}/N$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why does isolation forest isolate anomalies in fewer splits?</summary>

Anomalies are few and lie in sparse regions, far from the bulk, so random axis-aligned splits separate them early. Points in dense regions need many splits to be isolated. A shorter expected path length means more anomalous.
</details>

<details>
<summary>2. LOF is local: what global failure does it fix?</summary>

With clusters of different density, global distance or density thresholds either flag the whole sparse cluster or miss points just outside a dense one. LOF compares each point with its own neighbourhood's density, catching local anomalies (the demo: LOF beats isolation forest on PR-AUC).
</details>

<details>
<summary>3. Why is ROC-AUC misleading at 0.1% anomalies?</summary>

The FPR is diluted by the enormous normal class, so a small FPR still means many false alarms relative to the few true anomalies. Report PR-AUC, precision@k at the review budget, and recall at that budget.
</details>

<details>
<summary>4. Choosing a threshold when analysts can review 50 alerts a day.</summary>

Set it at the score quantile that yields about 50 alerts per day ($1 - 50/N_\text{day}$) on recent traffic. Measure precision on the reviewed alerts, adjust as traffic changes, and combine with rules and feedback.
</details>

<details>
<summary>5. Why can an autoencoder trained on contaminated data reconstruct anomalies?</summary>

If the anomalies (especially recurring patterns) are in the training data, the model learns to reconstruct them, so their error is low. High capacity worsens this. Train on clean normal data (novelty detection), limit the capacity, or use robust training.
</details>

<details>
<summary>6. The detector flags mostly weekend transactions.</summary>

Weekend behaviour differs and is a minority of the data, so it looks globally unusual. That's a context effect, not an anomaly. Add context features, normalize per context (per user, per weekday and hour), or model each context separately (the demo: per-context z-scores fix it).
</details>

## Where this leads

Next: [EL-07 notes](07-mechanistic-interpretability.md). From finding unusual data to understanding unusual *models*: reverse-engineering what's computed inside a transformer.
