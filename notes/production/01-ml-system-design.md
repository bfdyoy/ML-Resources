# PROD-01 notes: ML System Design

[← Lesson PROD-01](../../lessons/production/01-ml-system-design.md) · [All notes](../README.md) · [← GEN-10 notes](../llms-genai/10-advanced-diffusion-flow-matching.md) · Next: [PROD-02 notes →](02-mlops-in-practice.md)

> **Reading time** ≈ 45 min. **You need:** [CORE-01 notes](../core-ml/01-ml-workflow-end-to-end.md) (framing, splits, leakage) and [MATH-03 notes](../math/03-probability-statistics.md) §D (tests). Useful: [CORE-11 notes](../core-ml/11-uncertainty-calibration-conformal.md) on shift.

---

## Where we are

The model is usually the smallest part of an ML system. Around it sit data collection, feature computation, serving, monitoring, and feedback, and most production failures happen there. This lesson is about designing the *whole* system, with the few quantitative tools that keep design decisions honest.

---

## 1. Framing: from a business goal to an ML objective

A design doc starts with four things:

1. **The business objective** ("reduce support cost"), and a measurable proxy (tickets resolved without a human).
2. **The ML task**: input, output, and *when* the prediction is made. That determines which features are legal (CORE-01 §6).
3. **Constraints**: latency (state percentiles, like "p99 < 100 ms", because averages hide the slow tail), throughput, cost per prediction, privacy and regulation, interpretability.
4. **Success metrics at three levels**: offline ML metrics (PR-AUC), online product metrics (conversion), and guardrail metrics (latency, complaint rate). The offline metric is only useful if it moves together with the online one, so verify that with an A/B test.

### 1.1 Start with a heuristic

Google's Rule #1 is "don't be afraid to launch a product without ML". A heuristic ("show the most popular items", "flag transactions over N euros abroad"):

- **ships the pipeline first:** data flow, logging, serving, and monitoring, which is most of the work and most of the risk;
- **sets a baseline** that any model must beat;
- **starts collecting the labels and feedback** the ML model will need;
- is easy to debug and explain.

Many teams find the heuristic is good enough, or that the ML model's gain is smaller than expected. Either way, you've saved months.

---

## 2. Data: the half of the system that breaks

- **Sources and labels:** where do the labels come from (human labeling, natural feedback such as clicks or returns, or delayed outcomes such as chargebacks 60 days later)? How noisy are they (CORE-10)?
- **Sampling:** is the training data representative of what you'll serve? Watch selection bias: a model trained only on *approved* loans never sees how rejected applicants would have behaved.
- **Imbalance:** CORE-10 §2.
- **Training/serving skew:** the model sees *different feature values* in serving than in training, for the same entity. Common causes:
  - **two code paths:** features computed in a Python batch job for training, but re-implemented in Java or SQL for serving, with subtle differences (time zones, null handling, rounding);
  - **time-travel:** training features computed with data that wasn't available yet at prediction time (an aggregate that includes later events);
  - **a stale or missing feature in serving** (the feature store isn't updated, or there's a default value where training had real values).

  The fixes: **one feature definition used by both paths** (a feature store, or shared transformation code), **logging the features at serving time** and training on those logs, and point-in-time correct joins.

---

## 3. Serving: batch vs online

| | Batch (precompute) | Online (on request) |
|---|---|---|
| How | Score everything on a schedule; store the results in a table or cache | Compute the features and prediction when the request arrives |
| Gains | Cheap, simple, no latency pressure; complex models are fine | Fresh: uses the latest context (current session, cart, query); handles new users and items |
| Loses | Stale; wasted work on entities nobody requests; can't react to new context | Latency budget; needs real-time feature infrastructure; harder to scale and debug |
| Typical | Daily churn scores, nightly recommendations | Search ranking, fraud at checkout |

Hybrid designs are common: precompute the heavy embeddings or candidates in batch, and do the light re-ranking online.
**Edge deployment** (on-device) adds privacy and offline capability, with tight memory and compute limits. That's where quantization, pruning, and distillation come in (GEN-08 §3).

---

## 4. Distribution shift, and detecting it without labels

Labels often arrive late (or never), so you need **label-free** signals. Three kinds of shift (PROD-03 goes deeper):

- **Covariate shift:** $p(x)$ changes while $p(y\mid x)$ stays the same (new user demographics).
- **Label (prior) shift:** $p(y)$ changes while $p(x\mid y)$ stays the same (fraud rate spikes during a holiday).
- **Concept drift:** $p(y\mid x)$ itself changes (fraudsters change tactics; the same pattern now means something else). This one can't be detected from inputs alone.

**Detecting covariate shift without labels:**

1. **Per-feature distribution tests**, comparing a reference window with the current window:
   - a two-sample KS test for numeric features, chi-squared for categoricals;
   - or an effect size like the **population stability index**, $\text{PSI} = \sum_b (a_b - e_b)\ln\frac{a_b}{e_b}$ over bins with expected fractions $e_b$ and actual fractions $a_b$. A common rule of thumb: below 0.1 means stable, 0.1–0.2 means moderate shift, above 0.2 means significant.
2. **Prediction distribution:** monitor the distribution of model scores and the predicted-class rates. They're cheap and sensitive.
3. **A domain classifier:** train a classifier to tell reference rows from current rows. **An AUC near 0.5 means no detectable shift. An AUC well above 0.5 means shift**, and its feature importances tell you *which* features moved. It's multivariate, so it catches shifts in correlations that per-feature tests miss.

Shift in an input isn't automatically a problem. The question is whether it moves *performance*. That's why proxies, and eventually labels, matter (PROD-03).

---

## 5. Design exercise: a "similar products" recommender

- **Objective:** more product-page engagement and add-to-cart. Offline proxy: recall@k of co-viewed or co-purchased items. Guardrails: latency, and catalogue coverage (not always showing the same bestsellers).
- **Data:** the product catalogue (text, images, attributes, price), user sessions (co-view and co-purchase pairs), and inventory status.
- **Model (two-stage, as in GEN-05):**
  - **candidate generation:** item embeddings (text and image encoders, plus co-occurrence embeddings like item2vec) with ANN search;
  - **re-ranking:** a GBM or a small network on features like similarity, price difference, popularity, stock, and margin.
  - **Heuristic baseline:** same category plus a similar price.
- **Serving:** precompute the top-50 neighbours per item in batch (nightly, plus incremental updates for new items). Filter out-of-stock items and re-rank online, at under 50 ms.
- **Cold start:** new items use their content embeddings (text and image) until they get interactions.
- **Monitoring:** CTR on the widget, coverage, latency, the embedding-drift domain classifier, the fraction of unavailable recommendations.
- **Feedback loop and bias:** the model only learns from items it chose to show (*position and exposure bias*). Reserve a small randomized exploration slot, and log impressions, not just clicks.

```python
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
rng = np.random.default_rng(0)

def psi(ref, cur, bins=10):
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref); a = np.histogram(cur, edges)[0] / len(cur)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return np.sum((a - e) * np.log(a / e))

ref = rng.normal(0, 1, (5000, 3))
cur_same = rng.normal(0, 1, (5000, 3))
cur_shift = rng.normal(0, 1, (5000, 3)); cur_shift[:, 1] += 0.5          # feature 1 shifted
cur_corr = rng.normal(0, 1, (5000, 3)); cur_corr[:, 2] = cur_corr[:, 0] * 0.9 + 0.44 * cur_corr[:, 2]   # same marginals, new correlation

for name, cur in [("no shift", cur_same), ("feature 1 shifted", cur_shift), ("correlation change only", cur_corr)]:
    psis = [psi(ref[:, j], cur[:, j]) for j in range(3)]
    ks = [stats.ks_2samp(ref[:, j], cur[:, j]).pvalue for j in range(3)]
    X = np.vstack([ref, cur]); y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    auc = cross_val_score(GradientBoostingClassifier(random_state=0), X, y, cv=3, scoring="roc_auc").mean()
    print(f"{name:24s} PSI per feature {np.round(psis, 3)}  KS p-values {np.round(ks, 3)}  domain-classifier AUC {auc:.3f}")

# Latency: mean hides the tail
lat = np.r_[rng.lognormal(3.0, 0.3, 9800), rng.lognormal(5.5, 0.4, 200)]      # ms, with a slow 2% tail
print(f"latency mean {lat.mean():.0f} ms, p50 {np.percentile(lat, 50):.0f} ms, p99 {np.percentile(lat, 99):.0f} ms")
```

---

## Pitfalls & misconceptions

- **Optimizing an offline metric that doesn't move the business metric.**
- **Building the model before the data and serving pipeline exists.**
- **Two implementations of the same feature.** That's skew waiting to happen.
- **Treating every drift alert as a reason to retrain** (PROD-02, PROD-03).
- **Ignoring feedback loops:** recommenders and fraud models shape the data they'll later train on.

## Cheat sheet

| Item | Rule / formula |
|---|---|
| Metrics | offline ML metric ↔ online product metric ↔ guardrails |
| First version | a heuristic + the full pipeline |
| Skew prevention | one feature definition, logged serving features, point-in-time joins |
| PSI | $\sum_b (a_b - e_b)\ln(a_b/e_b)$; > 0.2 is significant |
| Label-free shift detection | per-feature tests, prediction drift, a domain classifier (AUC > 0.5) |
| Latency | report p50/p95/p99, not the mean |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why start with a heuristic?</summary>

It delivers the end-to-end pipeline (data, logging, serving, monitoring), sets a baseline, starts collecting labels and feedback, and is easy to debug. Often it captures most of the value, and any model then has a clear number to beat.
</details>

<details>
<summary>2. Training-serving skew: two concrete causes.</summary>

(1) Features implemented twice (a batch pipeline for training, real-time code for serving) with subtle differences in nulls, time zones, or rounding. (2) Time-travel or staleness: training features use future information, or serving features are stale or defaulted. Fix: shared definitions, logged serving features, point-in-time joins.
</details>

<details>
<summary>3. Batch vs online prediction trade-offs.</summary>

Batch: cheap, simple, no latency constraint, but stale, and it can't use request-time context or handle new entities. Online: fresh and context-aware, but it needs real-time features and a latency budget, and costs more to run and debug. Hybrids precompute the heavy parts and finish online.
</details>

<details>
<summary>4. Detecting covariate shift without labels.</summary>

Compare reference and current windows: per-feature KS/chi-squared tests or PSI, drift in the prediction-score distribution, and a domain classifier whose AUC above 0.5 signals (multivariate) shift, with feature importances showing what moved. The demo shows the domain classifier catching a correlation change that per-feature tests miss.
</details>

<details>
<summary>5. Design a "similar products" recommender.</summary>

See §5: objective and metrics, catalogue + session data, two-stage retrieval (embeddings + ANN) with re-ranking, batch-precomputed neighbours with online filtering, content-based cold start, monitoring (CTR, coverage, drift), and exploration to counter feedback loops.
</details>

## Where this leads

Next: [PROD-02 notes](02-mlops-in-practice.md). A design is a plan. MLOps is the machinery that makes it repeatable: experiment tracking, pipelines, packaging, serving, CI/CD, and continual training.
