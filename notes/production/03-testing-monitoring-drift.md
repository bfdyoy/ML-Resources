# PROD-03 notes: Testing, Monitoring & Drift

[← Lesson PROD-03](../../lessons/production/03-testing-monitoring-drift.md) · [All notes](../README.md) · [← PROD-02 notes](02-mlops-in-practice.md) · Next: [PROD-04 notes →](04-distributed-training.md)

> **Reading time** ≈ 50 min. **You need:** [PROD-01 notes](01-ml-system-design.md) §4 (shift types, PSI, the domain classifier), [MATH-03 notes](../math/03-probability-statistics.md) §A2 and §D (Bayes, tests), and [CORE-03 notes](../core-ml/03-classification-and-metrics.md) §3 (the confusion matrix).

---

## Where we are

PROD-02 listed the tests and the monitoring an ML system needs. This note makes them precise:

- what to test, with concrete examples;
- the three kinds of drift, written as probability statements, with a way to *estimate* label shift without labels;
- why naive statistical alerts are useless at scale, and what to do instead.

---

## 1. Testing ML systems (the ML Test Score view)

The ML Test Score rubric scores four areas: **data, model development, infrastructure, monitoring**. The tests that pay off most:

### 1.1 Data tests

- **Schema:** columns, types, allowed categories.
- **Ranges and constraints:** age in [0, 120], non-negative prices, valid timestamps.
- **Missingness:** the null rate per column within bounds.
- **Distribution expectations:** the mean, quantiles, and category frequencies against a reference, with *tolerances*.
- **Relationships:** `end_date ≥ start_date`, totals that match their sums.
- **Leakage guards:** no columns derived from the label; no future timestamps in the features.

### 1.2 Behavioural (model) tests, CheckList-style

- **Minimum functionality:** simple cases the model *must* get right ("I love it" → positive).
- **Invariance:** perturbations that shouldn't change the output (for sentiment: swap names, locations, or neutral words; the prediction must stay the same).
- **Directional expectations:** perturbations with a known *direction* ("good" → "not good" must lower the positive score; adding "terrible" must not increase it).
- **Slices:** a minimum metric per important segment.

These catch failures that aggregate metrics hide, and they double as regression tests.

### 1.3 Training/serving skew test

Log the features and predictions in serving. Periodically **recompute** the features for the same entities with the *training* pipeline, and compare them value by value, and compare the predictions too. Any systematic difference is skew (PROD-01 §2).

---

## 2. Three kinds of drift, precisely

Write the joint distribution as $p(x, y) = p(y\mid x)\,p(x) = p(x\mid y)\,p(y)$:

| Drift | What changes | What stays the same | Fraud example |
|---|---|---|---|
| **Covariate shift** | $p(x)$ | $p(y\mid x)$ | More mobile and international transactions after launching in a new country |
| **Label (prior) shift** | $p(y)$ | $p(x\mid y)$ | A fraud wave during holidays: more fraud, but each fraud looks like before |
| **Concept drift** | $p(y\mid x)$ | — | Fraudsters adopt a new tactic: patterns that used to be safe are now fraud |

### 2.1 Estimating label shift without labels

Under label shift, the model's **confusion matrix** $C_{ij} = P(\hat y = i\mid y = j)$, estimated on labeled validation data, still applies in production (because $p(x\mid y)$ is unchanged). The distribution of predictions you observe in production, $\mu_i = P_\text{prod}(\hat y = i)$, mixes those columns with the new class priors $q$:

```math
\mu = C\,q \quad\Longrightarrow\quad \hat q = C^{-1}\hat\mu .
```

That's **black-box shift estimation** (BBSE). It needs only the unlabeled production predictions and the validation confusion matrix. You can then reweight the training data by $q_j/p_j$, or adjust the thresholds and recalibrate the probabilities.
(The same idea underlies quantification methods for estimating prevalence.) The code shows it recovering a jump from a 5% to a 15% positive rate.

### 2.2 Monitoring quality when labels arrive late

When labels come weeks later (chargebacks, churn):

- **Proxy metrics:** the prediction-score distribution, the predicted-positive rate, confidence and entropy, and downstream business proxies (manual-review overturn rate, customer complaints).
- **Input drift** on the important features (weighted by importance).
- **BBSE-style prior estimates** (above).
- **Partial early labels:** some labels arrive fast (a customer reports fraud immediately). Track them, knowing they're biased.
- **Back-filled performance** once the labels mature, with the evaluation aligned to the prediction date.

---

## 3. Alerts that are actionable, not noisy

### 3.1 Why tests fire constantly at scale

A significance test asks "is there *any* difference?" With 1M rows per window, the standard error is tiny, so a KS test detects shifts far too small to matter. **p-values shrink with sample size, even for a fixed, negligible effect.** The code shows a 0.02-standard-deviation shift that is "highly significant" at $n = 10^6$ and invisible at $n = 10^3$.
Monitoring 200 features at $\alpha = 0.05$ also guarantees false alarms every day (multiple testing, MATH-03 §D4).

### 3.2 Make alerts actionable

- Alert on **effect sizes** (PSI, the Wasserstein distance, the change in a mean in units of its standard deviation), with thresholds set on historical data, not on p-values.
- **Weight by feature importance.** Drift in unimportant features goes into a weekly report, not a page.
- Use **windows and persistence:** alert when drift persists for $k$ consecutive windows, which dampens noise.
- Prefer **prediction-level and performance-level** alerts as the primary pages. Use feature drift for diagnosis.
- **Group related alerts.** Twelve features drifting at once after a deployment usually has *one* cause.

**Debug (the lesson's question 6): 12 features alert right after a deployment, and accuracy is unchanged.** The first hypothesis is a **pipeline or data change**, not the world changing: a new feature-computation version, a unit or encoding change, a schema change, a different upstream source, or the *reference window* being recomputed.
Check the deployment diff and the feature logs. Real-world drift rarely hits a dozen features at the same instant.

---

## 4. When to retrain

| Strategy | When it fits |
|---|---|
| **Scheduled** (daily or weekly) | The data changes steadily, labels arrive regularly, and retraining is cheap and safe |
| **Triggered** (performance drop or confirmed drift) | Retraining is expensive; changes are episodic |
| **Continual / online** | Very fast-moving domains (ads, news), with robust automated validation |

Whichever you use: the promotion gate from PROD-02 §5.2 always applies, and fresh data must pass the data tests *before* training on it.

```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(0)

# --- p-values shrink with n for a fixed tiny effect; an effect-size measure doesn't --------------------
def psi(ref, cur, bins=10):
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-6, None); a = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-6, None)
    return np.sum((a - e) * np.log(a / e))
for n in [1_000, 100_000, 1_000_000]:
    ref, cur = rng.normal(0, 1, n), rng.normal(0.02, 1, n)       # a negligible 0.02-sd shift
    print(f"n={n:>9,}: KS p-value {stats.ks_2samp(ref, cur).pvalue:.2e}   PSI {psi(ref, cur):.4f}")

# --- Black-box shift estimation (BBSE): recover the new class prior from unlabeled predictions -------------
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
X, y = make_classification(n_samples=60_000, n_features=8, n_informative=5, weights=[0.95], random_state=0)
tr, va, pool = slice(0, 20_000), slice(20_000, 40_000), slice(40_000, None)
clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
yhat_va = clf.predict(X[va])
C = np.array([[np.mean(yhat_va[y[va] == j] == i) for j in (0, 1)] for i in (0, 1)])   # P(yhat=i | y=j)
# Production: resample the pool so positives go from about 5% to 15% (label shift: p(x|y) unchanged)
pos, neg = np.where(y[pool] == 1)[0], np.where(y[pool] == 0)[0]
prod_idx = np.r_[rng.choice(pos, 1500), rng.choice(neg, 8500)]
Xp = X[pool][prod_idx]
mu = np.array([np.mean(clf.predict(Xp) == i) for i in (0, 1)])
q_hat = np.linalg.solve(C, mu)
print(f"naive positive rate from predictions {mu[1]:.3f} | BBSE estimate {q_hat[1]:.3f} | truth 0.150")

# --- Behavioural tests for a toy sentiment scorer --------------------------------------------------------------
def sentiment(text):
    words = text.lower().split(); score = 0
    for i, w in enumerate(words):
        v = {"good": 1, "great": 2, "terrible": -2, "bad": -1}.get(w, 0)
        if i > 0 and words[i - 1] == "not": v = -v
        score += v
    return score
inv = all(sentiment(f"{name} said the food was great") == sentiment("Maria said the food was great") for name in ["Ion", "Aisha", "Wei"])
directional = sentiment("the service was not good") < sentiment("the service was good")
print("invariance (names) passes:", inv, "| directional (negation lowers score) passes:", directional)
```

---

## Pitfalls & misconceptions

- **Alerting on p-values over huge windows.** Everything becomes "significant".
- **One alert per feature.** You get alert fatigue, and the real issue gets ignored.
- **Treating all drift as concept drift.** Label shift has a cheap fix (re-estimate the priors), and pipeline bugs need a pipeline fix.
- **Monitoring inputs only.** Track predictions, proxies, and back-filled performance too.
- **Aggregate-only evaluation.** Slices and behavioural tests catch silent regressions.

## Cheat sheet

| Item | Rule / formula |
|---|---|
| Covariate / label / concept drift | $p(x)$ / $p(y)$ / $p(y\mid x)$ changes |
| BBSE | $\hat q = C^{-1}\hat\mu$, with $C_{ij} = P(\hat y=i\mid y=j)$ |
| Behavioural tests | minimum functionality, invariance, directional |
| Actionable alerts | effect sizes + importance weighting + persistence + grouping |
| Many alerts after a deploy | suspect the pipeline first |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Covariate, label, and concept drift in fraud detection.</summary>

Covariate: the transaction mix shifts toward new countries or devices, while the fraud pattern for a given transaction stays the same. Label: the fraud rate spikes during a holiday wave, with fraud looking the same as before. Concept: fraudsters change tactics, so features that meant "safe" now co-occur with fraud.
</details>

<details>
<summary>2. Monitoring quality when labels arrive weeks later.</summary>

Proxy signals (prediction distribution, positive rate, confidence, review overturn rates), importance-weighted input drift, BBSE prior estimates, partial early labels, and back-filled performance once the labels mature, aligned by prediction date.
</details>

<details>
<summary>3. Why drift tests fire constantly at scale, and how to make alerts actionable.</summary>

With huge $n$, tests detect negligible differences (the demo: p ≈ 1e-29 while PSI is below 0.001), and many features times daily tests means many false alarms. Use effect-size thresholds calibrated on history, weight by importance, require persistence, group the alerts, and page on prediction or performance signals.
</details>

<details>
<summary>4. Invariance vs directional tests for a sentiment model.</summary>

Invariance: changes that shouldn't matter (names, places, neutral paraphrases) must not change the prediction. Directional: changes with a known effect ("good" → "not good") must move the score in the expected direction.
</details>

<details>
<summary>5. Training-serving skew, and the test that catches it.</summary>

Feature values differ between the training and serving pipelines for the same entity. Test it by logging the serving features and predictions, recomputing them with the training pipeline for the same entities and timestamps, and comparing (a parity test).
</details>

<details>
<summary>6. 12 features alert after a deployment, accuracy unchanged.</summary>

First hypothesis: a pipeline or data change shipped with the deployment (a feature-code version, units, encoding, schema, source, or a reset reference window), not genuine world drift. Diff the deployment and compare the feature logs before and after.
</details>

## Where this leads

Next: [PROD-04 notes](04-distributed-training.md). Monitoring tells you *when* to retrain. For big models, retraining itself is the challenge: PROD-04 is the arithmetic of training across many GPUs.
