# PROD-02 notes: MLOps in Practice

[← Lesson PROD-02](../../lessons/production/02-mlops-in-practice.md) · [All notes](../README.md) · [← PROD-01 notes](01-ml-system-design.md) · Next: [PROD-03 notes →](03-testing-monitoring-drift.md)

> **Reading time** ≈ 40 min. **You need:** [PROD-01 notes](01-ml-system-design.md) and [MATH-03 notes](../math/03-probability-statistics.md) §D (paired comparisons). Some familiarity with git and Docker helps.

---

## Where we are

PROD-01 designed the system. MLOps is the engineering that makes it **reproducible, testable, deployable, and maintainable**, so a model can be retrained and safely replaced by someone who isn't you, six months from now. This note covers the concepts and the decision rules; the lesson's resources cover the tools.

---

## 1. Reproducibility: what to log

A trained model is a function of **code + data + configuration + environment + randomness**. Log all five, or you can't reproduce it:

| Ingredient | Log | Tooling |
|---|---|---|
| Code | git commit hash (and refuse to train with uncommitted changes) | git |
| Data | an immutable snapshot or a **content hash** of the training data and the split definitions | DVC, lakeFS, table versions |
| Config | every hyperparameter and preprocessing option | config files (YAML), MLflow params |
| Environment | package versions (a lock file), the container image digest, hardware | `uv.lock`/`requirements.txt`, Docker |
| Randomness | seeds (and the knowledge that GPU nondeterminism may still differ slightly) | — |

Also log the **outputs**: metrics (overall *and* per slice), plots, and the model artifact with its signature (input schema). An experiment tracker (MLflow, W&B) ties these together, and a **model registry** versions the artifacts with stages (staging → production → archived) and lineage back to the run.

---

## 2. From notebook to pipeline

A notebook is for exploring. A pipeline is for repeating. Refactor into steps with clear inputs and outputs:
**ingest → validate → featurize → train → evaluate → register**. Each step is a tested function or script, configured rather than hand-edited, and orchestrated (Airflow, Prefect, Dagster, Kubeflow) so it can run on a schedule or a trigger, with retries and caching.

**Package the model with its preprocessing** (one scikit-learn `Pipeline`, or one exported graph), so training and serving share it. That's the skew fix from PROD-01 §2.

---

## 3. Tests an ML system needs

| Layer | Examples |
|---|---|
| **Code** (unit) | feature functions on hand-computed inputs; edge cases (nulls, empty, unicode) |
| **Data** | schema and types; ranges; null rates; uniqueness; distribution checks against a reference; no label leakage columns |
| **Model** | beats the baseline; meets per-slice minimums (no group silently regressing); **behavioural tests**: invariance (changing a name doesn't flip sentiment), directional expectations (adding "not" lowers the score), minimum functionality |
| **Pipeline / infra** | training runs end-to-end on a tiny sample; serialized model loads and gives identical predictions; serving latency and memory within budget; training/serving parity on logged examples |

Run fast tests in CI on every commit, and the full evaluation on every candidate model. [PROD-03 notes](03-testing-monitoring-drift.md) expands the data and behavioural tests.

---

## 4. Serving and packaging

- **API:** a thin web service (FastAPI or similar) that validates the request schema, loads the model **once** at startup, and returns predictions together with the model version.
- **Container:** pin the base image and dependencies, copy in only the artifact, run as non-root, and add health and readiness endpoints.
- **Rollout:** don't swap models blindly.
  - **Shadow:** the new model scores live traffic, but its outputs aren't used. You compare offline.
  - **Canary:** a small fraction of traffic first.
  - **A/B test:** a randomized comparison on the business metric.
  - Keep a **rollback** to the previous registry version one command away.

---

## 5. CI/CD and continual training

- **CI:** tests on code, data samples, and a smoke training run.
- **CD:** build the image, deploy to staging, run the evaluation and behavioural suites, then promote.
- **CT (continual training):** retrain automatically.

### 5.1 What triggers a retrain?

- **A schedule** (weekly): simple and predictable. Good when the data changes steadily.
- **Performance degradation** on fresh labeled data, below a threshold. This is the best signal, but labels can be delayed.
- **Significant drift** in important features or in the prediction distribution, as an *early warning* that prompts an investigation and possibly a retrain.
- **New data volume** (for example, 20% more labeled data since the last training).

### 5.2 What gates promotion? Champion vs challenger

The new model (challenger) must beat the current production model (champion) **on the same recent held-out data**:

- primary metric **non-inferior or better**, judged with a **paired** comparison and a confidence interval (MATH-03 §D3), not just a higher point estimate;
- no slice regressions beyond a tolerance; all behavioural tests pass; latency and size within budget;
- then shadow or canary checks in production before full rollout.

### 5.3 Drift alert, but performance unchanged: retrain?

**Not automatically.** First investigate:

- **Is the drifting feature important?** A shift in a feature the model barely uses doesn't matter.
- **Is it a data-quality bug** (a unit change, an upstream pipeline break), rather than real-world change? Then fix the pipeline. Retraining on broken data would make things worse.
- **Is performance *really* unchanged,** or are the labels just delayed? Check the proxy metrics and the label arrival times.
- **Is the shift in a region where the model still generalizes?**

If it's genuine, harmless covariate shift, keep monitoring and maybe update the reference window. Retraining has costs and risks of its own (a new model, new bugs). Do it on evidence.

```python
import hashlib, json, numpy as np, platform, sys
rng = np.random.default_rng(0)

# --- Content-hash a dataset so a run can record exactly which data it saw ---------------------------
data = rng.normal(size=(1000, 5)).round(6)
def data_hash(arr): return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
run_record = {
    "git_commit": "<git rev-parse HEAD>",
    "data_hash": data_hash(data),
    "params": {"model": "gbm", "learning_rate": 0.05, "n_estimators": 400, "seed": 42},
    "env": {"python": platform.python_version(), "numpy": np.__version__},
    "metrics": {"val_pr_auc": 0.71, "val_pr_auc_by_segment": {"new_users": 0.62, "returning": 0.74}},
}
print(json.dumps(run_record, indent=1)[:400], "...")
data2 = data.copy(); data2[0, 0] += 1e-6
print("hash changes when a single value changes:", data_hash(data) != data_hash(data2))
```

```python
# --- Promotion gate: paired bootstrap of challenger vs champion on the same recent data ---------------
n = 3000
y = rng.random(n) < 0.3
p_champ = np.clip(0.3 + 0.35 * (y - 0.3) + rng.normal(0, 0.18, n), 0, 1)
p_chall = np.clip(p_champ + 0.04 * (y - 0.3) + rng.normal(0, 0.03, n), 0, 1)   # slightly better
from sklearn.metrics import roc_auc_score
diffs = []
for _ in range(1000):
    i = rng.integers(0, n, n)
    diffs.append(roc_auc_score(y[i], p_chall[i]) - roc_auc_score(y[i], p_champ[i]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"AUC champion {roc_auc_score(y, p_champ):.4f}, challenger {roc_auc_score(y, p_chall):.4f}; 95% CI of the gain {lo:+.4f} .. {hi:+.4f}")
print("promote" if lo > 0 else "keep the champion (gain not established)")
```

---

## Pitfalls & misconceptions

- **"It's in MLflow, so it's reproducible."** Not without the data version and the environment.
- **Preprocessing living outside the model artifact.**
- **Promoting a challenger on a single higher number.** Use a paired CI and slice checks.
- **Auto-retraining on every drift alert.**
- **No rollback path.**

## Cheat sheet

| Item | Rule |
|---|---|
| Reproducibility | code hash + data hash/snapshot + config + environment lock + seeds |
| Pipeline | ingest → validate → featurize → train → evaluate → register |
| Tests | code, data, model (incl. behavioural and slices), infra (parity, latency) |
| Rollout | shadow → canary → A/B, with rollback ready |
| Retrain triggers | schedule, performance drop, investigated drift, data volume |
| Promotion gate | paired CI non-inferiority + slices + behaviour + budgets |

## Answer sketches for the lesson's self-check

<details>
<summary>1. The minimum to log for a reproducible experiment.</summary>

The code version (commit), the exact data (snapshot or content hash plus split definitions), all config and hyperparameters, the environment (lock file or container digest), the seeds, and the resulting metrics and artifacts.
</details>

<details>
<summary>2. Tests beyond code unit tests.</summary>

Data validation (schema, ranges, nulls, distributions, leakage), model tests (beats the baseline, slice minimums, behavioural invariance and directional tests), and infrastructure tests (end-to-end smoke training, serialization round-trip, training/serving parity, latency and memory).
</details>

<details>
<summary>3. Retrain triggers and promotion gates.</summary>

Triggers: a schedule, measured performance degradation, investigated significant drift, or new data volume. Gate: the challenger beats or matches the champion on the same recent data with a paired confidence interval, with no slice regressions, passing behavioural tests, within latency and size budgets, then shadow or canary before full rollout.
</details>

<details>
<summary>4. Drift alert, performance unchanged: retrain?</summary>

Not automatically. Check feature importance, data-quality bugs, label delays, and whether the shift matters for performance. Fix the pipeline if it's broken, keep monitoring if the shift is harmless, and retrain only on evidence of (impending) degradation.
</details>

## Where this leads

Next: [PROD-03 notes](03-testing-monitoring-drift.md). This note listed the tests and the monitoring. PROD-03 makes them rigorous: data and behavioural tests in depth, the math of the three kinds of drift, and how to build alerts that aren't noise.
