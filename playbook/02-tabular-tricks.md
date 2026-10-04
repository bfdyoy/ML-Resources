# Playbook 2: Tabular tricks

[← Playbook](README.md) · Previous: [Choosing algorithms](01-choosing-algorithms.md) · Next: [Deep learning tricks](03-deep-learning-tricks.md)

> These are the moves that separate a solid tabular model from a winning one. Most come from competition practice, summarized in NVIDIA's
> [Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data),
> and from the leakage lessons of [CORE-07](../lessons/core-ml/07-feature-engineering-pipelines-leakage.md).
> Every demo below is synthetic and seeded. It shows the *mechanism*, not a benchmark. **You need:** CORE-04, CORE-05, CORE-07.

Each trick follows the same format: **when** to use it → **why** it works → **how**, as a runnable demo → **the trap**.

---

## 1. Adversarial validation: can a model tell train from test?

**When:** before trusting *any* CV score, and whenever your CV and the leaderboard (or production) disagree.
**Why:** label train rows 0 and test rows 1, then train a classifier. If its AUC is ≈ 0.5, the two sets look alike and your CV is a fair proxy.
If the AUC is well above 0.5, the sets differ, and the classifier's feature importances tell you *which* columns differ.

```python
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error

rng = np.random.default_rng(0)
n = 2000
def make(shift):
    return pd.DataFrame({"age": rng.normal(40, 10, n), "income": rng.lognormal(10, 0.5, n),
                         "tenure": rng.exponential(3, n) + shift})
train, test = make(0.0), make(1.0)                    # only `tenure` drifts between the two
X_adv = pd.concat([train, test], ignore_index=True)
is_test = np.r_[np.zeros(n), np.ones(n)]

def adv_auc(cols):
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=0)
    p = cross_val_predict(clf, X_adv[cols], is_test, cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(is_test, p)

print("all columns:", round(adv_auc(list(X_adv)), 3))
for c in X_adv:
    print(f"{c:7s}", round(adv_auc([c]), 3))
print("without tenure:", round(adv_auc(["age", "income"]), 3))
```

The adversarial AUC is **0.664** with all columns. Column by column, `tenure` alone reaches 0.653, while `age` (0.531) and `income` (0.5) are near chance.
Dropping `tenure` brings the AUC down to 0.517. Now you know what to do: drop the column, transform it (a rank *within* each period), or build validation folds out of the training rows that look most like test.

**The trap:** an ID or timestamp column will always separate train from test. Drop such columns from the check, or you'll learn nothing.

## 2. Out-of-fold everything

**When:** any feature or model that is fitted *using the target*: target encoding, stacking, kNN-target features, pseudo-labels.
**Why:** if a row's feature was computed with that row's own label, the model learns to read the label back out. The training score soars, and the test score doesn't.
**How:** compute it out of fold. Fit on k−1 folds and fill in the k-th, exactly like [Lab 05 part C](../labs/05-trees-and-cross-validation/README.md).
The [noise-feature null](#3-the-noise-feature-null) below shows how big the effect is.

**The trap:** OOF features for training, but a feature fitted on *all* training rows for test. That's correct, but the test-time feature is slightly stronger (fitted on n rows, not n·(k−1)/k), so use enough folds (5–10).

## 3. The noise-feature null

**When:** you want to know whether a feature, or a feature-engineering step, adds signal or only adds capacity to overfit.
**Why:** a column of pure noise gives you the *null distribution* of "importance" and of CV gain. Anything that scores no better than noise is noise.

```python
n = 3000
cat = rng.integers(0, 500, n)                          # a 500-level category...
y_noise = rng.integers(0, 2, n)                        # ...and labels that have NOTHING to do with it

def in_sample_te(c, y, m=1.0):                         # the leaky way: each row's own label is in its mean
    s = pd.DataFrame({"c": c, "y": y}).groupby("c")["y"].agg(["sum", "count"])
    prior = y.mean()
    return ((s["sum"] + m * prior) / (s["count"] + m)).reindex(c).to_numpy()

def oof_te(c, y, m=1.0, k=5):
    out = np.empty(len(y))
    for tr, va in KFold(k, shuffle=True, random_state=0).split(c):
        s = pd.DataFrame({"c": c[tr], "y": y[tr]}).groupby("c")["y"].agg(["sum", "count"])
        prior = y[tr].mean()
        enc = (s["sum"] + m * prior) / (s["count"] + m)
        out[va] = pd.Series(c[va]).map(enc).fillna(prior).to_numpy()
    return out

for name, f in [("in-sample", in_sample_te(cat, y_noise)), ("out-of-fold", oof_te(cat, y_noise))]:
    auc = cross_val_score(LogisticRegression(), f[:, None], y_noise, cv=5, scoring="roc_auc").mean()
    print(f"{name:12s} CV AUC on pure-noise labels: {auc:.3f}")
```

With the in-sample encoding, a model scores a CV AUC of **0.718** on labels that are pure coin flips. The leak survives cross-validation, because the leak is
*inside the feature*, computed before the split. The out-of-fold encoding gives **0.490**, which is chance, as it should be.
The general recipe: add 1–3 random columns (Gaussian, uniform, a shuffled copy of a real column), and **drop every feature whose importance is below the best noise column's**.

**The trap:** permutation importance and tree split importance are both noisy. Repeat with several seeds before you drop anything.

## 4. Transform the target, not just the features

**When:** a skewed, positive regression target (prices, counts, durations), or a metric defined on a log or relative scale (RMSLE, MAPE).
**Why:** a model trained on squared error in raw units spends its capacity on the few huge values. Training on `log1p(y)` makes errors *relative*.
And if the metric is RMSLE, training on `log1p(y)` with RMSE *is* optimizing the metric.

```python
n = 4000
Xr = rng.normal(size=(n, 4))
y_price = np.exp(1.0 + Xr[:, 0] + 0.5 * Xr[:, 1] * Xr[:, 2] + rng.normal(scale=0.3, size=n))   # log-normal target
tr, te = np.arange(3000), np.arange(3000, n)
rmsle = lambda a, b: np.sqrt(np.mean((np.log1p(a) - np.log1p(b)) ** 2))

raw = HistGradientBoostingRegressor(random_state=0).fit(Xr[tr], y_price[tr]).predict(Xr[te])
logm = np.expm1(HistGradientBoostingRegressor(random_state=0).fit(Xr[tr], np.log1p(y_price[tr])).predict(Xr[te]))
print(f"RMSLE  raw target: {rmsle(np.clip(raw, 0, None), y_price[te]):.3f}   log1p target: {rmsle(logm, y_price[te]):.3f}")
```

Same model, same features: RMSLE drops from **0.355** to **0.255** just from training on `log1p(y)`.
Related moves: predict a **ratio** (price per square metre, then multiply back), or a **difference** (the change since last week instead of the level).

**The trap:** `expm1(mean of log y)` estimates the *median* of y, not its mean. If the business needs totals (e.g. summed revenue), multiply by a smearing factor,
or train on raw y with a Poisson/Tweedie loss.

## 5. Monotone constraints: encode what you already know

**When:** domain knowledge says "more X never lowers the prediction": price vs size, risk vs days past due, demand vs discount.
**Why:** it removes implausible wiggles, it makes the model easier to defend to stakeholders, and it often *improves* generalization on small or noisy data.
**How:** `HistGradientBoostingRegressor(monotonic_cst=[1, 0, -1, ...])` (+1 increasing, −1 decreasing, 0 free). XGBoost and LightGBM have the same option.

```python
n = 300
x_size = rng.uniform(0, 10, n)
y_mono = np.log1p(x_size) + rng.normal(scale=0.4, size=n)          # truly increasing, small and noisy
grid = np.linspace(0, 10, 200)[:, None]
free = HistGradientBoostingRegressor(random_state=0).fit(x_size[:, None], y_mono).predict(grid)
mono = HistGradientBoostingRegressor(random_state=0, monotonic_cst=[1]).fit(x_size[:, None], y_mono).predict(grid)
truth = np.log1p(grid[:, 0])
for name, f in [("free", free), ("monotone", mono)]:
    print(f"{name:9s} downward steps: {np.sum(np.diff(f) < -1e-9):3d}   RMSE vs truth: {np.sqrt(np.mean((f - truth) ** 2)):.3f}")
```

The unconstrained model has **38** downward steps on a curve that only ever rises. The constrained one has **0**, and is also closer to the truth (RMSE **0.093** vs **0.153**).

**The trap:** a wrong constraint is worse than none. Constrain only what you'd defend in a code review, and check it on partial-dependence plots.

## 6. Nearest-neighbour features

**When:** the label of similar rows is informative ("customers like this one churned"), or the data has a geography or a similarity structure that trees split clumsily.
**Why:** a tree carves the space into axis-aligned boxes. The out-of-fold mean target of the k nearest neighbours hands it a smooth, multivariate signal in a single column.
**How:** standardize, fit `NearestNeighbors` on the training folds only, and for each held-out row compute the mean target of its k neighbours (and maybe the distance to them).
It's [§2 out-of-fold everything](#2-out-of-fold-everything) again, with a different statistic.

**The trap:** computing neighbours among *all* rows, including the row itself (distance 0, so its own label leaks). Always query from outside the fitted fold.

## 7. Boost the residuals of a simple model

**When:** the target has a strong, near-linear trend (prices over time, dose–response curves), and the test set may extend *beyond* the training range.
**Why:** trees predict a constant outside the range they've seen, so they can't extrapolate. Let a linear model carry the trend, and let the GBM learn only what's left over (nonlinearities, interactions).

```python
n = 2000
x_time = rng.uniform(0, 10, n)
z = rng.normal(size=n)
y_trend = 3 * x_time + 2 * np.sin(2 * z) + rng.normal(scale=0.5, size=n)
Xt = np.c_[x_time, z]
inside, future = x_time < 8, x_time >= 8                             # train on the past, test on the "future"

gbm = HistGradientBoostingRegressor(random_state=0).fit(Xt[inside], y_trend[inside])
lin = LinearRegression().fit(Xt[inside], y_trend[inside])
res = HistGradientBoostingRegressor(random_state=0).fit(Xt[inside], y_trend[inside] - lin.predict(Xt[inside]))
hybrid = lin.predict(Xt[future]) + res.predict(Xt[future])
rmse = lambda p: np.sqrt(mean_squared_error(y_trend[future], p))
print(f"future RMSE  GBM: {rmse(gbm.predict(Xt[future])):.2f}  linear: {rmse(lin.predict(Xt[future])):.2f}  linear + GBM on residuals: {rmse(hybrid):.2f}")
```

On the unseen future range, the GBM alone scores an RMSE of **3.69** (it flat-lines), the linear model **1.35** (it misses the sine), and **linear + GBM on its residuals scores 0.54**.
The same idea, in other clothes: a physics or business formula as the base model, a seasonal-naive forecast as the base, or a pretrained model's prediction as a feature.

**The trap:** fit the base model inside each CV fold too, or its residuals will leak.

## 8. Blend by hill climbing on out-of-fold predictions

**When:** you have several decent models (a GBM, a linear model, a kNN, an MLP) and want the best combination without overfitting the blend.
**Why:** different model families make different errors, so averaging cancels some of them. Greedy **hill climbing** (Caruana's ensemble selection) starts from the best model and repeatedly adds, *with replacement*, the model that most improves the OOF score.
It yields sparse, positive weights and rarely overfits. **Rank averaging** (average the ranks, not the probabilities) helps when models are calibrated differently and the metric is AUC.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer

n = 3000
Xb = rng.normal(size=(n, 6))
logit = Xb[:, 0] - Xb[:, 1] + np.sin(2 * Xb[:, 2]) + Xb[:, 3] * Xb[:, 4]
yb = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
models = {
    "gbm": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=0),
    "logreg": make_pipeline(StandardScaler(), LogisticRegression()),
    "spline-logreg": make_pipeline(SplineTransformer(), LogisticRegression(max_iter=2000)),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(50)),
}
oof = {k: cross_val_predict(m, Xb, yb, cv=5, method="predict_proba")[:, 1] for k, m in models.items()}
for k, p in oof.items():
    print(f"{k:14s} OOF AUC {roc_auc_score(yb, p):.4f}")

def hill_climb(oof, y, steps=20):
    names = list(oof)
    chosen = [max(names, key=lambda k: roc_auc_score(y, oof[k]))]
    for _ in range(steps):
        cand = {k: roc_auc_score(y, np.mean([oof[c] for c in chosen + [k]], axis=0)) for k in names}
        best = max(cand, key=cand.get)
        if cand[best] <= roc_auc_score(y, np.mean([oof[c] for c in chosen], axis=0)):
            break
        chosen.append(best)
    return {k: chosen.count(k) / len(chosen) for k in set(chosen)}, roc_auc_score(y, np.mean([oof[c] for c in chosen], axis=0))

weights, blend_auc = hill_climb(oof, yb)
print("weights:", {k: round(v, 2) for k, v in sorted(weights.items())}, " blend OOF AUC:", round(blend_auc, 4))
```

The best single model here is the kNN (OOF AUC **0.7978**), just ahead of the GBM (**0.7933**). The hill-climbed blend picks kNN and GBM in a 2:1 ratio and reaches **0.8052**,
better than either alone, because a smooth local model and a box-carving tree make different mistakes. The two logistic models never get picked: they add nothing the others don't already have.

**The trap:** a blend optimized on OOF predictions is itself a fitted model. Judge the final blend on a hold-out that hill climbing never saw, or with nested CV.

## 9. Pseudo-labelling (carefully)

**When:** a large unlabeled set (often the test set itself) from the same distribution, and a model that's already good.
**Why:** confident predictions on unlabeled rows become extra training data. That helps when the unlabeled rows cover regions the labeled set doesn't.
**How:** train, predict on the unlabeled rows, and keep only the most confident ones (or use soft labels). Retrain on labeled + pseudo-labeled data, **inside each CV fold**
(the pseudo-labels for fold k must come from a model that never saw fold k's labels).

**The trap:** the model confirms its own mistakes (confirmation bias), and a careless CV setup leaks the validation labels through the pseudo-labels.

## 10. Quantile regression for honest ranges

**When:** stakeholders need a range ("between 40 and 70 units"), or the costs of over- and under-predicting differ (see [Outside the box §6](05-outside-the-box.md#6-asymmetric-costs-change-the-loss-not-the-threshold)).
**How:** `HistGradientBoostingRegressor(loss="quantile", quantile=0.1)` and `quantile=0.9` give an 80% band. Check the band's empirical coverage on held-out data,
or wrap a point model in conformal intervals for a guarantee ([CORE-11](../lessons/core-ml/11-uncertainty-calibration-conformal.md), [Lab 12](../labs/12-calibration-conformal-drift/README.md)).

## 11. Group-by aggregate features

**When:** rows belong to entities (users, stores, devices), and the entity's history matters more than the single row.
**How:** for each row, aggregate its entity's *past* rows: counts, means, the time since the last event, the deviation from the entity mean (`x − mean_by_user(x)`).
[Lab 02](../labs/02-pandas-wrangling/README.md) builds these with `groupby().transform` and per-group `shift`.

**The trap:** including the current row or *future* rows in the aggregate. Use `shift(1)` before rolling, and compute aggregates inside the CV split for time-ordered data.

---

## Cheat sheet

| Situation | First trick to try |
|---|---|
| CV and leaderboard disagree | Adversarial validation (§1) |
| A high-cardinality category | Out-of-fold target encoding (§2), then check it against the noise null (§3) |
| Skewed positive target | Train on `log1p(y)` (§4) |
| "Can't be decreasing" domain rule | Monotone constraints (§5) |
| Test range beyond train range | Linear base + GBM on its residuals (§7) |
| Several models of similar quality | Hill-climbing blend on OOF predictions (§8) |
| A range is needed, not a point | Quantile GBM or conformal (§10) |
| Entity-level data | Past-only group aggregates (§11) |
