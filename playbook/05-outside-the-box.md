# Playbook 5: Outside the box

[← Playbook](README.md) · Previous: [LLM & retrieval tricks](04-llm-and-retrieval-tricks.md) · Next: [Debugging playbook](06-debugging-playbook.md)

> These are ideas that aren't in the algorithm chapters: using a tool built for one job to do a different one, or reframing the problem until it gets easy.
> Each comes with a small synthetic demo that shows *why* it works. **You need:** Path 1 through CORE-07. Some sections refer to DL and GEN lessons.

---

## 1. The no-model baselines

**The idea:** before any model, compute what you'd get from **no model at all**: the training mean, the most frequent class, the last observed value, the same day last week.
Then report every model as an improvement *over that*.
**Why it's outside the box:** people skip it because it feels trivial. Yet in forecasting, recommendations and anything with a strong trend, the naive baseline is often very hard to beat, and sometimes a fancy model *doesn't*.

```python
import gzip
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor, export_text

rng = np.random.default_rng(0)
walk = np.cumsum(rng.normal(size=3000))                   # a random walk, like many prices
lags = np.column_stack([walk[5 - L:-L] for L in range(1, 6)])     # the 5 previous values
target = walk[5:]
split = 2000
mse = lambda p: np.mean((p - target[split:]) ** 2)
gbm = HistGradientBoostingRegressor(random_state=0).fit(lags[:split], target[:split])
print(f"MSE  training mean: {mse(np.full(len(target) - split, target[:split].mean())):.1f}   "
      f"last value: {mse(lags[split:, 0]):.3f}   ridge on 5 lags: {mse(Ridge().fit(lags[:split], target[:split]).predict(lags[split:])):.3f}   "
      f"GBM on 5 lags: {mse(gbm.predict(lags[split:])):.3f}")
```

On a random walk, "predict the training mean" is hopeless (MSE **2853.2**), and "tomorrow = today" scores **0.964**, about the noise level. Ridge on five lags only matches it (**0.967**).
The GBM does far *worse* (**506.759**), because the test values wander outside the training range and trees can't extrapolate ([tabular tricks §7](02-tabular-tricks.md#7-boost-the-residuals-of-a-simple-model)).
Without the naive baseline in the table, a "GBM with lag features" could look like a reasonable result.

## 2. A classifier is a two-sample test

**The idea:** "do these two datasets come from the same distribution?" is a hard statistical question in many dimensions. **Train a classifier to tell them apart.**
If its held-out AUC is ≈ 0.5, it can't, so the samples look the same. If the AUC is well above 0.5, they differ, and the classifier tells you *where*.
This is the classifier two-sample test, and you've already met it as [adversarial validation](02-tabular-tricks.md#1-adversarial-validation-can-a-model-tell-train-from-test).
**Why it's outside the box:** classical per-feature tests (Kolmogorov–Smirnov, PSI) look at one column at a time, so they miss changes in the *relationship* between columns.

```python
n = 3000
cov_a, cov_b = [[1, 0], [0, 1]], [[1, 0.6], [0.6, 1]]      # same marginals, different correlation
A_s = rng.multivariate_normal([0, 0], cov_a, n)
B_s = rng.multivariate_normal([0, 0], cov_b, n)
print("per-column KS p-values:", [round(ks_2samp(A_s[:, j], B_s[:, j]).pvalue, 3) for j in range(2)])
Xs, src = np.r_[A_s, B_s], np.r_[np.zeros(n), np.ones(n)]
p = cross_val_predict(HistGradientBoostingClassifier(random_state=0), Xs, src, cv=5, method="predict_proba")[:, 1]
print(f"classifier two-sample test AUC: {roc_auc_score(src, p):.3f}")
```

Both columns pass the KS test (p = **0.143** and **0.087**: nothing significant column by column). The classifier separates the two samples with an AUC of **0.641**, because it sees the correlation change.
Other uses of the same trick: **drift monitoring** (last week vs this week), **leak hunting** (can a model predict the target from the row ID or the file name? If it can, something leaks),
and **synthetic data quality** (can a model tell synthetic rows from real ones?).

## 3. Compression is a similarity measure

**The idea:** if two texts share patterns, compressing them *together* takes fewer extra bytes than compressing them apart. The **normalized compression distance**
`NCD(x, y) = (C(xy) − min(C(x), C(y))) / max(C(x), C(y))`, with C = the length after gzip, is a parameter-free similarity. Plug it into kNN and you have a classifier with no training and no embeddings.

```python
C = lambda s: len(gzip.compress(s.encode("utf-8")))
def ncd(a, b):
    ca, cb = C(a), C(b)
    return (C(a + " " + b) - min(ca, cb)) / max(ca, cb)

train_txt = {
    "en": ["the weather is nice today and we are going to the park", "she said that the meeting was moved to thursday afternoon",
           "this is one of the best books I have read in a long time", "please remember to bring your passport to the airport"],
    "ro": ["vremea este frumoasă astăzi și mergem în parc", "ea a spus că ședința a fost mutată joi după-amiază",
           "aceasta este una dintre cele mai bune cărți pe care le-am citit", "te rog să nu uiți pașaportul când mergi la aeroport"],
    "py": ["for i in range(len(items)): total += items[i]", "def train(model, loader): return [model(x) for x, y in loader]",
           "import numpy as np; x = np.zeros((3, 4)); print(x.shape)", "if score > best: best, best_epoch = score, epoch"],
}
test_txt = [("we will meet at the station before the train leaves", "en"), ("mâine mergem la munte cu prietenii noștri", "ro"),
            ("while not done: state, reward, done = env.step(action)", "py"), ("the children are playing in the garden after school", "en"),
            ("cartea aceasta mi-a plăcut foarte mult", "ro"), ("with open(path) as f: lines = f.read().splitlines()", "py")]
pool = [(t, lab) for lab, ts in train_txt.items() for t in ts]
acc = lambda pred: np.mean([p_ == y_ for p_, (_, y_) in zip(pred, test_txt)])

pred_knn = [min(pool, key=lambda tl: ncd(x, tl[0]))[1] for x, _ in test_txt]           # per-example 1-NN on NCD
corpus = {lab: " ".join(ts) for lab, ts in train_txt.items()}                           # one "document" per class
pred_cls = [min(corpus, key=lambda lab: C(corpus[lab] + " " + x) - C(corpus[lab])) for x, _ in test_txt]
print("NCD 1-NN on single lines:", pred_knn, acc(pred_knn))
print("extra bytes given each class's text:", pred_cls, acc(pred_cls))
```

Two versions, two very different results. **Per-example NCD with 1-NN fails** on these one-line texts: it predicts "en" for all six (accuracy **0.33**).
gzip's fixed header and its minimum block overhead swamp the few bytes of shared pattern in a single short line.
**Compressing against each class's whole text works:** ask how many *extra* bytes the new line costs once gzip has already seen all of a class's examples.
That labels **all six** correctly, with no learned parameters at all.

The [gzip + kNN paper](https://arxiv.org/abs/2212.09410) (an ACL 2023 version followed) made per-example NCD + kNN famous on topic classification of longer texts. Then came the twist:
[Ken Schutte's analysis](https://kenschutte.com/gzip-knn-paper/) found that the paper's k = 2 evaluation counted a tie as correct whenever *either* tied label was right.
That makes it closer to a top-2 accuracy, which inflated the headline numbers. **Three lessons:** compression is a real, training-free baseline;
how you *use* it (per-example vs per-class) matters as much as the idea; and **audit the evaluation code before you trust a surprising result.**

## 4. Reframe the problem

**The idea:** the task as stated is rarely the only possible formulation. Different framings have different difficulty, data needs, and metrics:
- **Regression → classification:** "will demand exceed capacity?" may be all the decision needs, and it is easier to learn than the exact demand.
- **Classification → ranking:** if you'll act on the top 100 cases, optimize and evaluate precision@100, not accuracy.
- **Point → distribution:** predict quantiles (§6) or prediction sets ([CORE-11](../lessons/core-ml/11-uncertainty-calibration-conformal.md)) when the decision depends on risk.
- **Level → change:** predict the difference or the ratio to a baseline, not the raw value ([tabular tricks §4](02-tabular-tricks.md#4-transform-the-target-not-just-the-features)).
- **Supervised → self-supervised:** if labels are scarce but raw data is plentiful, pretrain on a task where the data *is* the label (masking, next token, contrastive), then fine-tune ([CV-02](../lessons/vision/02-vision-transformers-self-supervised.md)).
- **One model → per-segment rules + model:** if 30% of cases follow a known rule, hard-code the rule and let the model handle the rest.

Ask *"what decision will this prediction drive?"* The answer usually picks the framing.

## 5. Train a model on your model's errors

**The idea:** average metrics hide **slices** where the model fails, for example one region, one device type, or one age band. Instead of guessing at slices,
**fit a small, interpretable model that predicts the error** from the input features. Its splits *are* the failing slices.

```python
n = 4000
df = pd.DataFrame({"age": rng.uniform(18, 80, n), "income": rng.normal(50, 15, n),
                   "region": rng.integers(0, 4, n), "device": rng.integers(0, 3, n)})
noise = np.where((df.region == 2) & (df.age > 60), 3.0, 0.5)          # a hidden slice the model handles badly
y_err = 0.05 * df.age + 0.02 * df.income + rng.normal(size=n) * noise
oof = cross_val_predict(LinearRegression(), df, y_err, cv=5)
abs_err = np.abs(y_err - oof)
tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=100, random_state=0).fit(df, abs_err)
print(export_text(tree, feature_names=list(df.columns), decimals=2))
```

Two splits are enough for the error tree to recover the hidden slice: **age > 60.18 and region > 1.5** (region 2 or 3, with region 2 driving it). The mean absolute error there is **1.38**, about 3.4× the ≈ 0.4 of every other leaf.
Next steps: look at examples from that slice, collect more data there, add a feature, or route those cases to a human.
The same trick works on LLM eval failures: tag each failure, then fit a model on the tags.

## 6. Asymmetric costs: change the loss, not the threshold

**The idea:** if under-forecasting costs 4 EUR per unit (a lost sale) and over-forecasting costs 1 EUR (storage), the best forecast **is not the mean**.
It's the quantile at `c_under / (c_under + c_over) = 0.8`, the classic newsvendor solution. Train with the **pinball (quantile) loss** at that level instead of squared error.

```python
demand = rng.lognormal(3, 0.6, size=100_000)
def cost(q, y=demand, c_under=4.0, c_over=1.0):
    return np.mean(c_under * np.maximum(y - q, 0) + c_over * np.maximum(q - y, 0))
for name, q in [("mean", demand.mean()), ("median", np.median(demand)), ("0.8-quantile", np.quantile(demand, 0.8))]:
    print(f"order {name:13s} = {q:6.1f} units -> expected cost {cost(q):.2f} EUR")
```

Ordering the mean demand (**24.0** units) costs **28.38** EUR per period on average, and the median (**20.1**) costs **33.15** EUR. Ordering the 0.8-quantile (**33.2** units) costs **24.62** EUR, the minimum.
For a conditional forecast, use `HistGradientBoostingRegressor(loss="quantile", quantile=0.8)`.
The classification analogue is the cost-based threshold from [Lab 04](../labs/04-logistic-regression-metrics/README.md). In both cases, the costs belong in the objective.

## 7. Random projections are almost free dimensionality reduction

**The idea:** multiply your data by a **random** Gaussian matrix to cut the dimension from d to k. No fitting at all.
The Johnson–Lindenstrauss lemma says that all pairwise distances between n points are preserved within a factor 1 ± ε once k is on the order of `log(n) / ε²`. **k depends on the number of points, not on d.**

```python
n_pts, d = 300, 10_000
Xh = rng.normal(size=(n_pts, d))
D0 = pdist(Xh)
for k in [50, 200, 1000]:
    R = rng.normal(size=(d, k)) / np.sqrt(k)
    ratio = pdist(Xh @ R) / D0
    print(f"k = {k:5d}: distance ratio in [{ratio.min():.3f}, {ratio.max():.3f}], typical error {np.std(ratio):.3f}")
```

Going from 10,000 to 1,000 dimensions keeps every one of the 44,850 pairwise distances within about **±10%** (ratios **0.897–1.105**). Even k = 200 keeps them within **±25%** (ratios 0.784–1.242), with a typical error of **0.052**.
Uses: speeding up kNN and clustering on huge sparse vectors, sketching embeddings, quick privacy-ish obfuscation, and the hashing trick for text features.
(PCA would preserve more variance, but it needs a fit on the data, which takes time and can leak through the CV split.)

## 8. Frozen embeddings + a linear model is a strong baseline for anything

**The idea:** for images, text, audio, and even molecules or code, a pretrained encoder (CLIP/SigLIP, DINOv2, a sentence-embedding model) turns raw data into vectors.
A **logistic regression on those frozen vectors** is often within a few points of full fine-tuning, trains in seconds on a laptop, and needs only hundreds of labels.
**Why it's outside the box:** people jump to fine-tuning. Start with the linear probe instead: it's the baseline your fine-tune has to beat, and sometimes it's the product.
**How:** embed once and cache. Then use `LogisticRegression` (or kNN for few-shot), and use the same embeddings for search, deduplication (near-identical vectors), outlier detection, and clustering your errors (§5).
See [CV-02](../lessons/vision/02-vision-transformers-self-supervised.md) for linear probes and [CV-03](../lessons/vision/03-clip-vision-language-models.md) for zero-shot.

## 9. Shuffle the labels: a lie detector for your pipeline

**The idea:** run your **entire** pipeline (feature selection, encoding, tuning, CV) with the labels randomly permuted. Every honest pipeline must score **chance**.
If it scores better, information is leaking from the labels into the features *outside* the CV loop.

```python
n, p_feat = 100, 5000
Xn = rng.normal(size=(n, p_feat))                      # pure noise features
y_shuf = rng.integers(0, 2, n)                         # labels with no relationship to X (as if shuffled)

leaky_X = SelectKBest(f_classif, k=20).fit_transform(Xn, y_shuf)        # feature selection on ALL rows, before CV
leaky = cross_val_score(LogisticRegression(), leaky_X, y_shuf, cv=5).mean()
honest = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression()), Xn, y_shuf, cv=5).mean()
print(f"CV accuracy on meaningless labels  leaky pipeline: {leaky:.2f}   honest pipeline: {honest:.2f}")
```

Selecting features on all rows *before* cross-validation gives **0.87** accuracy on labels that mean nothing. The same steps inside a `Pipeline`, refitted per fold, give **0.48** (chance, plus the usual noise of 100 samples).
This is the classic error in genomics papers, described in ESL §7.10.2. A label-shuffle run costs one extra run and catches it every time.

## 10. Plant a signal you know

**The idea:** to test a method (a feature-importance tool, a causal estimator, an anomaly detector, a drift monitor), **simulate data where you know the answer**, and check that the method finds it.
**Why it's outside the box:** we usually test methods on real data, where the truth is unknown. That's exactly why we can't tell whether they work.
Examples: add a column that is *exactly* the target plus noise (importance tools must rank it first); inject a shift into a copy of last week's data (the drift monitor must fire);
simulate a treatment effect of +2 (your estimator must find about 2). The demos on these playbook pages are all planted signals.

## 11. Route by confidence: cascades

**The idea:** a cheap model answers the easy cases, and only uncertain cases go to an expensive model (or a human).
If 80% of traffic is easy, the cost falls by almost 80%, and accuracy hardly changes.
**How:** pick the confidence threshold on validation data so that the cheap model's accuracy *on the cases it keeps* meets the bar.
Calibrated probabilities or conformal set size ([CORE-11](../lessons/core-ml/11-uncertainty-calibration-conformal.md)) make good routing signals. For LLMs: a small model, or a cached answer, first.

---

## Cheat sheet

| Question | Outside-the-box answer |
|---|---|
| Is my model any good? | Compare with no-model baselines (§1) |
| Did the data change? Is there a leak? | Train a classifier to tell the datasets apart (§2) |
| Need a similarity, and have no model? | Compression distance (§3), random projections (§7) |
| Stuck on the problem as stated? | Reframe it around the decision (§4) |
| Where does my model fail? | Fit a shallow tree on its errors (§5) |
| Costs are asymmetric? | Quantile/pinball loss at `c_u / (c_u + c_o)` (§6) |
| Images, text, or audio with few labels? | Frozen embeddings + logistic regression (§8) |
| Is my pipeline honest? | Shuffled labels must score chance (§9) |
| Does this method work at all? | Plant a known signal (§10) |
| Too slow or expensive? | Confidence cascades (§11) |
