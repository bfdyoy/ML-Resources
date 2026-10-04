# EL-10 notes: Deep Learning for Tabular Data

[← Lesson EL-10](../../lessons/electives/10-tabular-deep-learning.md) · [All notes](../README.md) · [← EL-09 notes](09-audio-and-speech.md) · Back to: [All notes](../README.md)

> **Reading time** ≈ 45 min. **You need:** [CORE-05 notes](../core-ml/05-trees-and-ensembles.md) (GBMs), [CORE-07 notes](../core-ml/07-feature-engineering-pipelines-leakage.md) (categorical encodings), [DL-03 notes](../deep-learning/03-training-deep-networks.md) (training networks), and [MATH-03 notes](../math/03-probability-statistics.md) §D (paired comparisons).

---

## Where we are

Deep learning took over images, text, and audio, but on typical tables (rows of heterogeneous features: age, income, category codes), **gradient-boosted trees remain the default**. This note explains *why*, in terms of inductive bias, where neural nets genuinely help, and how to run a comparison that won't fool you.

---

## 1. Why trees usually win: three inductive biases

Grinsztajn et al. ("Why do tree-based models still outperform deep learning on tabular data?") isolated three reasons:

1. **Tabular targets are irregular.** The target often changes abruptly with a feature (thresholds: "income > 50k", "age ≥ 65"). Trees build piecewise-constant functions, exactly that. MLPs are biased toward **smooth** functions (the spectral bias, CV-04 §3.3), so they need lots of data to fit sharp jumps.
2. **Uninformative features hurt MLPs more.** Tables often have many irrelevant columns. Trees ignore them: a feature that never gives a good split is never used. An MLP mixes every input into every hidden unit from the first layer, so noise features degrade it.
3. **Tabular features are not rotation-invariant, and MLPs are.** Each column *means* something (age is age). Trees split along these meaningful axes. An MLP's first layer is a dense linear map, so it would learn the same function if you **rotated** the feature space. That's a symmetry tables *don't* have, so the MLP wastes capacity rediscovering the axes.
   The demo shows this directly: rotate the features randomly, and the GBM degrades while the MLP barely changes.

## 2. Making neural nets work on tables

- **Numeric preprocessing:** standardize, or better, a **quantile transform** (to Gaussian), which tames skew and outliers. Networks are scale-sensitive in a way trees aren't (CORE-05 §1.4). Learned **numerical embeddings** (piecewise-linear or periodic encodings of each scalar) help MLPs represent sharp, threshold-like effects. They directly attack bias 1.
- **Categorical features → entity embeddings:** map each category to a learned $d$-dimensional vector (DL-05 §1.1), trained end-to-end. Categories that behave similarly end up close together (stores in similar neighbourhoods, products in similar niches).
  **Reuse them:** feed the trained embedding vectors into a GBM as extra numeric features. That often beats one-hot or target encoding for high-cardinality columns.
- **Architectures:** a well-tuned **MLP with good preprocessing** (plus regularization) is a strong neural baseline. ResNet-style MLPs and **FT-Transformer** (which tokenizes each feature and applies attention across features) are good too. Fancier architectures rarely beat these by much.
- **Where neural nets genuinely help:** very large data sets, **multimodal rows** (tables plus text or images, embedded jointly), many high-cardinality categoricals, transfer or pretraining across tables, and end-to-end systems where the tabular model is one part of a larger network.

## 3. Tabular foundation models: TabPFN

TabPFN is a transformer **pretrained on millions of synthetic data sets**, drawn from a prior over data-generating processes (random causal graphs, functions, noise). At prediction time, you feed it **the whole training set plus the test points as one input**, and it predicts in a single forward pass. There's no gradient training on your data: it's **in-context learning** for tables.
In effect it approximates the Bayesian posterior predictive under its prior, which is why it shines on **small data sets**, where good priors matter most.

**When to pick it:** small tables (up to thousands of rows and a modest number of features). It needs no tuning, is very fast to try, and gives a strong baseline in seconds.
**When it fails or doesn't apply:** beyond its supported limits (large $n$ or many features or classes; check the current version's limits), on data far from its synthetic prior, or when the context length makes it slow or memory-hungry. Then GBMs win again.

---

## 4. Running a fair benchmark

A "my MLP beats XGBoost" claim needs all of the following:

- **The same splits** for every model (the same CV folds, or the same train/validation/test sets).
- **Equal tuning budgets:** the same number of trials or the same wall-clock time for each family. Comparing a tuned MLP with default XGBoost is the most common way to get a bogus win.
- **Several seeds and several data sets.** Report the mean ± std, and compare **paired** per split (MATH-03 §D3). Across many data sets, use rank-based tests (for example, a Wilcoxon signed-rank test, or average-rank / critical-difference plots).
- **No leakage:** preprocessing fitted inside the folds, for *both* models (CORE-01 §5).
- **Practical costs too:** training time, inference latency, tuning effort, robustness.

**Debug: the MLP beats XGBoost by 3% on one random split.** Before believing it, check:

1. the **variance**: repeat over 5–10 seeds and splits, and compute a paired CI. 3% may be within the noise;
2. the **tuning budgets**: was XGBoost tuned at all? Was early stopping used fairly for both?
3. **leakage**: was the MLP's preprocessing (scaler, target encoding, embeddings) fitted on the full data?
4. the **metric and split**: is the comparison on a held-out test set, not on the validation set used for tuning?
5. **other data sets**: does the win hold up?

```python
import numpy as np, warnings
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import ortho_group
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

# An "irregular" tabular task: a label driven by thresholds on a few features, plus uninformative columns
n = 3000
X = rng.normal(size=(n, 10))
y = (((X[:, 0] > 0.3) & (X[:, 1] < 0.5)) | (X[:, 2] > 1.2)).astype(int)
flip = rng.random(n) < 0.05; y[flip] = 1 - y[flip]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
gbm = HistGradientBoostingClassifier(random_state=0)
mlp = make_pipeline(QuantileTransformer(output_distribution="normal", n_quantiles=200),
                    MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=500, early_stopping=True, random_state=0))

R = ortho_group.rvs(10, random_state=0)                 # a random rotation of the feature space
for name, data in [("original axes", X), ("randomly rotated", X @ R)]:
    g = cross_val_score(gbm, data, y, cv=cv).mean(); m = cross_val_score(mlp, data, y, cv=cv).mean()
    print(f"{name:17s}: GBM {g:.3f}  MLP {m:.3f}")
```

```python
# --- Uninformative features: add 50 noise columns -------------------------------------------------------------------
Xn = np.c_[X, rng.normal(size=(n, 50))]
for name, model in [("GBM", gbm), ("MLP", mlp)]:
    print(f"{name}: 10 features {cross_val_score(model, X, y, cv=cv).mean():.3f} -> with 50 noise features {cross_val_score(model, Xn, y, cv=cv).mean():.3f}")

# --- A fair comparison: paired scores over repeated splits --------------------------------------------------------------
diffs = []
for seed in range(5):
    cv_s = StratifiedKFold(5, shuffle=True, random_state=seed)
    g = cross_val_score(gbm, X, y, cv=cv_s); m = cross_val_score(mlp, X, y, cv=cv_s)
    diffs.extend(m - g)                                    # the same folds for both models -> paired
diffs = np.array(diffs)
se = diffs.std(ddof=1) / np.sqrt(len(diffs))
print(f"MLP - GBM over 25 paired folds: mean {diffs.mean():+.4f}, range {diffs.min():+.3f}..{diffs.max():+.3f}, approx 95% CI ±{1.96*se:.4f}")
print("-> here the GBM wins on every fold, a real difference. When the gap is near zero, single splits point either way,")
print("   which is why you repeat with paired folds before believing a few-percent 'win'.")
```

---

## Pitfalls & misconceptions

- **Comparing a tuned neural net with default GBMs** (or the reverse).
- **One split, one seed, one data set.**
- **Feeding raw, skewed numeric features to an MLP.**
- **One-hot encoding huge categoricals for an MLP.** Use embeddings.
- **Assuming TabPFN scales to any data set size.** Check its limits.

## Cheat sheet

| Item | Rule |
|---|---|
| Why trees win | irregular targets, uninformative features, meaningful axes (no rotation invariance) |
| NN preprocessing | quantile or standard scaling; numerical embeddings; entity embeddings for categoricals |
| Reuse embeddings | trained category vectors → GBM features |
| TabPFN | in-context learning with a synthetic-data prior; small data; no tuning |
| Fair benchmark | same splits, equal tuning budget, several seeds and data sets, paired tests, no leakage |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Two inductive biases that help trees but hurt MLPs.</summary>

Trees fit piecewise-constant, threshold-like functions (the targets are irregular, while MLPs prefer smooth functions). Trees split along individual, meaningful axes and ignore useless features, while MLPs are rotation-invariant and mix every input, including noise columns. The demos show the rotation and noise effects.
</details>

<details>
<summary>2. What do entity embeddings learn, and how do you reuse them in a GBM?</summary>

A dense vector per category, placed so that categories with similar effects on the target are close together. After training the network, append the embedding vectors as numeric features to the GBM's input, in place of (or alongside) other encodings.
</details>

<details>
<summary>3. Why does scaling matter for neural nets but not trees?</summary>

Network optimization and initialization assume inputs of comparable scale (DL-03 §1). Unscaled features distort the gradients and the conditioning. Tree splits depend only on the order of the values, so monotone rescaling changes nothing (CORE-05 §1.4).
</details>

<details>
<summary>4. When to pick TabPFN, and when it fails.</summary>

Small tabular data sets within its supported size limits, when you want a strong result with no tuning. It fails or doesn't apply on large data sets, many features or classes beyond its limits, or data unlike its synthetic prior. Then use tuned GBMs.
</details>

<details>
<summary>5. What makes a tabular benchmark fair?</summary>

Identical splits, equal tuning budgets, several seeds and data sets, leakage-free preprocessing inside the folds for every model, paired statistical comparisons, and reporting the costs as well as the accuracy.
</details>

<details>
<summary>6. The MLP beats XGBoost by 3% on one random split.</summary>

Check the variance over seeds and splits with paired CIs, the equality of the tuning budgets (and early stopping), leakage in the MLP's preprocessing, that the evaluation was on an untouched test set, and replication on other data sets. Only then believe it.
</details>

## Where this leads

That's the last note. The whole curriculum, from linear algebra to diffusion and distributed training, now has written explanations. Go back to the [notes index](../README.md) to pick your next path, or revisit any note's cheat sheet when you need a refresher.
