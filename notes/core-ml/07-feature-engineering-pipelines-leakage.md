# CORE-07 notes: Feature Engineering, Pipelines & Leakage

[← Lesson CORE-07](../../lessons/core-ml/07-feature-engineering-pipelines-leakage.md) · [All notes](../README.md) · [← CORE-06 notes](06-unsupervised-learning.md) · Next: [CORE-08 notes →](08-interpretability-and-responsible-ml.md)

> **Reading time** ≈ 50 min. **You need:** [CORE-01 notes](01-ml-workflow-end-to-end.md) §5 (pipelines) and [CORE-04 notes](04-generalization-validation-regularization.md) §2 (CV).

---

## Where we are

On tabular problems, **features usually matter more than the choice of algorithm**. This lesson covers how to turn raw columns into features a model can use,
and how to catch **leakage**, the bug that makes a project look brilliant in validation and fail in production.

---

## 1. Categorical variables

| Encoding | How | Good for | Watch out |
|---|---|---|---|
| One-hot | One 0/1 column per level | Linear models, few levels | Dimension explodes with many levels |
| Ordinal | Map levels to integers | Trees; truly ordered levels (S < M < L) | Linear models read the integers as distances |
| Target (mean) encoding | Replace each level with a statistic of $y$ for that level | High cardinality | **Leaks** unless done out-of-fold |
| Hashing | Hash the level into one of $m$ buckets | Huge or unbounded cardinality, streaming | Collisions (usually harmless) |
| Learned embedding | A dense vector per level, trained with the model | Neural nets with many levels | Needs enough data per level |

### 1.1 Target encoding, done safely

The naive version replaces category $c$ with the mean of $y$ over the training rows in $c$. Two problems:

1. **Rare levels are noisy.** A level seen once gets exactly that row's label.
2. **It leaks.** Each row's encoding *contains its own label*. A model can learn "encoding high → label high", which is perfect in training and useless on new data.

**Fix 1: smoothing (shrinkage toward the global mean).** With $n_c$ rows in level $c$, level mean $\bar y_c$, global mean $\bar y$, and a strength $m$:

```math
\text{enc}(c) = \frac{n_c\,\bar y_c + m\,\bar y}{n_c + m} .
```

This is a Bayesian posterior mean, with the prior "worth" $m$ pseudo-observations. Frequent levels keep their own mean, and rare ones fall back toward the global mean.
It's the same shrinkage idea as ridge (CORE-04 §3).

**Fix 2: out-of-fold (OOF) encoding.** Split the training data into $k$ folds, and encode each fold's rows using statistics computed **only on the other folds**. No row ever sees its own label.
Test data is encoded with statistics from the full training set. scikit-learn's `TargetEncoder` does exactly this (cross-fitting) inside `fit_transform`.

The demo below uses a 1,000-level categorical with **no relationship** to the target. The naive encoding gives a training AUC of about 0.75 on pure noise. OOF encoding correctly gives about 0.5.

### 1.2 50,000 levels?

Use hashing (fixed memory), OOF target encoding, a frequency encoding (the count of each level), or native categorical handling (CatBoost uses *ordered* target statistics, LightGBM
has categorical splits). In a neural net, use an embedding. Group the rare levels into an "other" bucket first.

---

## 2. Numeric features

- **Scaling:** standardization for linear models, kNN, SVMs, neural nets, PCA. Trees don't need it (CORE-05 §1.4).
- **Skewed values** (incomes, counts): `log1p` or a quantile transform. This helps linear models and neural nets.
- **Ratios and differences** often carry the signal: debt/income, price/sq-ft, time since the last purchase.
- **Interactions:** a linear model can't represent "the effect of $x_1$ depends on $x_2$" unless you give it the product $x_1 x_2$. Trees find interactions on their own, through nested splits.
- **Binning:** occasionally useful for interpretability, but it usually throws information away.

### 2.1 Cyclical features: hours, weekdays, months

Encode hour $h$ as the integer 0–23 and the model thinks 23:00 and 00:00 are far apart. Map it onto a circle instead:

```math
h \mapsto \Big(\sin\frac{2\pi h}{24},\ \cos\frac{2\pi h}{24}\Big).
```

Now the Euclidean distance between 23:00 and 00:00 is the same as between 00:00 and 01:00 (about 0.26). You need both sine and cosine: sine alone gives 06:00 and 18:00 opposite values but maps 03:00 and 09:00 to the same one.

### 2.2 Dates and aggregations

From a timestamp you can extract the day of week, whether it's a holiday, and "days since X". **Group-by aggregates** (a customer's average order value, the number of logins in the last 7 days) are often the
strongest features of all, and the easiest to leak. Each aggregate must use **only data from before the prediction time**, never the current row's own target.

### 2.3 Missing values

Why is a value missing? *Completely at random* (a sensor glitch), *at random given other features*, or *because of its value* (high earners skip the income question).
In the last case, **missingness itself is informative**: add an `is_missing` indicator alongside the imputed value. Gradient-boosting libraries handle `NaN` natively by learning which branch missing values should go to.
Always fit imputers on the training fold only, inside the pipeline.

---

## 3. Text, the classic way: TF-IDF

Count words (a **bag of words**) and down-weight the words that appear everywhere:

```math
\text{tfidf}(t, d) = \text{tf}(t, d)\cdot\log\frac{1 + N}{1 + \text{df}(t)} + \text{tf}(t,d)
```

This is scikit-learn's smoothed form, $\text{tf}\cdot(\text{idf} + 1)$. Each document's vector is then L2-normalized. Here $\text{df}(t)$ is the number of documents containing term $t$.
"the" appears everywhere, so its idf is small. A rare, specific term gets a large weight. TF-IDF + logistic regression is still a strong text baseline. In [GEN-05](../llms-genai/05-retrieval-engineering.md), BM25 refines the same idea for search.

---

## 4. Leakage: a field guide

Leakage means **the model was trained or validated with information it won't have at prediction time.** There are three kinds:

| Kind | Example | Symptom | Prevention |
|---|---|---|---|
| **Target leakage** | `days_since_account_closed` in a churn model: it's only defined *because* the customer churned | One feature dominates the importance ranking, and the score is "too good" | For every feature, ask: *would I know this at prediction time?* |
| **Train/test contamination** | Scaling, imputing, selecting features, or target-encoding on the full data set | CV beats production | Every `fit` inside a `Pipeline`, inside CV |
| **Temporal leakage** | A random split on time-ordered data; aggregates that include future rows | A random split beats a time split by a lot | Time-based splits, point-in-time feature computation |

**The time-split test** (the lesson's debug question): if a random-split CV gives 0.92 and a time-based split gives 0.64, the model is exploiting information from the future: leaky features,
or near-duplicate rows across time. Or the relationship drifts over time, so the random split overestimates what deployment will deliver. Either way, **0.64 is the honest number**.

### 4.1 Feature selection must happen inside CV

The CORE-01 demo selected 20 out of 10,000 noise features using all the labels, and reached about 90% "accuracy" on pure noise. Feature selection is a fitted step, so it goes inside the pipeline.
That's the general rule for every data-driven step.

```python
import numpy as np, pandas as pd
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
rng = np.random.default_rng(0)

# --- Target-encoding leakage on a category UNRELATED to y ------------------------
n = 5000
cat = rng.integers(0, 1000, n).reshape(-1, 1)         # 1,000 levels, about 5 rows each
y = rng.integers(0, 2, n)                              # pure noise target
naive = pd.Series(y).groupby(cat[:, 0]).transform("mean").to_numpy().reshape(-1, 1)
oof = TargetEncoder(random_state=0).fit_transform(cat, y)   # cross-fitted (out-of-fold)
for name, enc in [("naive in-sample", naive), ("out-of-fold", oof)]:
    m = LogisticRegression().fit(enc, y)
    print(f"{name:16s} training AUC = {roc_auc_score(y, m.predict_proba(enc)[:, 1]):.3f}")

# Smoothing formula by hand
n_c, ybar_c, ybar, m_strength = 3, 1.0, 0.2, 10
print("smoothed encoding of a rare level:", (n_c * ybar_c + m_strength * ybar) / (n_c + m_strength))

# --- Cyclical hour encoding -----------------------------------------------------------
enc = lambda h: np.array([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)])
print("dist(23h, 0h) =", np.linalg.norm(enc(23) - enc(0)).round(3), " dist(0h, 1h) =", np.linalg.norm(enc(0) - enc(1)).round(3))
```

```python
# --- TF-IDF by hand vs sklearn ------------------------------------------------------
docs = ["the cat sat", "the dog sat", "the cat ate the fish"]
tv = TfidfVectorizer().fit(docs)
vocab = tv.get_feature_names_out()
N = len(docs)
df = np.array([sum(t in d.split() for d in docs) for t in vocab])
idf = np.log((1 + N) / (1 + df)) + 1
tf = np.array([d.split().count(t) for t in vocab for d in docs[2:3]])
v = tf * idf; v /= np.linalg.norm(v)
print("vocab   :", list(vocab))
print("by hand :", v.round(3))
print("sklearn :", tv.transform(docs[2:3]).toarray()[0].round(3))

# --- A leak-proof mixed-type pipeline -------------------------------------------------
df_ = pd.DataFrame({"age": rng.normal(40, 10, 400), "income": rng.lognormal(10, 1, 400),
                    "city": rng.choice(["A", "B", "C"], 400)})
df_.loc[rng.random(400) < 0.1, "income"] = np.nan
target = (df_["age"] + rng.normal(0, 10, 400) > 40).astype(int)
pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), ["age", "income"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])
pipe = make_pipeline(pre, LogisticRegression(max_iter=1000))
print("CV AUC of the full pipeline:", cross_val_score(pipe, df_, target, cv=5, scoring="roc_auc").mean().round(3))
```

---

## Pitfalls & misconceptions

- **"I scaled before splitting, but the effect is tiny."** For scaling, often true. For target encoding, selection, and imputation from the target, it can be catastrophic. Build the habit, so it never depends on which step you're doing.
- **Aggregates computed over the whole table**, including future rows.
- **`fit_transform` on the test set.** Test data only ever gets `transform`.
- **Dropping rows with missing values** without asking *why* they're missing.
- **Integer-encoding a nominal category for a linear model.** It then assumes city B sits "between" A and C.

## Cheat sheet

| Item | Rule / formula |
|---|---|
| Smoothed target encoding | $(n_c\bar y_c + m\bar y)/(n_c + m)$, computed out-of-fold |
| Cyclical feature | $(\sin 2\pi x/P,\ \cos 2\pi x/P)$ |
| TF-IDF (sklearn) | $\text{tf}\cdot(\ln\frac{1+N}{1+\text{df}} + 1)$, then L2-normalize |
| Leakage test | "Would I know this at prediction time?" + time-split vs random-split CV |
| Pipeline rule | every `fit` inside the `Pipeline`, inside CV |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why is target encoding dangerous, and how do you do it safely?</summary>

Each row's encoding includes its own label, which is direct target leakage, and rare levels are noisy. Use out-of-fold (cross-fitted) encoding, smoothed toward the global mean, fitted inside the CV pipeline. The demo: naive training AUC is far above 0.5 on pure noise, and OOF is about 0.5.
</details>

<details>
<summary>2. Why suspect days_since_account_closed?</summary>

It only exists once a customer has *already* churned, so it's a consequence of the label, not a predictor available beforehand. That's classic target leakage. Huge importance is the red flag.
</details>

<details>
<summary>3. Why must feature selection happen inside each CV fold?</summary>

Selection looks at the labels. Doing it on all the data lets the validation labels influence which features are chosen, so the CV score becomes optimistic, sometimes wildly (CORE-01 demo: about 90% on pure noise).
</details>

<details>
<summary>4. When is missingness itself useful?</summary>

When values are missing *because of* something related to the target (missing not at random): unreported income, an untaken test, an unanswered survey question. Add a missingness indicator.
</details>

<details>
<summary>5. A categorical feature with 50,000 levels?</summary>

OOF target encoding with smoothing, frequency encoding, the hashing trick, native categorical support (CatBoost/LightGBM), or a learned embedding in a neural net. Collapse the rare levels into "other" first.
</details>

<details>
<summary>6. Random-split CV 0.92, time-based split 0.64.</summary>

The model benefits from information that won't exist at prediction time (temporal leakage, future-derived aggregates, near-duplicates across time), or the relationship drifts over time. Trust the time split, investigate the features that drive the gap, and validate with point-in-time features.
</details>

## Where this leads

Next: [CORE-08 notes](08-interpretability-and-responsible-ml.md). You can now build strong models on good features. Next comes asking them *why* they predict
what they do, and whether they treat people fairly.
