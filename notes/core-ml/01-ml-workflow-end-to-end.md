# CORE-01 notes: The ML Workflow, End to End

[← Lesson CORE-01](../../lessons/core-ml/01-ml-workflow-end-to-end.md) · [All notes](../README.md) · [← MATH-03 notes](../math/03-probability-statistics.md) · Next: [CORE-02 notes →](02-linear-models-gradient-descent.md)

> **Reading time** ≈ 40 min. **You need:** Python and pandas. [MATH-03 notes](../math/03-probability-statistics.md) §D1 (error bars) helps.

---

## Where we are

This is the first lesson of Path 1. Before any particular algorithm, you need the **frame** every algorithm lives in:

- what exactly a model is trying to do;
- why we split data the way we do;
- why so many "great" models fail in production.

Get this frame right and every later lesson slots into it.

---

## 1. The big picture

Supervised learning in one sentence: **find a function $f$ that predicts $y$ from $x$ well on data you haven't seen yet.**
The last five words are the hard part. Everything in this lesson (splits, baselines, pipelines) exists to *measure* performance on unseen data honestly.

The workflow:

```
frame the problem → get data → explore → split → baseline → preprocess + model (in a pipeline)
      → validate (CV) → iterate → final test (once) → deploy → monitor
```

---

## 2. The math of "doing well on unseen data"

### 2.1 Risk and empirical risk

Assume examples $(x, y)$ come from some unknown distribution $P$. For a loss $\ell$ (say squared error), the quantity we truly care about is the **risk**,
the expected loss on a fresh example:

```math
R(f) = \mathbb{E}_{(x,y)\sim P}\big[\ell(f(x), y)\big].
```

We can't compute it, because we don't know $P$. We only have a sample, so we compute the **empirical risk**, the average loss on the training set:

```math
\hat R_{\text{train}}(f) = \frac{1}{N}\sum_{i=1}^N \ell(f(x_i), y_i).
```

Training is **empirical risk minimization** (ERM): pick the $f$, from some family of models, that minimizes $\hat R_{\text{train}}$.

### 2.2 Why training error lies

For a *fixed* $f$ chosen before seeing the data, $\hat R$ is an unbiased estimate of $R$ (that's the law of large numbers). But we **chose** $f$ *because*
it scores well on these exact points. So $\hat R_{\text{train}}(f)$ is **optimistically biased**: the model has fitted the noise in the sample as well as
the signal. The difference $R(f) - \hat R_{\text{train}}(f)$ is the **generalization gap**. Flexible models can drive training error to zero while the
real risk stays high. That's overfitting, and CORE-04 is all about it.

### 2.3 The fix: data the model never saw

Hold out a **test set**. Since $f$ was chosen without looking at it, the test error *is* an unbiased estimate of $R(f)$, with a standard error you can compute
(MATH-03 §D1).

**But only if you look at it once.** Every time you use test performance to make a decision (pick a model, tweak a feature), the test set becomes part of the
training process, and its estimate becomes optimistic too. Here is why.

*Selection bias, worked example.* Ten models all have **true** accuracy 80%. You evaluate each on the same 500-example test set. Each score is
$0.80 \pm \sqrt{0.8 \cdot 0.2/500} \approx 0.80 \pm 0.018$ of noise. The **best** of ten such noisy scores is typically around 0.82–0.83. You'd report a 2–3% improvement
that doesn't exist. The code below simulates it.

### 2.4 Three sets, three jobs

| Set | Used for | Touched |
|---|---|---|
| **Training** | Fitting parameters ($w$, tree splits) | Constantly |
| **Validation** (or CV folds) | Choosing *hyperparameters*, features, model family | Many times, so it becomes slightly optimistic |
| **Test** | One final, unbiased estimate | **Once**, at the end |

Cross-validation (CORE-04) replaces the single validation set with $k$ rotating ones, so you use your data more efficiently.

---

## 3. Splitting well

- **Stratify** classification splits so that each split keeps the class proportions. With 2% positives and a 200-example test set, a random split could easily give
  1 positive or 8, and your recall estimate would be meaningless. Stratification matters most for **small data sets and rare classes**.
- **Group** split when examples are related (several rows per patient, user, or document). Otherwise the model can memorize the *entity* and look great.
- **Split by time** when you'll predict the future. Train on the past, validate on later data. A random split lets the model peek at the future.

The rule behind all three: **the split must mimic how the model will meet new data in deployment.**

---

## 4. Baselines: the number to beat

Before any real model, compute the score of something trivial:

- **predict the mean** (regression) or **the majority class** (classification);
- a one-feature rule ("churn if last login > 30 days");
- the current system, or a human.

Why first? Because a number like "0.91 accuracy" means nothing on its own. If the majority class is 90%, it's barely better than nothing. A baseline also catches bugs:
a model that **can't beat the baseline** usually has a pipeline bug, not a modelling problem.

---

## 5. Pipelines and leakage

**Leakage** means information that won't be available at prediction time gets into training. It makes validation scores lie. The most common form is subtle:
**fitting preprocessing on the full data set.**

`StandardScaler().fit(X)` computes means and standard deviations *using the test rows*. The model has now been told something about the test distribution.
For scaling, the effect is usually small. For **feature selection**, **target encoding**, or **imputation from the target**, it can be enormous.

*The dramatic demo.* Make 100 examples with 10,000 features of pure random noise and random labels. Nothing is learnable, so honest accuracy is 50%.
Now select the 20 features most correlated with the labels **using all the data**, then cross-validate a classifier on those 20. You'll get around 90% accuracy (0.93 in the run below)
on pure noise, because the selection step already "saw" the validation labels. Put the selection **inside** a `Pipeline` and it drops back to about 50%.

The cure, structurally: every step that **learns** anything from data (`fit`) goes in a `Pipeline`, and the `Pipeline` goes inside the CV loop. scikit-learn then refits each step on
each training fold only.

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

rng = np.random.default_rng(0)

# 1) Selection bias: best of 10 equally good models on one test set
true_acc, n_test = 0.80, 500
best = [max(rng.binomial(n_test, true_acc, size=10) / n_test) for _ in range(2000)]
print(f"true accuracy 0.80; average 'best of 10' test score = {np.mean(best):.3f}")

# 2) Leakage: feature selection on pure noise
X = rng.normal(size=(100, 10_000))
y = rng.integers(0, 2, size=100)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

top = SelectKBest(f_classif, k=20).fit(X, y)                  # WRONG: sees all labels
leaky = cross_val_score(LogisticRegression(max_iter=1000), top.transform(X), y, cv=cv).mean()

pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
honest = cross_val_score(pipe, X, y, cv=cv).mean()           # RIGHT: selection refit per fold
print(f"accuracy on pure noise: leaky selection = {leaky:.2f}, inside pipeline = {honest:.2f}")
```

---

## 6. Framing: is ML even the right tool?

ML pays off when the following hold:

1. there is a **pattern** to learn;
2. the pattern is **hard to write down as rules**;
3. you have **data** (with labels, for supervised learning) that reflects deployment;
4. **mistakes are tolerable**, or can be caught downstream.

Framing means turning a business wish into: an **input** available at prediction time, a **label** you can actually get, a **metric** tied to the real cost of errors,
and a **baseline**. A vague goal like "use AI to reduce churn" becomes "predict, on day 1 of each month, the probability that a subscriber cancels within 30 days, measured
by PR-AUC on a time-based holdout, against the rule 'no login for 14 days'".

---

## Pitfalls & misconceptions

- **Tuning on the test set**, even "just a little". It's selection bias (§2.3).
- **Preprocessing before splitting** (§5).
- **A random split on time-ordered or grouped data** (§3).
- **No baseline**, so you can't tell good from bad.
- **Using features unavailable at prediction time**, e.g. "number of support calls *this month*" when you predict on day 1.

## Cheat sheet

| Item | Rule |
|---|---|
| Risk $R(f)$ | Expected loss on new data. Training error underestimates it; the test set estimates it, used once |
| How to split | Mimic deployment: stratify for rare classes, group for related rows, split by time for forecasting |
| Baseline | Build it first. If a real model can't beat it, suspect a bug |
| Leakage rule | Anything that calls `fit` goes in the `Pipeline`, and the `Pipeline` goes inside CV |

## Answer sketches for the lesson's self-check

<details>
<summary>1. A problem where ML is the wrong tool.</summary>

Computing tax owed, or validating an email format. The rules are known exactly, and mistakes are unacceptable, so write deterministic code.
ML is also wrong when there's no data reflecting deployment, or no way to get labels.
</details>

<details>
<summary>2. Why stratify, and when does it matter most?</summary>

To keep class proportions equal across splits, so every split is representative and the metric estimates are stable. It matters most with rare classes and small data sets, where a random split can leave a split with almost no positives.
</details>

<details>
<summary>3. What is a baseline and why build it first?</summary>

A trivial predictor (the majority class, the mean, a one-line rule, or the current system). It gives meaning to your metric ("better than what?") and catches pipeline bugs: a real model that can't beat it is usually broken.
</details>

<details>
<summary>4. Why is StandardScaler().fit(X) on the full data a bug?</summary>

The scaler's means and standard deviations then include test (or validation) rows, so information from the evaluation data leaks into training, and the evaluation becomes optimistic. The same mistake with feature selection or target encoding can be dramatic (§5 demo). Fit on the training fold only, via a `Pipeline`.
</details>

<details>
<summary>5. Validation vs test set, and what goes wrong if you tune on test.</summary>

Validation is for making choices; test is for one final unbiased estimate. Tuning on test turns it into a validation set. The best-looking configuration is partly lucky, so the reported score is optimistically biased (§2.3), and you have no honest estimate left.
</details>

<details>
<summary>6. 0.97 in CV, 0.71 deployed: three causes.</summary>

(1) Leakage: preprocessing or selection fitted outside CV, or a feature that encodes the target or the future.
(2) A split that doesn't match deployment: a random split on time-ordered or grouped data.
(3) Distribution shift: production data differs (new users, a changed upstream pipeline, training/serving skew in feature computation).
Also possible: tuning so much that CV itself became optimistic.
</details>

## Where this leads

Next: [CORE-02 notes](02-linear-models-gradient-descent.md). Now that you know *how* to measure a model honestly, you'll build the first real one,
linear regression, and train it two ways: in closed form and by gradient descent.
