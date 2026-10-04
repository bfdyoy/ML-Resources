# CORE-10 notes: Data-Centric ML

[← Lesson CORE-10](../../lessons/core-ml/10-data-centric-ml.md) · [All notes](../README.md) · [← CORE-09 notes](09-kernels-svms-nearest-neighbours.md) · Next: [CORE-11 notes →](11-uncertainty-calibration-conformal.md)

> **Reading time** ≈ 55 min. **You need:** [CORE-03 notes](03-classification-and-metrics.md) (probabilities, thresholds, PR) and [CORE-04 notes](04-generalization-validation-regularization.md) (CV).

---

## Where we are

Model-centric ML holds the data fixed and improves the model. Data-centric ML holds the model fixed and improves the data. On real projects the second is often cheaper and pays off more,
because real data is messy:

- some labels are **wrong**;
- the classes are **imbalanced**;
- labelers **disagree**;
- most of the data is **unlabeled**.

This note gives you the math behind the standard tool for each problem.

---

## 1. Label errors

### 1.1 Why a few wrong labels matter

**In training,** flexible models memorize the wrong labels (deep nets can fit 100% random labels), and that warps the boundary nearby.

**In testing,** wrong labels distort the score itself. For a binary task with symmetric label-noise rate $\varepsilon$, a model with true accuracy $a$ *measures* as

```math
a_\text{observed} = a(1 - \varepsilon) + (1 - a)\,\varepsilon .
```

At $\varepsilon = 5\%$, a perfect model scores 95%, and a 96%-accurate model scores 91.4%. Test-set label errors can even **reverse model rankings**: a model that learned the labelers' systematic mistakes can beat a better one on the noisy test set.

### 1.2 Confident learning: finding the likely errors

The idea: a good model, **evaluated out-of-sample**, will confidently disagree with wrong labels. Done carefully:

1. Get **out-of-sample** predicted probabilities $\hat p(k\mid x_i)$ for every training example using cross-validation (`cross_val_predict`). In-sample probabilities are useless, because the model has memorized the given labels, wrong ones included.
2. For each class $k$, compute a **per-class threshold** $t_k$: the average $\hat p(k\mid x)$ over the examples *labeled* $k$. This adapts to how confident the model typically is for each class.
3. Example $i$, labeled $\tilde y_i$, is counted as "truly class $j$" if $\hat p(j\mid x_i) \ge t_j$ (and $j$ has the highest probability among the classes above their thresholds).
   These counts form the **confident joint** $C_{\tilde y, y^*}$.
4. The off-diagonal entries ($\tilde y \ne y^*$) are the likely label errors. Rank them by how much the model disagrees (for example, the self-confidence $\hat p(\tilde y_i\mid x_i)$, lowest first).

Then **review** the flagged examples (don't blindly flip them), and retrain. `cleanlab` implements all of this. The code below reimplements the core and measures how many planted errors it recovers.

---

## 2. Class imbalance

### 2.1 First, what's actually wrong?

With 1% positives, a model can still *rank* well. The usual problems are (a) using accuracy or the 0.5 threshold (CORE-03), and (b) too few positive examples to learn from. The tools differ in which problem they address:

| Tool | What it does mathematically | When it helps |
|---|---|---|
| **Threshold tuning** | Leaves the model alone and changes the decision rule | Almost always. Try it first. It's free. |
| **Class weights** | Multiplies the loss of class $c$ by $w_c$ | Weak or regularized models whose fit is dominated by the majority |
| **Random undersampling** | Drops majority rows | Huge data sets (it saves compute); otherwise it wastes data |
| **Oversampling / SMOTE** | Duplicates minority rows, or interpolates new ones, $x_\text{new} = x_i + \lambda(x_{nn} - x_i)$ | Sometimes helps weak learners; rarely helps strong GBMs |

**Why weights and resampling shift probabilities.** Weighting positives by $w$ is equivalent to multiplying the positive class's prior odds by $w$. For logistic regression, the learned log-odds shift by about $\log w$.
So the model is **no longer calibrated** for the real base rate. If you undersample negatives, keeping a fraction $\beta$, the model sees inflated odds, and you correct the prediction $p_s$ like this:

```math
p = \frac{\beta\, p_s}{\beta\, p_s - p_s + 1} .
```

For ranking, none of this matters (the order is unchanged). For decisions based on probabilities, it does. That's why the imbalanced-learn pitfalls page recommends **threshold tuning on a calibrated model** as the default, and resampling only when you can show it helps.

### 2.2 Resampling before the split is a leak

If you oversample (or run SMOTE) and *then* split, copies or interpolations of the same minority example land in both training and test. The model is then tested on near-duplicates of its training data,
and the scores look great. **Resample only the training fold, inside the CV loop** (imbalanced-learn's `Pipeline` does this).

---

## 3. Label quality: measuring agreement

Two annotators agree on 90% of items. Is that good? If 85% of items are "negative" and both annotators mostly say "negative", they'd agree a lot **by chance**. **Cohen's kappa** corrects for this:

```math
\kappa = \frac{p_o - p_e}{1 - p_e}, \qquad p_e = \sum_k p_{A}(k)\,p_{B}(k),
```

where $p_o$ is the observed agreement and $p_e$ is the agreement expected if the annotators labeled independently, each at their own rates.

*Worked example.* Annotator A labels 80% of items positive, annotator B labels 70% positive, and they agree on 75% of items. Then $p_e = 0.8\cdot0.7 + 0.2\cdot0.3 = 0.62$ and $\kappa = (0.75 - 0.62)/(1 - 0.62) \approx 0.34$.
That's "fair" agreement at best, despite 75% raw agreement.
Low $\kappa$ usually means **the guidelines are ambiguous**, not that the labelers are careless. Fix the guidelines with examples of the edge cases, then relabel.

---

## 4. Few labels: semi-supervised learning

### 4.1 Self-training (pseudo-labelling)

Train on the labeled data. Predict the unlabeled data. Add the **confident** predictions (say $\hat p > 0.95$) as if they were labels, and retrain. Repeat.

**Why it can work:** the **cluster assumption**. If classes form separated clusters, confident pseudo-labels spread the labels along each cluster, which pushes the boundary into the low-density gap.

**When it makes things worse:**

- **Confirmation bias:** early mistakes become training labels and get reinforced.
- **Imbalance amplification:** the majority class produces more confident pseudo-labels, so it grows.
- **Distribution mismatch:** the unlabeled data comes from a different distribution, or contains classes you don't model.
- **The cluster assumption fails:** the classes overlap, and there is no low-density gap to find.

### 4.2 Consistency regularization (the FixMatch idea)

Add a loss that makes the model's prediction on a **strongly augmented** unlabeled example match its confident prediction on a **weakly augmented** version of the same example.
The model learns that the label is invariant to these transformations, so the unlabeled data teaches the shape of each class. This is the backbone of modern semi-supervised vision. It's also a close cousin of self-supervised learning (CV-02).

---

## 5. Choosing what to label: active learning

Labeling budgets are finite, so spend them where the model learns the most. The loop: train, then score the unlabeled pool, then label the top-$b$ examples, then retrain.

**Uncertainty sampling** scores each pool example by:

- **least confidence**, $1 - \max_k \hat p_k$;
- **margin**, $\hat p_{(1)} - \hat p_{(2)}$ (small means uncertain);
- **entropy**, $-\sum_k \hat p_k\log\hat p_k$.

**Failure modes:**

1. **Redundancy:** the top-$b$ uncertain points are often near-identical, all in the same confusing region, so you pay for the same lesson $b$ times.
2. **Outliers:** they look uncertain but teach nothing useful.
3. **Sampling bias:** the labeled set stops looking like the real distribution.

**Diversity sampling** fixes the first two: cluster the candidates and pick uncertain points from *different* clusters, or choose a "core set" that covers the pool's embedding space.
Always keep a small **randomly sampled** set for unbiased evaluation.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import cohen_kappa_score
rng = np.random.default_rng(0)

# --- Confident learning from scratch: find planted label errors ---------------------
X, y_true = make_classification(n_samples=3000, n_features=10, n_informative=6, n_classes=3,
                                n_clusters_per_class=1, class_sep=1.5, random_state=0)
y_noisy = y_true.copy()
flip = rng.random(len(y_true)) < 0.08                           # corrupt 8% of labels
y_noisy[flip] = (y_true[flip] + rng.integers(1, 3, flip.sum())) % 3
P = cross_val_predict(LogisticRegression(max_iter=2000), X, y_noisy, cv=5, method="predict_proba")
t = np.array([P[y_noisy == k, k].mean() for k in range(3)])     # per-class thresholds
above = P >= t
cand = np.where(above, P, -1).argmax(1)                         # best class among those above threshold
has_any = above.any(1)
issue = has_any & (cand != y_noisy)
prec = (issue & flip).sum() / issue.sum(); rec = (issue & flip).sum() / flip.sum()
print(f"flagged {issue.sum()} examples; precision {prec:.2f}, recall {rec:.2f} of the {flip.sum()} planted errors")

# Observed accuracy under label noise
a, eps = 0.96, 0.05
print("a 96%-accurate model measures as", a * (1 - eps) + (1 - a) * eps, "on 5%-noisy labels")
```

```python
# --- Cohen's kappa: the worked example -------------------------------------------------
n = 1000
A = np.r_[np.ones(800), np.zeros(200)]
B = np.r_[np.ones(625), np.zeros(175), np.ones(75), np.zeros(125)]     # B: 70% positive, 75% agreement
print(f"raw agreement {np.mean(A == B):.2f}, P(B=1) {B.mean():.2f}, kappa {cohen_kappa_score(A, B):.3f}")

# --- Undersampling shifts probabilities; the correction formula undoes it ------------------
Xi, yi = make_classification(n_samples=200_000, n_features=5, weights=[0.99], flip_y=0, random_state=1)
beta = 0.05                                                          # keep 5% of the negatives
keep = (yi == 1) | (rng.random(len(yi)) < beta)
m_all = LogisticRegression(max_iter=1000).fit(Xi, yi)
m_sub = LogisticRegression(max_iter=1000).fit(Xi[keep], yi[keep])
ps = m_sub.predict_proba(Xi[:20000])[:, 1]
corrected = beta * ps / (beta * ps - ps + 1)
print(f"mean predicted P(pos): full data {m_all.predict_proba(Xi[:20000])[:,1].mean():.4f}, "
      f"undersampled {ps.mean():.4f}, corrected {corrected.mean():.4f}, true rate {yi[:20000].mean():.4f}")
```

---

## Pitfalls & misconceptions

- **Running confident learning on in-sample probabilities.** The model has memorized the noisy labels, so it can't flag them.
- **Auto-flipping flagged labels.** Some will be genuinely ambiguous, or rare-but-correct. Review them.
- **SMOTE as a reflex.** Try threshold tuning (and calibration) first. Measure whether resampling helps at all.
- **Oversampling before the split.**
- **Reporting raw agreement instead of $\kappa$.**
- **Pseudo-labeling with a weak initial model and no confidence threshold.**

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Observed accuracy under symmetric noise | $a(1-\varepsilon) + (1-a)\varepsilon$ |
| Confident-learning threshold | $t_k$ = mean $\hat p(k\mid x)$ over examples labeled $k$ (out-of-sample) |
| Undersampling correction | $p = \beta p_s/(\beta p_s - p_s + 1)$ |
| Cohen's kappa | $(p_o - p_e)/(1 - p_e)$ |
| Uncertainty scores | $1 - \max p$; margin $p_{(1)} - p_{(2)}$; entropy |
| SMOTE | $x_i + \lambda(x_{nn} - x_i)$, $\lambda\sim U[0,1]$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. What does confident learning estimate, and why out-of-sample probabilities?</summary>

It estimates the joint distribution of given labels vs true labels (the confident joint), which identifies the examples whose given label is probably wrong. In-sample probabilities reflect memorization of the given labels, so wrong labels look "confident". Out-of-sample predictions reveal the disagreement.
</details>

<details>
<summary>2. 1% positives: class weights, undersampling, SMOTE, or thresholds?</summary>

Start with threshold tuning on a well-calibrated model, evaluated with PR-AUC and precision/recall at the operating point. It's free and doesn't distort the probabilities. Then try class weights (cheap, no data loss). Undersample for compute reasons on huge data. Treat SMOTE as an experiment, not a default: it seldom helps strong learners.
</details>

<details>
<summary>3. Why is oversampling before the split disastrous?</summary>

Duplicates (or SMOTE interpolations) of the same minority example end up in both training and test, so the test measures memorization of near-copies. That's train/test contamination, and it inflates scores. Resample inside each training fold only.
</details>

<details>
<summary>4. What does Cohen's κ measure that raw agreement doesn't?</summary>

Agreement beyond chance, given each annotator's label rates. High raw agreement can happen by chance when one class dominates (the worked example: 75% raw agreement, κ ≈ 0.34).
</details>

<details>
<summary>5. When does self-training make a model worse?</summary>

When the initial model's confident mistakes are reinforced (confirmation bias), when imbalance makes the majority class's pseudo-labels dominate, when the unlabeled data comes from a different distribution, or when the classes overlap so there's no low-density boundary for the pseudo-labels to find.
</details>

<details>
<summary>6. Uncertainty sampling's failure mode; how does diversity help?</summary>

It selects batches of redundant, near-identical uncertain points (and outliers), so the budget is wasted on one region. Diversity sampling (clustering or core sets) spreads the batch across the data space, so each label teaches something new.
</details>

<details>
<summary>7. Fixing 3% of training labels made test accuracy drop.</summary>

Most likely the **test labels contain the same systematic errors**. The old model learned the labelers' mistakes and matched the noisy test set. Clean (a sample of) the test set too, and re-evaluate. Otherwise, check whether the "fixes" were themselves wrong.
</details>

## Where this leads

Next: [CORE-11 notes](11-uncertainty-calibration-conformal.md). Imbalance handling showed that probabilities can be distorted. CORE-11 makes "trustworthy probabilities" a goal in its own right,
then goes beyond it to prediction **sets** with a guarantee.
