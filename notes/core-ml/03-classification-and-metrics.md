# CORE-03 notes: Classification & Evaluation Metrics

[← Lesson CORE-03](../../lessons/core-ml/03-classification-and-metrics.md) · [All notes](../README.md) · [← CORE-02 notes](02-linear-models-gradient-descent.md) · Next: [CORE-04 notes →](04-generalization-validation-regularization.md)

> **Reading time** ≈ 55 min. **You need:** [CORE-02 notes](02-linear-models-gradient-descent.md) and [MATH-03 notes](../math/03-probability-statistics.md) §A2 (Bayes) and §B3 (Bernoulli likelihood).

---

## Where we are

CORE-02 predicted a number. Now the target is a **class** (spam/not spam, fraud/legit), and two new questions appear:

1. How do we turn a linear score into a **probability**?
2. How do we judge a classifier when **mistakes have different costs** and the classes are imbalanced?

---

## 1. Logistic regression: a linear model for log-odds

### 1.1 Why not just regress on 0/1?

A straight line will predict values like −0.3 or 1.4, which are not probabilities. And a few extreme points can swing it badly. We want a model whose output is
always in $(0,1)$ but which keeps the simple, interpretable linear score inside.

### 1.2 Odds and log-odds

If $p$ is the probability of the positive class, the **odds** are $\frac{p}{1-p}$ (with $p = 0.8$, the odds are 4, or "4 to 1"). Odds range over $(0, \infty)$, and
**log-odds** over $(-\infty, \infty)$. That's exactly the range of a linear score. So logistic regression **models the log-odds as linear**:

```math
\log\frac{p}{1-p} = w^\top x = z \quad\Longleftrightarrow\quad p = \sigma(z) = \frac{1}{1 + e^{-z}} .
```

*Reading a coefficient:* increasing $x_j$ by one unit **adds $w_j$ to the log-odds**, which **multiplies the odds by $e^{w_j}$**. With $w_j = 0.7$, the odds are multiplied by
$e^{0.7} \approx 2$. That's why we model log-odds: effects become additive on a scale where any value is legal.

### 1.3 The loss: cross-entropy, from maximum likelihood

Each label is a Bernoulli outcome with probability $p_i = \sigma(w^\top x_i)$. The negative log-likelihood (MATH-03 §B3) is the **log loss**:

```math
L(w) = -\frac1N\sum_{i=1}^N \Big[y_i\log p_i + (1-y_i)\log(1-p_i)\Big].
```

### 1.4 Its gradient is beautifully simple

Take one example. Use $\frac{d\sigma}{dz} = \sigma(1-\sigma)$ and differentiate $\ell = -[y\log\sigma(z) + (1-y)\log(1-\sigma(z))]$ with respect to $z$:

```math
\frac{\partial \ell}{\partial z} = -\Big[y\frac{\sigma(1-\sigma)}{\sigma} - (1-y)\frac{\sigma(1-\sigma)}{1-\sigma}\Big] = -\big[y(1-\sigma) - (1-y)\sigma\big] = \sigma(z) - y = p - y .
```

Then the chain rule through $z = w^\top x$ gives:

```math
\nabla_w L = \frac1N X^\top (p - y) .
```

Compare it with linear regression's $\frac2N X^\top(\hat y - y)$: **the same shape, "features times errors"**. The $\sigma(1-\sigma)$ factor cancelled.
With squared error on top of a sigmoid, it would *not* cancel, and gradients would vanish whenever the model is confidently wrong (MATH-02 self-check 1).
That's the deep reason classification uses cross-entropy. The same pairing (softmax + cross-entropy) is the output layer of every classifier network (DL-01).

The log loss is **convex** in $w$, so GD finds the global optimum. There's no closed form, which is the first time we *need* GD.

*Separable data warning:* if a line separates the classes perfectly, the loss keeps decreasing as $\lVert w\rVert \to \infty$, so the weights blow up. Regularization (the `C` parameter
in scikit-learn, where `C` = 1/λ) prevents this.

### 1.5 More than two classes: softmax

With $K$ classes, keep one weight vector per class, compute scores $z_k = w_k^\top x$, and normalize:
$p_k = e^{z_k}/\sum_j e^{z_j}$. The loss is $-\log p_{y}$ and the gradient with respect to $z$ is $p - \text{onehot}(y)$. It's the same pattern.

---

## 2. From probabilities to decisions: the threshold

A probability isn't a decision. You predict positive when $p > t$, and the right $t$ comes from **costs**, not from habit (0.5).

Let a false positive cost $C_{FP}$ and a false negative cost $C_{FN}$. Predicting positive has expected cost $(1-p)\,C_{FP}$, and predicting negative has expected cost $p\,C_{FN}$.
Predict positive when the first is smaller:

```math
(1-p)C_{FP} < p\,C_{FN} \;\Longleftrightarrow\; p > t^* = \frac{C_{FP}}{C_{FP} + C_{FN}} .
```

*Example:* missing a fraud costs 20× more than a false alarm, so $t^* = 1/21 \approx 0.048$. You should flag anything above about 5%.
This only works if $p$ is **calibrated**, meaning that among cases scored 0.05, about 5% really are positive (§6, and CORE-11).

---

## 3. The confusion matrix and its ratios

|  | Predicted + | Predicted − |
|---|---|---|
| **Actually +** | TP | FN |
| **Actually −** | FP | TN |

| Metric | Formula | Question it answers |
|---|---|---|
| Precision | TP / (TP + FP) | Of my alarms, how many were real? |
| Recall (sensitivity, TPR) | TP / (TP + FN) | Of the real positives, how many did I catch? |
| Specificity (TNR) | TN / (TN + FP) | Of the real negatives, how many did I leave alone? |
| FPR | FP / (FP + TN) = 1 − specificity | How often do negatives trigger alarms? |
| F1 | $2PR/(P+R)$ | Harmonic mean of precision and recall; low if either one is low |
| Accuracy | (TP + TN) / all | Misleading when classes are imbalanced |

*Worked example.* 1,000 emails, 100 of them spam. The filter flags 90, and 72 of those are spam. So TP = 72, FP = 18, FN = 28, TN = 882.
Precision = 72/90 = **0.80**, recall = 72/100 = **0.72**, specificity = 882/900 = **0.98**, F1 = 2(0.8)(0.72)/1.52 ≈ **0.758**, and accuracy = 954/1000 = **0.954**.
Note that "always say not-spam" already scores 0.90 accuracy.

**The trade-off.** Lower the threshold and you flag more: recall rises (or stays the same) and precision usually falls, because you're now flagging less certain cases, which are more often negatives.

---

## 4. Threshold-free summaries: ROC and PR curves

Sweep the threshold from 1 down to 0 and trace the points:

- **ROC curve:** TPR (recall) vs FPR. A random model gives the diagonal. **ROC-AUC has a lovely meaning: the probability that a randomly chosen positive gets a higher score
  than a randomly chosen negative.** It is a pure ranking measure.
- **PR curve:** precision vs recall. Its area (average precision, PR-AUC) has a **baseline equal to the positive rate**: 0.01 if 1% of examples are positive.

**Why ROC can flatter you on imbalanced data.** FPR divides by the number of negatives. With 99,000 negatives, 990 false alarms is only an FPR of 0.01, which looks
excellent on the ROC plot. But if there are only 1,000 positives and you catch 800 of them, precision = 800/(800+990) ≈ 0.45: more than half your alarms are false.
**The PR curve shows that pain; the ROC curve hides it.** Rule of thumb: when positives are rare and you care about the alarms you raise, report PR-AUC
(plus precision at your operating threshold).

---

## 5. Generative classifiers: LDA and Naive Bayes

Logistic regression is **discriminative**: it models $p(y\mid x)$ directly. **Generative** classifiers instead model how each class *produces* data, $p(x\mid y)$,
together with the class priors $p(y)$, and then flip it with Bayes' rule:

```math
p(y = k\mid x) = \frac{p(x\mid y=k)\,p(y=k)}{\sum_j p(x\mid y=j)\,p(y=j)} .
```

- **LDA** assumes each class is Gaussian, **with a shared covariance $\Sigma$**. Expand the log-ratio of two classes and the quadratic terms $x^\top\Sigma^{-1}x$ cancel,
  leaving a **linear** decision boundary. Same form as logistic regression, but fitted differently (from means and a covariance, not by maximizing the conditional likelihood).
- **QDA** gives each class its own covariance, so the quadratic terms don't cancel and the boundary is curved.
- **Naive Bayes** assumes the features are independent given the class: $p(x\mid y) = \prod_j p(x_j\mid y)$. The assumption is "wrong" but often fine for *ranking*, and the model is very fast. It's a classic for text.

**When does LDA beat logistic regression?** When its Gaussian assumption roughly holds, it uses that information and needs **less data**.
It also stays stable when the classes are perfectly separable, where unregularized logistic regression diverges (§1.4). When the assumption is badly wrong, logistic regression is more robust.

---

## 6. Calibration, briefly

A classifier is **calibrated** if, among all cases where it says 0.7, about 70% are positive. Check this with a **reliability diagram**: bin the predictions and plot the mean
prediction against the observed frequency. Logistic regression tends to be well calibrated because it's fitted by likelihood. Trees, SVMs, Naive Bayes, and deep nets
often aren't. Calibration matters whenever you use the probability itself: thresholds from costs (§2), expected-value calculations, or risk scores shown to people. Full treatment in [CORE-11 notes](11-uncertainty-calibration-conformal.md).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
rng = np.random.default_rng(0)
sig = lambda z: 1 / (1 + np.exp(-z))

# --- Logistic regression by gradient descent vs sklearn ------------------
N = 2000
X = rng.normal(size=(N, 2))
y = (rng.random(N) < sig(1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5)).astype(int)
Xb = np.c_[np.ones(N), X]
w = np.zeros(3)
for _ in range(5000):
    p = sig(Xb @ w)
    w -= 0.5 * Xb.T @ (p - y) / N               # gradient = X^T (p - y) / N
sk = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)   # C huge = (almost) no regularization
print("GD      :", w.round(3))
print("sklearn :", np.r_[sk.intercept_, sk.coef_[0]].round(3))
print("odds multiplier per unit of x1:", np.exp(w[1]).round(2))

# --- The worked confusion-matrix example ------------------------------------
TP, FP, FN, TN = 72, 18, 28, 882
P, R = TP / (TP + FP), TP / (TP + FN)
print(f"precision {P:.3f} recall {R:.3f} F1 {2*P*R/(P+R):.3f} specificity {TN/(TN+FP):.3f} accuracy {(TP+TN)/1000:.3f}")
```

```python
# --- ROC-AUC vs PR-AUC when positives are rare -------------------------------
n_neg, n_pos = 99_000, 1_000
scores = np.r_[rng.normal(0, 1, n_neg), rng.normal(2.5, 1, n_pos)]   # a decent ranker
labels = np.r_[np.zeros(n_neg), np.ones(n_pos)]
print(f"ROC-AUC {roc_auc_score(labels, scores):.3f}   PR-AUC {average_precision_score(labels, scores):.3f}   (PR baseline = {n_pos/(n_pos+n_neg):.3f})")

# AUC = P(random positive outscores random negative)
pos, neg = scores[labels == 1], scores[labels == 0]
print("pairwise estimate of AUC:", np.mean(rng.choice(pos, 200_000) > rng.choice(neg, 200_000)).round(3))

# --- Threshold from costs: C_FN = 20 * C_FP ------------------------------------
probs = sig(scores - 4.6)                     # pretend these are calibrated-ish probabilities
for t in [0.5, 1 / 21]:
    pred = probs > t
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    print(f"t={t:.3f}: precision {tp/(tp+fp):.2f} recall {tp/(tp+fn):.2f} total cost {fp*1 + fn*20}")
```

---

## Pitfalls & misconceptions

- **Reporting accuracy on imbalanced data.** Always compare it with the majority-class baseline.
- **Keeping the 0.5 threshold by default.** The threshold is a business decision (§2).
- **ROC-AUC as the headline metric for rare-positive problems.** Pair it with PR-AUC and with precision/recall at the operating point.
- **Choosing the threshold on the test set.** It's a hyperparameter: choose it on validation data.
- **Interpreting logistic coefficients as probability changes.** They are changes in log-odds. The effect on $p$ depends on where you start.

## Cheat sheet

| Item | Formula |
|---|---|
| Logistic model | $p = \sigma(w^\top x)$, log-odds $= w^\top x$ |
| Log loss | $-[y\log p + (1-y)\log(1-p)]$ |
| Gradient | $\frac1N X^\top(p - y)$ |
| Odds ratio for $x_j$ | $e^{w_j}$ |
| Cost-optimal threshold | $C_{FP}/(C_{FP}+C_{FN})$ |
| Precision / recall | TP/(TP+FP) / TP/(TP+FN) |
| F1 | $2PR/(P+R)$ |
| ROC-AUC | P(score of a random positive > score of a random negative) |
| PR-AUC baseline | the positive rate |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why model log-odds instead of the probability directly?</summary>

A probability lives in $[0,1]$, but a linear score ranges over all reals. Log-odds range over all reals, so a linear model is legal there. The effects of features become additive in log-odds (multiplicative in odds), and the resulting log loss is convex with a clean gradient, $X^\top(p-y)$.
</details>

<details>
<summary>2. 1% positives: why can ROC-AUC look great while the model is useless?</summary>

FPR divides by the huge number of negatives, so even thousands of false alarms are a small FPR, and the ROC curve looks excellent. Precision, which is what the people handling the alarms experience, can still be terrible. Report PR-AUC (baseline 0.01) and precision/recall at the chosen threshold. In the demo, ROC-AUC ≈ 0.96 while PR-AUC is far lower.
</details>

<details>
<summary>3. Moving the threshold from 0.5 to 0.2.</summary>

More cases are flagged. Recall rises or stays the same (more positives caught), and precision usually falls, because the newly flagged cases are less certain and more of them are negatives.
</details>

<details>
<summary>4. When would LDA beat logistic regression?</summary>

When the classes are roughly Gaussian with a shared covariance (LDA then uses that structure efficiently), on small data sets, and when the classes are well separated (unregularized logistic regression is unstable there).
</details>

<details>
<summary>5. What is calibration, and when do you care?</summary>

The predicted probability matches the observed frequency: of the cases scored 0.3, about 30% are positive. You care whenever the probability is used as a number: cost-based thresholds, expected values, risk communication, or combining models.
</details>

<details>
<summary>6. 99.2% accuracy, catches almost no fraud: diagnose and fix.</summary>

Check the base rate: if fraud is 0.8%, "never fraud" scores 99.2%. Look at the confusion matrix and the recall, then switch to PR-AUC and recall at a fixed precision.
Fixes: a cost-based threshold (§2), class weights, better features, and checking calibration. Validate on a time-based split, because fraud patterns drift.
</details>

## Where this leads

Next: [CORE-04 notes](04-generalization-validation-regularization.md). Both models so far can overfit, and we've been splitting data by instinct. CORE-04 makes generalization
precise (bias-variance), makes validation rigorous (cross-validation), and introduces the main cure for overfitting: regularization.
