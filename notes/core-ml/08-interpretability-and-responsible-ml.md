# CORE-08 notes: Interpreting Models & Responsible ML

[← Lesson CORE-08](../../lessons/core-ml/08-interpretability-and-responsible-ml.md) · [All notes](../README.md) · [← CORE-07 notes](07-feature-engineering-pipelines-leakage.md) · Next: [CORE-09 notes →](09-kernels-svms-nearest-neighbours.md)

> **Reading time** ≈ 55 min. **You need:** [CORE-05 notes](05-trees-and-ensembles.md) (trees, impurity) and [CORE-03 notes](03-classification-and-metrics.md) §3 (TPR, FPR, precision).

---

## Where we are

Path 1 has produced accurate models. Now the questions people will ask about them:

- **Why** did it predict that?
- **Which features** does it rely on?
- **Is it fair** to the people it affects?

Interpretability tools answer the first two, *if you know what each tool actually measures*. Fairness metrics answer the third, and they come with a surprising impossibility theorem.

---

## 1. Global vs local

- **Global:** how does the model behave *overall*? (Feature importance, partial dependence.)
- **Local:** why *this* prediction? (SHAP values for one row, LIME.)

Use global explanations to debug and to understand a model. Use local ones to justify individual decisions ("your loan was declined mainly because of X").

---

## 2. Feature importance

### 2.1 Impurity-based importance (and its bias)

Trees report how much each feature reduced impurity, summed over every split that used it. Features with **many possible split points** (continuous values, IDs, high-cardinality categoricals)
get more chances to reduce impurity by luck on the *training* data. So a random ID column can rank near the top. It also says nothing about *held-out* performance.

### 2.2 Permutation importance

Shuffle column $j$ in the **validation** data, which breaks its relationship with $y$ while keeping its distribution. The drop in score is its importance:

```math
I_j = s(\text{model}, X_\text{val}, y) - \mathbb{E}_\pi\big[s(\text{model}, X_\text{val}^{(\pi_j)}, y)\big] .
```

It's honest about generalization, and it works with any model. **Caveat:** with two correlated features, shuffling one leaves the other to carry the signal, so *both* look unimportant.
Group correlated features and permute them together, or drop one and refit.

---

## 3. Partial dependence (PDP) and ICE

The PDP for feature $S$ is "the average prediction when I force $x_S$ to a value, keeping each row's other features $x_C$":

```math
\hat f_S(v) = \frac1N\sum_{i=1}^N f\big(x_S = v,\ x_C^{(i)}\big).
```

**The hidden assumption: features are independent.** Suppose house size and number of rooms are strongly correlated. The PDP for "rooms = 8" then pairs 8 rooms with *every* row's size, including 40 m² flats.
Those combinations never occur, so the model's predictions there are extrapolation, and the curve can be nonsense.
**ICE curves** (one line per row instead of the average) reveal heterogeneity that the average hides. **ALE plots** fix the correlation problem by averaging local *differences* within narrow bins of $x_S$, using only realistic rows.

---

## 4. SHAP: Shapley values from game theory

### 4.1 The idea

Treat the features as players cooperating to produce a prediction. How do you split the "payout", $f(x) - \mathbb{E}[f(X)]$, fairly among them? Shapley's answer is to average each feature's
**marginal contribution** over every order in which features could be added:

```math
\phi_j = \sum_{S \subseteq F\setminus\{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\Big[v(S\cup\{j\}) - v(S)\Big],
```

where $v(S)$ is the expected prediction when only the features in $S$ are known (the others are averaged over a background data set).

### 4.2 The properties that make it the standard

- **Efficiency:** $\sum_j \phi_j = f(x) - \mathbb{E}[f(X)]$. The attributions add up exactly to the prediction's deviation from the average.
- **Symmetry:** features that contribute equally get equal credit.
- **Dummy:** a feature that never changes the output gets 0.

*A sanity check you can do by hand.* For a linear model with independent features, the formula collapses to $\phi_j = w_j\,(x_j - \mathbb{E}[x_j])$: the coefficient times how unusual this row's value is.
The code below computes Shapley values by brute force over all subsets and confirms this.

TreeSHAP computes exact values for tree ensembles in polynomial time. KernelSHAP approximates them for any model.

### 4.3 Reading SHAP plots, and their limits

- **Summary (beeswarm) plot:** one dot per row per feature, with position = SHAP value and colour = feature value. It shows both importance and direction.
- **Dependence plot:** SHAP value against the feature's value. It's like a PDP, but built from local attributions.

**SHAP explains the model, not the world.** A high SHAP value for `zip_code` means the *model* leans on zip code. It does *not* mean zip code *causes* default.
Zip code can be a **proxy** for protected attributes (race, ethnicity), so a "neutral" feature can carry discrimination into the model. Causal claims need causal methods (EL-05).
And with correlated features, the "averaging over the others" step again builds unrealistic combinations, so credit can be split between correlated features in unintuitive ways.

---

## 5. Fairness: criteria, and why you can't have them all

Let $A$ be a group attribute, $\hat Y$ the decision, and $Y$ the outcome.

| Criterion | Requires | Plain words |
|---|---|---|
| **Demographic parity** | $P(\hat Y = 1\mid A = a)$ is the same for every group | Equal selection rates |
| **Equalized odds** | Equal TPR **and** FPR across groups | Equal error rates, given the truth |
| **Equal opportunity** | Equal TPR only | Qualified people are caught equally often |
| **Predictive parity / calibration** | Equal precision (PPV), or calibrated scores, in every group | A score of 0.7 means 70% for everyone |

### 5.1 The impossibility, in one line of algebra

Within a group with base rate $p = P(Y=1)$, precision is determined by TPR and FPR:

```math
\text{PPV} = \frac{p\cdot\text{TPR}}{p\cdot\text{TPR} + (1-p)\cdot\text{FPR}} .
```

Fix PPV and TPR to be equal across two groups. If their base rates $p$ differ, this equation **forces their FPRs to differ**. So, unless the base rates are equal or the classifier is perfect,
you **cannot have both predictive parity and equalized odds** (Chouldechova 2017; Kleinberg et al. 2016). Demographic parity is in tension with both whenever base rates differ.

That's why fairness is a **choice of criterion, justified by context**. Which error is more harmful, and to whom? It is not a box to tick. The numbers below make the clash concrete.

### 5.2 A practical audit

1. Compute selection rate, TPR, FPR, precision, and calibration **per group**.
2. Look for proxy features (zip code, name-derived features), and check how missing values are distributed across groups.
3. Decide which criterion fits the harm, then mitigate: better data, constraints during training, or group-specific thresholds (where that's legal and justified).
4. Document all of it in a **model card**: intended use, data, metrics per group, and limitations.

```python
import numpy as np
from itertools import combinations
from math import factorial
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(0)

# --- Impurity vs permutation importance with a random ID column ---------------------
n = 2000
signal = rng.normal(size=n)
X = np.c_[signal, rng.normal(size=n), rng.permutation(n)]       # [real, noise, random "ID"]
y = (signal + 0.5 * rng.normal(size=n) > 0).astype(int)
Xtr, Xva, ytr, yva = train_test_split(X, y, random_state=0)
rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
perm = permutation_importance(rf, Xva, yva, n_repeats=10, random_state=0).importances_mean
for name, imp, p in zip(["signal", "noise", "random ID"], rf.feature_importances_, perm):
    print(f"{name:10s} impurity={imp:.3f}  permutation={p:+.3f}")
```

```python
# --- Exact Shapley values by brute force for a small linear model --------------------
w = np.array([2.0, -1.0, 0.5]); f = lambda Z: Z @ w
background = rng.normal(loc=[1, 2, 3], size=(5000, 3))          # E[x] = (1, 2, 3)
x = np.array([3.0, 0.0, 3.0])
def v(S):                       # expected prediction when the features in S are fixed to x
    Z = background.copy(); Z[:, list(S)] = x[list(S)]
    return f(Z).mean()
F = range(3); phi = np.zeros(3)
for j in F:
    others = [k for k in F if k != j]
    for r in range(len(others) + 1):
        for S in combinations(others, r):
            weight = factorial(len(S)) * factorial(3 - len(S) - 1) / factorial(3)
            phi[j] += weight * (v(S + (j,)) - v(S))
print("brute-force Shapley:", phi.round(3))
print("w * (x - E[x])     :", (w * (x - background.mean(0))).round(3))
print("sum of phi =", phi.sum().round(3), " f(x) - E[f] =", (f(x) - f(background).mean()).round(3))

# --- Fairness: fix TPR and PPV across groups with different base rates -> FPR must differ
def fpr_needed(p, tpr, ppv):    # solve the PPV equation for FPR
    return p * tpr * (1 - ppv) / ((1 - p) * ppv)
for p in [0.5, 0.2]:
    print(f"base rate {p}: TPR=0.8 and PPV=0.8 force FPR = {fpr_needed(p, 0.8, 0.8):.3f}")
```

---

## Pitfalls & misconceptions

- **Ranking features by impurity importance.** It's biased toward high-cardinality features and computed on training data.
- **Reading SHAP as causal.** It describes the model's reliance on a feature, not a mechanism in the world.
- **PDPs with strongly correlated features.** They show the model on impossible inputs. Use ALE plots or conditional methods.
- **"We removed the protected attribute, so the model is fair."** Proxies remain. Audit the outcomes per group.
- **Optimizing one fairness metric without saying why** it's the right one, and what it costs on the others.

## Cheat sheet

| Tool | Answers | Watch out |
|---|---|---|
| Permutation importance | How much does held-out performance depend on $x_j$? | Correlated features share (and hide) importance |
| PDP / ICE / ALE | How does the prediction change with $x_j$? | PDP assumes independence; ALE doesn't |
| SHAP | How much did each feature push *this* prediction from the average? | Explains the model, not causation |
| Demographic parity / equalized odds / predictive parity | Is the treatment of groups fair? | They can't all hold when base rates differ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why can impurity importance rank a random ID column highly?</summary>

A unique ID offers a huge number of split points. Deep trees use it to carve out individual training rows, which reduces *training* impurity by memorization. Permutation importance on validation data exposes this (about 0 in the demo).
</details>

<details>
<summary>2. What does a PDP assume, and what if it's false?</summary>

That the feature is independent of the others: it pairs each value with every row's other features. With correlation, it evaluates unrealistic combinations, so the curve reflects extrapolation, not the data. Use ALE or conditional explanations.
</details>

<details>
<summary>3. A high SHAP value vs causation.</summary>

SHAP says how much the model's output moved because of this feature's value, relative to the average. The model may use the feature as a proxy or a correlate. Intervening on the feature in the world may change nothing. Causation needs a causal design (EL-05).
</details>

<details>
<summary>4. Demographic parity vs equalized odds; why not both?</summary>

Parity equalizes selection rates. Equalized odds equalizes TPR and FPR given the true outcome. If base rates differ, equal error rates produce different selection rates, so you can't have both (except with a trivial or perfect classifier). Similarly, equalized odds and predictive parity conflict through the PPV equation (§5.1).
</details>

<details>
<summary>5. SHAP says zip_code is the top feature of a loan model.</summary>

Zip code can proxy for protected attributes (redlining). Check outcome disparities by group, test the model without zip code (and its proxies), look at the per-group TPR, FPR, and calibration, check the legal constraints, and document the decision in the model card.
</details>

## Where this leads

That completes the eight essential lessons of Path 1. Next: [CORE-09 notes](09-kernels-svms-nearest-neighbours.md), the first of the extended lessons.
It covers a different way to build a model: compare a new point with the training points (nearest neighbours, kernels), instead of learning weights per feature.
