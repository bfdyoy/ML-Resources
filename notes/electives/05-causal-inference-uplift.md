# EL-05 notes: Causal Inference & Uplift Modeling

[← Lesson EL-05](../../lessons/electives/05-causal-inference-uplift.md) · [All notes](../README.md) · [← EL-04 notes](04-recommender-systems.md) · Next: [EL-06 notes →](06-anomaly-detection.md)

> **Reading time** ≈ 70 min. **You need:** [CORE-02 notes](../core-ml/02-linear-models-gradient-descent.md) §1 (coefficients, omitted-variable bias), [CORE-03 notes](../core-ml/03-classification-and-metrics.md) (classifiers, for propensity scores), and [MATH-03 notes](../math/03-probability-statistics.md) §D (tests).

---

## Where we are

Every model so far predicts $y$ from $x$. Causal inference asks a different question: **what would happen to $y$ if we *changed* $x$?** Should we send the discount? Does the feature reduce churn? Prediction can't answer that.
A great churn predictor tells you who will leave, not whom an intervention would keep. CORE-08 warned that "SHAP is not causal". This lesson gives you the tools that are.

---

## 1. Potential outcomes

For each unit $i$ and a binary treatment $T$, imagine **two** outcomes: $Y_i(1)$ if treated and $Y_i(0)$ if not. The **individual treatment effect** is $\tau_i = Y_i(1) - Y_i(0)$.

**The fundamental problem of causal inference:** we observe only **one** of the two, $Y_i = T_iY_i(1) + (1-T_i)Y_i(0)$. The other is a **counterfactual**, which by definition never happens. So individual effects are never observed directly. We estimate *averages*:

- **ATE** $= \mathbb{E}[Y(1) - Y(0)]$, over everyone;
- **ATT** = the ATE among the treated;
- **CATE** $\tau(x) = \mathbb{E}[Y(1) - Y(0)\mid X = x]$, the effect for units like $x$. This is what uplift models target.

### 1.1 Why naive comparisons fail: confounding

Decompose the naive difference in means:

```math
\underbrace{\mathbb{E}[Y\mid T{=}1] - \mathbb{E}[Y\mid T{=}0]}_{\text{naive}} = \underbrace{\mathbb{E}[Y(1) - Y(0)\mid T{=}1]}_{\text{ATT}} + \underbrace{\mathbb{E}[Y(0)\mid T{=}1] - \mathbb{E}[Y(0)\mid T{=}0]}_{\text{selection bias}} .
```

If the treated would have differed *anyway* (the sickest patients get the drug; the most engaged users adopt the feature), the selection-bias term is non-zero, and the naive comparison can have **the wrong sign**.

**Randomization fixes it:** if $T$ is assigned by coin flip, $T$ is independent of $(Y(0), Y(1))$, the selection bias is 0, and the difference in means is unbiased for the ATE. That's why A/B tests are the gold standard.

---

## 2. Causal graphs (DAGs): what to control for

Draw arrows from causes to effects. Three building blocks:

| Structure | Picture | Control for the middle variable? |
|---|---|---|
| **Confounder** | $T \leftarrow C \rightarrow Y$ | **Yes**: it opens a non-causal "back-door" path |
| **Mediator** | $T \rightarrow M \rightarrow Y$ | **No**, if you want the total effect (controlling removes part of the effect) |
| **Collider** | $T \rightarrow K \leftarrow Y$ | **No!** Conditioning on it *creates* a spurious association |

**The collider example:** suppose talent and looks are independent in the population, but a film studio casts people who are talented *or* good-looking. **Among actors** (conditioning on being cast), talent and looks become *negatively* correlated: knowing a cast actor isn't talented tells you they're probably good-looking.
Controlling for a collider (or selecting your sample on it) manufactures bias. The code demonstrates this.

**The back-door criterion:** adjust for a set of variables that blocks every back-door path from $T$ to $Y$, and that contains no descendant of $T$ (no mediators, no colliders caused by $T$).

---

## 3. Identification strategies

### 3.1 Adjustment: regression, matching, IPW

These assume **unconfoundedness** (all confounders are measured: $Y(t)\perp T\mid X$) and **overlap** (every kind of unit has some chance of either treatment: $0 < e(x) < 1$, where $e(x) = P(T{=}1\mid X{=}x)$ is the **propensity score**).

- **Regression adjustment:** model $\mathbb{E}[Y\mid T, X]$, and average the predicted difference.
- **Matching:** compare treated units with similar untreated ones (on $X$, or on $e(x)$).
- **Inverse propensity weighting (IPW):** reweight each unit by the inverse of the probability of the treatment it actually got, to build a pseudo-population where treatment is independent of $X$:

```math
\widehat{\text{ATE}}_\text{IPW} = \frac1n\sum_i\Big[\frac{T_iY_i}{\hat e(x_i)} - \frac{(1-T_i)Y_i}{1-\hat e(x_i)}\Big].
```

  When overlap is poor, propensities near 0 or 1 explode the weights and the variance. Trim them, or use **doubly robust** estimators (AIPW), which combine an outcome model and a propensity model, and stay consistent if *either* one is correct.

### 3.2 Difference-in-differences (DiD)

A treated group and a control group, observed before and after a change. The effect is the treated group's change minus the control group's change:

```math
\hat\tau_\text{DiD} = (\bar Y_{\text{T,after}} - \bar Y_{\text{T,before}}) - (\bar Y_{\text{C,after}} - \bar Y_{\text{C,before}}).
```

The control group's change estimates what the treated group would have done without treatment. **The key assumption is parallel trends:** absent treatment, both groups would have moved in parallel.
**Sanity checks:** plot several *pre-period* time points (the trends should already be parallel before treatment), run "placebo" DiDs on pre-periods (they should show no effect), and look for events coinciding with the treatment.

### 3.3 Instrumental variables and regression discontinuity

- **IV:** an instrument $Z$ that shifts $T$, but affects $Y$ *only through* $T$ (for example, a randomized encouragement to use a feature). The effect is $\frac{\text{effect of }Z\text{ on }Y}{\text{effect of }Z\text{ on }T}$ (the Wald estimator), and it identifies the effect for "compliers". The exclusion restriction is untestable, so it must be argued.
- **Regression discontinuity:** treatment is assigned by a threshold on a score (a scholarship for a test score ≥ 80). Units just above and just below are comparable, so the jump in the outcome at the cutoff is the local effect.

---

## 4. Heterogeneous effects: meta-learners and double ML

To target an intervention (uplift modeling), you need the CATE $\tau(x)$: whom does it help?

- **S-learner:** one model $\hat\mu(x, t)$ with $T$ as a feature, so $\hat\tau(x) = \hat\mu(x,1) - \hat\mu(x,0)$. **It struggles** when the effect is small relative to the outcome's variation: a regularized tree model may barely split on $T$, which shrinks the effects toward 0.
- **T-learner:** separate models $\hat\mu_1$ and $\hat\mu_0$ on the treated and control data, with $\hat\tau = \hat\mu_1 - \hat\mu_0$. **It struggles** when one group is small (that model is noisy), and because the two models' different errors don't cancel.
- **X-learner:** fit the T-learner, then impute individual effects (treated: $Y - \hat\mu_0(x)$; control: $\hat\mu_1(x) - Y$), model those, and blend them with propensity weights. It's **good with unbalanced groups**.
- **Double/debiased ML (DML):** predict $Y$ from $X$ and $T$ from $X$ with flexible ML (cross-fitted), and regress the **residuals** of $Y$ on the residuals of $T$. That removes confounding by $X$ with valid confidence intervals (a "partialling-out" estimator, the Frisch–Waugh–Lovell idea, made robust to ML bias).

### 4.1 Evaluating uplift models

You can't compute "accuracy" on an effect you never observe for any individual. Instead, use **randomized data**:

1. rank the units by the predicted uplift;
2. for each top-$k$ fraction, compare the treated and control outcomes **within that fraction**.

The **uplift (Qini) curve** shows the incremental gain from targeting the top $k$% by the model, against random targeting. A good model concentrates the real effect at the top.

**Debug: observational data says the feature *increases* churn; the A/B test says it *decreases* churn.** Likely explanations:

- **confounding / reverse causation:** at-risk users seek out the feature (support tools, cancellation help), so users of it churn more because they were already leaving;
- **selection on a collider**, or adjusting for a mediator;
- **a different population or time window**: the A/B test measures an effect among those randomized, at that time;
- **heterogeneity:** both can be "right" for different subgroups.

Trust the randomized result for the causal question, and use the gap to learn about the confounding.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
rng = np.random.default_rng(0)

# --- Confounding flips the sign; adjustment and IPW recover the truth ------------------------------------
n = 20000
severity = rng.normal(size=n)                                   # confounder
T = (rng.random(n) < 1 / (1 + np.exp(-2 * severity))).astype(int)   # sicker -> more likely treated
Y = 1.0 * T - 2.0 * severity + rng.normal(size=n)               # TRUE effect of treatment = +1.0
naive = Y[T == 1].mean() - Y[T == 0].mean()
adj = LinearRegression().fit(np.c_[T, severity], Y).coef_[0]
e = LogisticRegression().fit(severity[:, None], T).predict_proba(severity[:, None])[:, 1]
ipw = np.mean(T * Y / e - (1 - T) * Y / (1 - e))
print(f"true ATE +1.00 | naive {naive:+.2f} | regression adjustment {adj:+.2f} | IPW {ipw:+.2f}")

# --- Collider bias: conditioning on a common effect creates a correlation -------------------------------------
talent, looks = rng.normal(size=n), rng.normal(size=n)
cast = (talent + looks) > 1.5
print(f"corr(talent, looks): everyone {np.corrcoef(talent, looks)[0,1]:+.2f}, among the cast {np.corrcoef(talent[cast], looks[cast])[0,1]:+.2f}")

# --- Difference-in-differences ---------------------------------------------------------------------------------------
pre_T, pre_C = 10.0, 8.0
post_T, post_C = pre_T + 1.5 + 3.0, pre_C + 1.5          # common trend +1.5, true effect +3.0
print("DiD estimate:", (post_T - pre_T) - (post_C - pre_C), "| naive post-only difference:", post_T - post_C)
```

```python
# --- Meta-learners on randomized data with a heterogeneous effect --------------------------------------------------
n = 8000
X = rng.normal(size=(n, 4))
T = rng.integers(0, 2, n)
tau = 0.5 * np.maximum(X[:, 0], 0)                           # only units with x0 > 0 benefit
Y = 2 * X[:, 1] + np.sin(3 * X[:, 2]) + tau * T + rng.normal(0, 1, n)
gbm = lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
tr, te = np.arange(n) < n // 2, np.arange(n) >= n // 2           # fit on one half, evaluate on the other
s = gbm().fit(np.c_[X, T][tr], Y[tr])
tau_s = s.predict(np.c_[X, np.ones(n)]) - s.predict(np.c_[X, np.zeros(n)])
m1, m0 = gbm().fit(X[tr & (T == 1)], Y[tr & (T == 1)]), gbm().fit(X[tr & (T == 0)], Y[tr & (T == 0)])
tau_t = m1.predict(X) - m0.predict(X)
for name, est in [("S-learner", tau_s), ("T-learner", tau_t)]:
    print(f"{name}: corr with the true CATE {np.corrcoef(est[te], tau[te])[0, 1]:.2f}, mean |error| {np.mean(np.abs(est[te] - tau[te])):.3f}")
# Here the S-learner wins: the effect is large enough for the trees to split on T, while the T-learner's
# two separately fitted models make errors that don't cancel. With a tiny effect, the S-learner would shrink it toward 0.

# Uplift curve on HELD-OUT randomized data: the effect within the top-k% ranked by predicted uplift
Xte, Tte, Yte, est = X[te], T[te], Y[te], tau_s[te]
order = np.argsort(-est)
for frac in [0.2, 0.5, 1.0]:
    idx = order[: int(frac * len(order))]
    lift = Yte[idx][Tte[idx] == 1].mean() - Yte[idx][Tte[idx] == 0].mean()
    print(f"top {frac:.0%} by predicted uplift: observed effect {lift:+.3f}  (true mean effect there {tau[te][idx].mean():+.3f})")
# Observed effects are differences of noisy means (standard error about 0.07 for the full half, larger for the
# top 20%), so they wobble around the truth. The ranking is what matters: the effect is concentrated at the top.
```

---

## Pitfalls & misconceptions

- **Reading regression coefficients or SHAP values as causal effects.**
- **"Control for everything."** Controlling for colliders or mediators adds bias.
- **IPW with extreme propensities** and no overlap check.
- **DiD without checking pre-trends.**
- **Evaluating uplift models with predictive accuracy** instead of uplift or Qini curves on randomized data.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Fundamental problem | only one of $Y(1), Y(0)$ is ever observed |
| Naive = ATT + selection bias | randomization zeroes the bias |
| Back-door | adjust for confounders; not for mediators or colliders |
| IPW | $\frac1n\sum\big[\frac{TY}{e(x)} - \frac{(1-T)Y}{1-e(x)}\big]$, needs overlap |
| DiD | (ΔT) − (ΔC), assumes parallel trends |
| IV (Wald) | effect of $Z$ on $Y$ / effect of $Z$ on $T$ |
| Meta-learners | S (one model), T (two models), X (imputed effects + propensity) |
| Uplift evaluation | Qini/uplift curves on randomized data |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why can't you observe an individual treatment effect?</summary>

Each unit receives one treatment at a time, so only one potential outcome is realized. The other is counterfactual. Effects are identified only as averages, under assumptions or with randomization.
</details>

<details>
<summary>2. A DAG where controlling for a variable introduces bias.</summary>

$T \rightarrow K \leftarrow Y$, with $K$ a collider. Conditioning on $K$ makes $T$ and $Y$ dependent even when they're causally unrelated: explaining-away. The demo shows independent talent and looks becoming negatively correlated among cast actors.
</details>

<details>
<summary>3. Propensity-score weighting's assumptions.</summary>

Unconfoundedness (all confounders measured, $Y(t)\perp T\mid X$), overlap or positivity ($0 < e(x) < 1$), and a correctly specified (or well-calibrated) propensity model. Doubly robust estimators relax the last one.
</details>

<details>
<summary>4. Parallel trends, and how to sanity-check it.</summary>

Without treatment, the treated and control outcomes would have evolved in parallel. Check the pre-period trends over several time points, run placebo DiDs on pre-periods, and look for coinciding shocks that hit only one group.
</details>

<details>
<summary>5. When does each meta-learner struggle?</summary>

S: when the effect is small relative to the outcome's variation, because the model regularizes the treatment away (biased toward 0). T: with unbalanced or small treatment groups, when the noise of two separate models doesn't cancel. X: needs good propensities, and adds complexity, though it's designed for imbalance.
</details>

<details>
<summary>6. Why not evaluate uplift models with ordinary accuracy?</summary>

The target (an individual effect) is never observed, so there's no label to compare against. Evaluate on randomized data with uplift or Qini curves: does targeting by the model's ranking capture more of the real effect than random targeting?
</details>

<details>
<summary>7. Observational: the feature increases churn. A/B test: it decreases churn.</summary>

Confounding or reverse causation (at-risk users self-select into the feature), collider or mediator adjustment, a different population or time period, or heterogeneous effects. The randomized test answers the causal question. The observational gap reveals the selection bias.
</details>

## Where this leads

Next: [EL-06 notes](06-anomaly-detection.md). Back to the unsupervised side: finding the rare, unusual cases when you have almost no labels at all.
