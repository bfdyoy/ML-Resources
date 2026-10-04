# CORE-12 notes: Hyperparameter Optimization

[← Lesson CORE-12](../../lessons/core-ml/12-hyperparameter-optimization.md) · [All notes](../README.md) · [← CORE-11 notes](11-uncertainty-calibration-conformal.md) · Next: [DL-01 notes →](../deep-learning/01-neural-networks-from-scratch.md)

> **Reading time** ≈ 50 min. **You need:** [CORE-04 notes](04-generalization-validation-regularization.md) §2 (CV, nested CV) and [CORE-09 notes](09-kernels-svms-nearest-neighbours.md) §4 (Gaussian processes).

---

## Where we are

Every model has knobs that training doesn't set: $\lambda$, $C$, $\gamma$, tree depth, learning rate. Choosing them is an optimization problem with three nasty properties:

- **Black-box:** there are no gradients with respect to the hyperparameters (in general).
- **Expensive:** each evaluation is a full training run.
- **Noisy:** CV scores wobble with the seed and the folds.

This note covers the four strategies you'll actually use, and how to avoid fooling yourself with the result.

---

## 1. Grid vs random search

### 1.1 The low effective dimension argument

Usually **only a few hyperparameters really matter**, and you don't know which ones in advance. Take 9 trials over two hyperparameters, where only one of them matters:

- **Grid (3×3):** only **3 distinct values** of the important hyperparameter get tried. The other 6 trials repeat those values while varying the irrelevant one.
- **Random:** **9 distinct values** of the important hyperparameter. Nothing is wasted.

With more dimensions, the gap grows: a grid with $n$ values per axis in $d$ dimensions costs $n^d$ trials but explores only $n$ values of each hyperparameter.

### 1.2 How many random trials?

If the "good region" holds a fraction $q$ of the search space, the chance that $n$ random trials all miss it is $(1-q)^n$. For the top 5% ($q = 0.05$):

```math
1 - 0.95^{n} \ge 0.95 \;\Longleftrightarrow\; n \ge \frac{\log 0.05}{\log 0.95} \approx 59 .
```

**About 60 random trials land in the top 5% with 95% probability, whatever the dimension.** That's a useful default budget.

### 1.3 Sample on the right scale

Learning rates, regularization strengths, $C$ and $\gamma$ matter **multiplicatively**: going from 0.001 to 0.01 is as big a step as 0.01 to 0.1. Sample them **log-uniformly**.
Sample integers like depth or the number of leaves uniformly, or log-uniformly when the range is wide.

---

## 2. Bayesian optimization: learn where to look

Random search ignores everything it has seen so far. **Bayesian optimization (BO)** builds a cheap model of "hyperparameters → score" and uses it to choose the next trial.

### 2.1 The surrogate

Fit a **Gaussian process** (CORE-09 §4) to the trials so far. At every candidate $x$, it gives a predicted score $\mu(x)$ **and** an uncertainty $\sigma(x)$.

### 2.2 The acquisition function: exploitation vs exploration

Choose the next $x$ by maximizing an **acquisition function** that balances "the predicted score is good" (exploit) against "we're uncertain there" (explore). The classic is **expected improvement (EI)**.
For minimization, with the best score so far $f^*$:

```math
\mathrm{EI}(x) = \mathbb{E}\big[\max(f^* - f(x),\, 0)\big] = (f^* - \mu)\,\Phi(Z) + \sigma\,\phi(Z),\qquad Z = \frac{f^* - \mu}{\sigma},
```

where $\Phi$ and $\phi$ are the standard normal CDF and PDF. The first term is large where the mean is already better than $f^*$ (exploitation). The second is large where $\sigma$ is large (exploration).
Where the model is both confident and bad, EI is about 0, so no trial is wasted there.

**Upper (lower) confidence bound**, $\mu(x) - \kappa\sigma(x)$, makes the trade-off explicit through $\kappa$.

### 2.3 TPE (what Optuna uses by default)

Instead of modelling score given $x$, the **Tree-structured Parzen Estimator** splits the trials into "good" (the top $\gamma$ fraction) and "bad", fits a density to each, $\ell(x)$ and $g(x)$,
and proposes points where $\ell(x)/g(x)$ is large. It handles conditional and categorical spaces naturally ("if optimizer = SGD, also tune momentum") and scales better than GPs to many trials.

**When is BO worth it?** When each trial is expensive (minutes to hours) and you can afford tens to hundreds of trials. For very cheap models, plain random search with more trials is often just as good.

---

## 3. Early stopping of bad trials: successive halving and Hyperband

Most configurations are clearly bad early on. **Successive halving:**

1. Start $n$ configurations with a small budget $r$ (a few epochs, or a data subset).
2. Keep the best $1/\eta$ of them (for example $\eta = 3$) and give the survivors $\eta\times$ the budget.
3. Repeat until one configuration remains.

Each round costs about the same total compute, so with $\log_\eta n$ rounds you screen $n$ configurations for about $n r \log_\eta n$ units, instead of $n \times$ the full budget.

**The slow-starter problem:** a configuration that learns slowly at first (for example a small learning rate) can be killed early, even though it would have won.
**Hyperband** hedges by running several successive-halving **brackets** with different trade-offs, from "many configs, tiny starting budget" to "few configs, big starting budget".
Optuna's pruners (median, successive halving, Hyperband) apply the same logic trial by trial.

---

## 4. Not fooling yourself

### 4.1 The best of many noisy scores is optimistic

If 500 configurations are truly equally good and CV noise has standard deviation $\sigma$, the best observed score is roughly $3\sigma$ better than the truth. (The rule of thumb $\sigma\sqrt{2\ln n}$ gives $3.5\sigma$; it slightly overestimates for moderate $n$.)
That's the selection bias from CORE-01 §2.3, at scale. Honest estimates need:

- **nested CV** (tuning inside, evaluation outside), or an **untouched test set** scored once;
- **logging every trial**, so you know how many configurations you actually tried.

### 4.2 Design the objective so noise can't win

If the "best" trial falls apart when you rerun it with another seed, the search was optimizing noise. Fixes:

- average the objective over **several folds and seeds** (repeated CV);
- use a **bigger validation set**;
- **re-evaluate the top-k** configurations with more seeds before picking one;
- prefer **simpler** configurations within noise of the best.

### 4.3 What to tune first

| Model | Tune first | Then | Usually leave at defaults |
|---|---|---|---|
| GBM (LightGBM/XGBoost) | learning rate + number of trees (by early stopping) | `num_leaves`/`max_depth`, `min_child_samples`, `subsample`, `colsample_bytree` | Many minor regularizers |
| Neural net | **learning rate** (by far), plus its schedule/warmup | weight decay, batch size, dropout, width/depth | Adam's betas and epsilon |
| SVM (RBF) | $C$ and $\gamma$ jointly, on a log grid | — | — |
| Ridge / lasso | $\lambda$ (log scale) | — | — |

The [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) frames network tuning as a sequence of *scientific experiments*: fix the "nuisance" hyperparameters,
vary the one you're studying, and conclude. That's more reliable than a giant blind search.

```python
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings; warnings.filterwarnings("ignore")     # GP fitting warnings are harmless here
rng = np.random.default_rng(0)

# --- Grid vs random when only 1 of 2 hyperparameters matters ------------------------------
important = lambda a: -(a - 0.73) ** 2                         # the score depends only on 'a'
grid = [(a, b) for a in np.linspace(0, 1, 3) for b in np.linspace(0, 1, 3)]
rand = rng.random((9, 2))
print("distinct 'a' values: grid", len({round(a, 3) for a, _ in grid}), " random", len(set(rand[:, 0].round(3))))
print("best |a - 0.73|: grid", min(abs(a - 0.73) for a, _ in grid).round(3), " random", np.abs(rand[:, 0] - 0.73).min().round(3))
print("trials for a 95% chance of hitting the top 5%:", int(np.ceil(np.log(0.05) / np.log(0.95))))
```

```python
# --- Bayesian optimization with a GP + expected improvement, vs random search -------------------
f = lambda x: np.sin(3 * x) + 0.3 * (x - 2) ** 2               # minimize on [0, 4]
grid_x = np.linspace(0, 4, 2001)
f_min = f(grid_x).min()

def EI(mu, sd, best):
    sd = np.maximum(sd, 1e-9); Z = (best - mu) / sd
    return (best - mu) * norm.cdf(Z) + sd * norm.pdf(Z)

def bo(n_iter=12, seed=0):
    r = np.random.default_rng(seed)
    X = list(r.uniform(0, 4, 3)); Y = [f(x) for x in X]
    for _ in range(n_iter):
        gp = GaussianProcessRegressor(Matern(nu=2.5), alpha=1e-6, normalize_y=True).fit(np.array(X)[:, None], Y)
        mu, sd = gp.predict(grid_x[:, None], return_std=True)
        x_next = grid_x[np.argmax(EI(mu, sd, min(Y)))]
        X.append(x_next); Y.append(f(x_next))
    return min(Y)

bo_gap = np.mean([bo(seed=s) - f_min for s in range(10)])
rs_gap = np.mean([min(f(np.random.default_rng(s).uniform(0, 4, 15))) - f_min for s in range(10)])
print(f"after 15 evaluations, mean gap to the true minimum: BO {bo_gap:.4f}   random search {rs_gap:.4f}")
```

```python
# --- Optimism of the best of many equally good configurations -------------------------------
sigma, n_cfg = 0.01, 500
best = [np.max(0.80 + sigma * rng.normal(size=n_cfg)) for _ in range(2000)]
print(f"true score 0.800; mean 'best of {n_cfg}' = {np.mean(best):.4f}  (theory ≈ 0.80 + sigma*sqrt(2 ln n) = {0.80 + sigma*np.sqrt(2*np.log(n_cfg)):.4f})")

# --- Successive halving: how often does the truly best config survive? -----------------------
n, eta = 81, 3
true_q = rng.normal(size=n)                                     # each config's final quality
def survives(noise_at_small_budget):
    alive = np.arange(n); budget = 1
    while len(alive) > 1:
        noisy = true_q[alive] + noise_at_small_budget / np.sqrt(budget) * rng.normal(size=len(alive))
        alive = alive[np.argsort(-noisy)[: max(1, len(alive) // eta)]]
        budget *= eta
    return alive[0] == np.argmax(true_q)
for noise in [0.1, 0.5, 1.5]:
    print(f"early-budget noise {noise}: best config survives {np.mean([survives(noise) for _ in range(400)]):.0%} of the time")
```

---

## Pitfalls & misconceptions

- **A grid over learning rates on a linear scale.** Use a log scale.
- **Reporting the best trial's CV score as the final performance.**
- **Optimizing a single-seed, single-split objective.**
- **Pruning too aggressively,** which kills slow starters. Hyperband's brackets hedge this.
- **Tuning dozens of hyperparameters at once.** Tune the 2–4 that matter. Fix the rest at sensible defaults.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Random trials for the top-$q$ region with probability $p$ | $n \ge \log(1-p)/\log(1-q)$ (≈ 60 for 5% / 95%) |
| Expected improvement | $(f^*-\mu)\Phi(Z) + \sigma\phi(Z)$, $Z = (f^*-\mu)/\sigma$ |
| Successive halving | keep the top $1/\eta$, multiply the budget by $\eta$ |
| Optimism of the max of $n$ | up to about $\sigma\sqrt{2\ln n}$ |
| Scales | log-uniform for learning rate, $\lambda$, $C$, $\gamma$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why does random search cover important dimensions better?</summary>

With $n$ trials, a grid tests only $n^{1/d}$ distinct values per hyperparameter, while random search tests $n$ distinct values of each. If one dimension dominates, random search samples it far more densely (9 vs 3 values in the demo).
</details>

<details>
<summary>2. What does the acquisition function trade off?</summary>

Exploitation (sampling where the surrogate predicts a good score) vs exploration (sampling where the surrogate is uncertain and might hide something better). EI combines both terms in closed form.
</details>

<details>
<summary>3. How does successive halving decide what to kill, and when does it kill a winner?</summary>

At each rung it ranks configurations by their score at the current small budget and keeps the top $1/\eta$. It kills an eventual winner when early performance doesn't predict final performance: slow starters, or noisy early scores (the demo: survival drops as early noise grows). Hyperband's brackets reduce the risk.
</details>

<details>
<summary>4. Why is the best of 500 trials optimistic, and how do you get an honest estimate?</summary>

The maximum of many noisy estimates includes the luckiest noise, roughly $3\sigma$ for 500 trials (the demo: 0.830 vs a true 0.800 with $\sigma = 0.01$). Use nested CV or a held-out test set evaluated once, and report how many trials you ran.
</details>

<details>
<summary>5. What to tune first for a GBM and for a neural net?</summary>

GBM: learning rate with early-stopped number of trees, then tree size (leaves or depth, min child samples), then subsampling. Neural net: the learning rate and its schedule, then weight decay, batch size, and dropout. Architecture size comes after the training settings are sound.
</details>

<details>
<summary>6. The best trial is much worse when rerun with another seed.</summary>

The objective was dominated by noise (one seed, one split, a small validation set), so the search found the luckiest configuration, not the best one. Average over seeds and folds, use bigger validation sets, re-evaluate the top-k configurations, and prefer robust configurations near the top.
</details>

## Where this leads

**Path 1 is complete.** Next: [DL-01 notes](../deep-learning/01-neural-networks-from-scratch.md). You've seen linear models trained by gradient descent and non-linear models built from trees and kernels.
Deep learning stacks many linear layers with non-linearities between them and trains them all with the chain rule (backpropagation). Every tool from Path 1 (losses from likelihood, GD, regularization, validation, tuning) carries straight over.
