# EL-02 notes: Bayesian & Probabilistic ML

[← Lesson EL-02](../../lessons/electives/02-bayesian-probabilistic-ml.md) · [All notes](../README.md) · [← EL-01 notes](01-time-series-forecasting.md) · Next: [EL-03 notes →](03-reinforcement-learning.md)

> **Reading time** ≈ 55 min. **You need:** [MATH-03 notes](../math/03-probability-statistics.md) §A2 (Bayes' rule), §B (likelihood, MAP), and §D1 (confidence intervals).

---

## Where we are

MAP estimation (MATH-03 §B4) used a prior and kept only the *peak* of the posterior. Bayesian inference keeps the **whole posterior distribution** over the parameters, so every prediction comes with uncertainty, small data is handled gracefully, and prior knowledge enters explicitly.
This note covers the core mechanics: conjugate updates, intervals, hierarchical models, and MCMC.

---

## 1. Bayesian updating

```math
\underbrace{p(\theta\mid\mathcal D)}_{\text{posterior}} \propto \underbrace{p(\mathcal D\mid\theta)}_{\text{likelihood}}\;\underbrace{p(\theta)}_{\text{prior}} .
```

### 1.1 A conjugate example: Beta–Binomial

A conversion rate $\theta$ has the prior $\text{Beta}(a, b)$, with density $\propto \theta^{a-1}(1-\theta)^{b-1}$. You observe $k$ conversions out of $n$. The likelihood is $\propto \theta^k(1-\theta)^{n-k}$. Multiply them:

```math
p(\theta\mid k, n) \propto \theta^{a + k - 1}(1-\theta)^{b + n - k - 1} \;\Longrightarrow\; \theta\mid\mathcal D \sim \text{Beta}(a + k,\ b + n - k).
```

The prior acts like **$a$ prior successes and $b$ prior failures**. The posterior mean is a weighted average of the prior mean and the data:

```math
\mathbb{E}[\theta\mid\mathcal D] = \frac{a + k}{a + b + n} = \underbrace{\frac{a+b}{a+b+n}}_{\text{prior weight}}\cdot\frac{a}{a+b} + \underbrace{\frac{n}{a+b+n}}_{\text{data weight}}\cdot\frac{k}{n}.
```

*Worked example:* a prior of Beta(2, 18) (about a 10% rate, worth 20 observations), and the data are 3 conversions in 10 visits. The posterior is Beta(5, 25), with mean $5/30 \approx 0.167$. The raw rate of 0.30 is pulled toward the prior, because 10 visits is little evidence. With 1,000 visits, the data would dominate.

**Conjugate** means the posterior is in the same family as the prior. That gives closed forms. Most real models aren't conjugate, hence MCMC (§3).

### 1.2 Credible vs confidence intervals

- **95% credible interval:** "given the data and the prior, there's a 95% probability that $\theta$ lies in this interval." It's a statement about $\theta$, conditional on *this* data.
- **95% confidence interval:** "the *procedure* that produced this interval captures the true $\theta$ in 95% of repeated experiments." $\theta$ is fixed. The randomness is in the interval. It does **not** say there's a 95% chance that $\theta$ is in *this* interval.

They often nearly coincide numerically (with flat priors and lots of data). They answer different questions, and the credible interval answers the one people usually ask.

### 1.3 Predictions integrate over the uncertainty

The **posterior predictive** averages over all plausible parameters: $p(\tilde y\mid\mathcal D) = \int p(\tilde y\mid\theta)\,p(\theta\mid\mathcal D)\,d\theta$. It's wider than plugging in one point estimate, which is honest when data is scarce.

**Bayesian A/B testing:** sample $\theta_A$ and $\theta_B$ from their posteriors, and report $P(\theta_B > \theta_A)$ and the expected lift, which are directly interpretable decision quantities.

---

## 2. Hierarchical models: partial pooling

There are many groups (schools, stores, hospitals), each with its own parameter $\theta_j$. There are three options:

- **No pooling:** estimate each $\theta_j$ separately. That's noisy for small groups, where a store with 3 sales gets an extreme estimate.
- **Complete pooling:** one shared $\theta$. That ignores real differences.
- **Partial pooling (hierarchical):** $\theta_j \sim \mathcal N(\mu, \tau^2)$, with $\mu$ and $\tau$ learned from *all* the groups. Each group's estimate is **shrunk toward the population mean**, by an amount set by how little data the group has:

```math
\hat\theta_j = \frac{\frac{n_j}{\sigma^2}\,\bar y_j + \frac{1}{\tau^2}\,\mu}{\frac{n_j}{\sigma^2} + \frac{1}{\tau^2}} .
```

That's a precision-weighted average. Groups with lots of data keep their own mean, and **groups with few observations borrow strength** from the others. Their estimates move toward $\mu$, and their uncertainty is honest.
This is the same shrinkage idea as ridge regression (CORE-04) and smoothed target encoding (CORE-07), but here the amount of shrinkage ($\tau$) is *learned* from the data.

---

## 3. MCMC: sampling the posterior when there's no closed form

The posterior is $p(\theta\mid\mathcal D) = p(\mathcal D\mid\theta)p(\theta)/p(\mathcal D)$, and the **evidence** $p(\mathcal D) = \int p(\mathcal D\mid\theta)p(\theta)\,d\theta$ is an integral over all parameters, which is intractable in high dimensions.
**MCMC** sidesteps it: it constructs a Markov chain whose stationary distribution *is* the posterior, using only the *unnormalized* density, because **ratios cancel the evidence**.

**Metropolis algorithm:**

1. From the current $\theta$, propose $\theta' = \theta + \text{noise}$.
2. Accept it with probability $\min\Big(1, \frac{p(\mathcal D\mid\theta')p(\theta')}{p(\mathcal D\mid\theta)p(\theta)}\Big)$. Otherwise stay put.
3. Repeat. After a warm-up, the visited values of $\theta$ are (correlated) samples from the posterior.

Modern samplers (**HMC/NUTS** in PyMC and Stan) use gradients to make long, informed moves, which is far more efficient in high dimensions.

**Convergence checks:**

- **Trace plots** of several chains should look like overlapping "hairy caterpillars".
- **R-hat** compares the between-chain variance with the within-chain variance. It should be **below about 1.01**. A value of 1.4 means the chains disagree: they haven't converged to the same distribution.
- **Effective sample size** (the number of samples adjusted for autocorrelation) should be large enough.
- **Divergences** (in HMC) signal regions of the posterior the sampler can't handle.

**Debug: R-hat = 1.4.** Try:

1. **Reparameterize:** hierarchical models often produce a "funnel" geometry. A **non-centred** parameterization ($\theta_j = \mu + \tau z_j$ with $z_j\sim\mathcal N(0,1)$) fixes it.
2. **Better, weakly informative priors** that rule out absurd regions.
3. **Longer warm-up, a higher `target_accept`**, more draws.
4. **Check for multimodality or non-identifiability** (for example, label switching in mixtures, or redundant parameters), and fix the model.
5. **Standardize the predictors.**

```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(0)

# --- Beta-Binomial update, credible interval vs a frequentist CI --------------------------------------
a, b, k, n = 2, 18, 3, 10
post = stats.beta(a + k, b + n - k)
print(f"posterior Beta({a+k},{b+n-k}): mean {post.mean():.3f}, 95% credible interval {post.ppf(0.025):.3f}..{post.ppf(0.975):.3f}")
p_hat = k / n; se = np.sqrt(p_hat * (1 - p_hat) / n)
print(f"MLE {p_hat:.2f}, Wald 95% CI {p_hat - 1.96*se:.3f}..{p_hat + 1.96*se:.3f}  (symmetric, ignores the prior; the normal approximation is shaky at n=10)")

# --- Bayesian A/B test: P(B > A) and the expected lift ------------------------------------------------------
A = stats.beta(1 + 120, 1 + 2000 - 120).rvs(100_000, random_state=1)
B = stats.beta(1 + 145, 1 + 2000 - 145).rvs(100_000, random_state=2)
print(f"P(B > A) = {np.mean(B > A):.3f}, expected relative lift {np.mean((B - A) / A):+.1%}")
```

```python
# --- Metropolis from scratch: the posterior of a logistic-regression slope; R-hat across chains --------------------
x = rng.normal(size=200); y = (rng.random(200) < 1 / (1 + np.exp(-(0.5 + 1.5 * x)))).astype(int)
def log_post(theta):                                  # prior N(0, 5^2) on both params; unnormalized
    a_, b_ = theta; z = a_ + b_ * x
    return np.sum(y * z - np.log1p(np.exp(z))) - np.sum(theta ** 2) / (2 * 25)
def metropolis(start, steps=6000, scale=0.15, seed=0):
    r = np.random.default_rng(seed); th = np.array(start, float); lp = log_post(th); out, acc = [], 0
    for _ in range(steps):
        prop = th + scale * r.normal(size=2); lp2 = log_post(prop)
        if np.log(r.random()) < lp2 - lp: th, lp, acc = prop, lp2, acc + 1     # the evidence cancels in the ratio
        out.append(th.copy())
    return np.array(out[1000:]), acc / steps
chains = [metropolis(s, seed=i) for i, s in enumerate([(-3, -3), (3, 3), (0, 0), (-2, 4)])]
samples = np.array([c for c, _ in chains])            # (chains, draws, params)
def rhat(ch):                                         # basic Gelman-Rubin statistic
    m, n_ = ch.shape; W = ch.var(1, ddof=1).mean(); B_ = n_ * ch.mean(1).var(ddof=1)
    return np.sqrt(((n_ - 1) / n_ * W + B_ / n_) / W)
print("acceptance rates:", [round(r, 2) for _, r in chains])
print("posterior mean slope:", samples[..., 1].mean().round(3), " R-hat slope:", rhat(samples[..., 1]).round(3))
short = np.array([metropolis(s, steps=1100, seed=i)[0][:60] for i, s in enumerate([(-3, -3), (3, 3)])])
print("R-hat with chains that barely started:", rhat(short[..., 1]).round(2), "<- not converged")
```

```python
# --- Partial pooling: shrinkage is strongest for small groups --------------------------------------------------------
J = 8; mu, tau, sigma = 50.0, 5.0, 20.0
n_j = np.array([3, 5, 10, 20, 40, 80, 150, 300])
true = rng.normal(mu, tau, J)
ybar = np.array([rng.normal(t, sigma, n).mean() for t, n in zip(true, n_j)])
pooled = (n_j / sigma**2 * ybar + mu / tau**2) / (n_j / sigma**2 + 1 / tau**2)
for n, yb, pp, tr in zip(n_j, ybar, pooled, true):
    print(f"n={n:3d}: raw mean {yb:6.1f}  partially pooled {pp:6.1f}  truth {tr:6.1f}")
print(f"RMSE raw {np.sqrt(np.mean((ybar-true)**2)):.2f} vs partially pooled {np.sqrt(np.mean((pooled-true)**2)):.2f}")
```

---

## Pitfalls & misconceptions

- **Reading a confidence interval as a probability statement about the parameter.**
- **"Uninformative" flat priors** on unbounded or transformed scales. They can be informative in unintended ways. Prefer weakly informative ones.
- **Ignoring convergence diagnostics.**
- **Over-trusting the posterior.** It's conditional on the model and the prior being reasonable. Do posterior predictive checks.
- **Centred hierarchical parameterizations with few data per group.** Funnel trouble.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Bayes | posterior ∝ likelihood × prior |
| Beta–Binomial | $\text{Beta}(a,b)$ + $k$/$n$ → $\text{Beta}(a+k, b+n-k)$ |
| Posterior mean | a precision-/count-weighted average of the prior mean and the data |
| Partial pooling | $\hat\theta_j = \frac{(n_j/\sigma^2)\bar y_j + \mu/\tau^2}{n_j/\sigma^2 + 1/\tau^2}$ |
| Metropolis accept | $\min(1, \tilde p(\theta')/\tilde p(\theta))$, so the evidence cancels |
| Convergence | R-hat < 1.01, enough ESS, no divergences, overlapping traces |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Confidence vs credible interval.</summary>

Credible: a 95% posterior probability that the parameter lies in it, given the data and prior. Confidence: the procedure covers the fixed true value in 95% of repeated samples. It says nothing probabilistic about this particular interval.
</details>

<details>
<summary>2. What does a hierarchical model do for a group with few observations?</summary>

It shrinks the group's estimate toward the population mean, in proportion to how little data the group has (partial pooling). That reduces variance and gives realistic uncertainty, while still letting well-observed groups keep their own values (the demo: lower RMSE than the raw means).
</details>

<details>
<summary>3. Why do we need MCMC?</summary>

The posterior's normalizing constant (the evidence) is a high-dimensional integral with no closed form for most models. MCMC samples using only the unnormalized density, because the acceptance ratios cancel the constant.
</details>

<details>
<summary>4. Chains don't mix (R-hat = 1.4).</summary>

Reparameterize (non-centred for hierarchical models), use weakly informative priors, standardize the predictors, lengthen the warm-up, raise `target_accept`, run more draws, and check for multimodality or non-identifiability.
</details>

## Where this leads

Next: [EL-03 notes](03-reinforcement-learning.md). Bayesian models quantify uncertainty about parameters. Reinforcement learning has to *act* under uncertainty, trading exploration against exploitation, and its policy gradients are exactly the machinery behind RLHF (GEN-07).
