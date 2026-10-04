# MATH-03 notes: Probability & Statistics for ML

[← Lesson MATH-03](../../lessons/math/03-probability-statistics.md) · [All notes](../README.md) · [Notation](../notation.md) · [← MATH-02 notes](02-calculus-optimization.md) · Next: [CORE-01 notes →](../core-ml/01-ml-workflow-end-to-end.md)

> **Reading time** ≈ 75 min, in four blocks matching the lesson. **You need:** [MATH-02 notes](02-calculus-optimization.md) Block A (derivatives), for MLE.

---

## Where we are

Calculus told us *how* to minimize a loss. This page answers *which* loss, and *how sure* we can be about the result.

- **Probability** lets a model say "70% spam" instead of "spam".
- **Maximum likelihood** turns "make the observed data probable" into the losses you'll use everywhere: MSE and cross-entropy.
- **Information theory** explains cross-entropy and KL divergence, which show up in classification, VAEs, distillation, and RLHF.
- **Statistics** tells you whether model B is *really* better than model A, or just lucky on this test set.

---

## Block A: Probability, Bayes, and distributions

### A1. Two rules generate everything

For events (or random variables) $A$ and $B$:

```math
\textbf{Sum rule: } p(A) = \sum_b p(A, B=b) \qquad \textbf{Product rule: } p(A, B) = p(A \mid B)\, p(B)
```

The sum rule is called **marginalization**: sum out what you don't care about. The product rule *defines* conditional probability, $p(A\mid B) = p(A,B)/p(B)$:
restrict the world to the cases where $B$ happened, then renormalize. **Independence** means $p(A,B) = p(A)p(B)$, or equivalently $p(A\mid B) = p(A)$.

### A2. Bayes' rule: updating beliefs

Apply the product rule both ways round, $p(A,B) = p(A\mid B)p(B) = p(B\mid A)p(A)$, and divide:

```math
\underbrace{p(H \mid D)}_{\text{posterior}} = \frac{\overbrace{p(D \mid H)}^{\text{likelihood}}\;\overbrace{p(H)}^{\text{prior}}}{\underbrace{p(D)}_{\text{evidence} = \sum_h p(D\mid h)p(h)}}
```

**Worked example (the lesson's question 1).** A disease has 1% prevalence. A test has 99% sensitivity, $p(+\mid\text{sick}) = 0.99$,
and 99% specificity, so $p(+\mid\text{healthy}) = 0.01$.

```math
p(\text{sick}\mid +) = \frac{0.99 \times 0.01}{0.99\times 0.01 + 0.01 \times 0.99} = \frac{0.0099}{0.0198} = 0.5 .
```

A positive result means only a **coin flip**. Picture 10,000 people: 100 are sick and 99 of them test positive; 9,900 are healthy and 99 of them
*also* test positive. The rare class gets swamped by false positives from the huge majority class. This is exactly why accuracy and ROC
curves mislead on imbalanced data (CORE-03), and why precision collapses when positives are rare.

### A3. Random variables, expectation, variance

A **random variable** $X$ assigns a number to each outcome. Its **expectation** is the probability-weighted average,
$\mathbb{E}[X] = \sum_x x\,p(x)$ (or $\int x\,p(x)\,dx$ for continuous variables). The **variance** is the expected squared distance from the mean:

```math
\operatorname{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}X)^2\big] = \mathbb{E}[X^2] - (\mathbb{E}X)^2 .
```

Four facts you'll use constantly:

1. **Linearity:** $\mathbb{E}[aX + bY] = a\mathbb{E}X + b\mathbb{E}Y$, *always*, even if $X$ and $Y$ are dependent.
2. $\operatorname{Var}(aX) = a^2\operatorname{Var}(X)$.
3. $\operatorname{Var}(X + Y) = \operatorname{Var}X + \operatorname{Var}Y + 2\operatorname{Cov}(X,Y)$. The covariance term vanishes if the variables are independent.
4. **So the average of $n$ independent copies has variance $\sigma^2/n$**, and its standard deviation shrinks like $1/\sqrt{n}$. This one fact explains:
   - why mini-batch gradients get less noisy with bigger batches (MATH-02 §B5);
   - why bagging and random forests reduce variance (CORE-05);
   - why a test set of 100 examples gives a wobbly accuracy estimate (Block D).

### A4. The distributions that matter

| Distribution | Models | Parameters | Mean, variance | Where in ML |
|---|---|---|---|---|
| Bernoulli | one yes/no | $p$ | $p$, $p(1-p)$ | Binary labels, logistic regression |
| Categorical | one of $K$ classes | $\pi_1..\pi_K$ | — | Softmax outputs, next-token prediction |
| Binomial | number of successes in $n$ trials | $n, p$ | $np$, $np(1-p)$ | Accuracy on a test set |
| Gaussian $\mathcal{N}(\mu,\sigma^2)$ | continuous noise | $\mu, \sigma^2$ | $\mu$, $\sigma^2$ | Regression noise, init, diffusion |
| Multivariate Gaussian | correlated vectors | $\mu, \Sigma$ | $\mu$, $\Sigma$ | GPs, VAEs, LDA, anomaly detection |

The Gaussian density is

```math
\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big).
```

Its log is a **negative quadratic** in $x$: $-\frac{(x-\mu)^2}{2\sigma^2} + \text{const}$. Keep that in mind; it's why MSE appears in Block B.

For the multivariate version, the contours are ellipses whose axes are the eigenvectors of $\Sigma$, with half-lengths proportional to $\sqrt{\lambda_i}$: the
same geometry as PCA (MATH-01 §C3).

### A5. Two limit theorems

- **Law of large numbers:** sample averages converge to the expectation. That's why "average loss on lots of data" estimates "expected loss on new data".
- **Central limit theorem:** the average of many independent variables is approximately Gaussian, with standard deviation $\sigma/\sqrt{n}$, *whatever the original distribution*.
  That's why error bars built from a normal approximation work for accuracy, for A/B tests, and so on.

```python
import numpy as np
from scipy import stats
rng = np.random.default_rng(0)

# Bayes, by formula and by simulation
prev, sens, fpr = 0.01, 0.99, 0.01
post = sens * prev / (sens * prev + fpr * (1 - prev))
sick = rng.random(1_000_000) < prev
pos = np.where(sick, rng.random(sick.size) < sens, rng.random(sick.size) < fpr)
print(f"P(sick | +): formula {post:.3f}, simulation {sick[pos].mean():.3f}")

# Var of the mean shrinks like 1/n; the CLT makes it Gaussian even for skewed data
draws = rng.exponential(scale=1.0, size=(20_000, 50))      # skewed distribution, var = 1
means = draws.mean(axis=1)
print(f"std of mean of 50: {means.std():.4f}  (theory 1/sqrt(50) = {1/np.sqrt(50):.4f})")
print(f"skewness of a single draw: {stats.skew(draws[:, 0]):.2f}, of the mean: {stats.skew(means):.2f}")
```

---

## Block B: Likelihood, MLE, and MAP: where loss functions come from

### B1. The likelihood

A model with parameters $\theta$ assigns a probability to the observed data. For independent examples,
$p(\mathcal{D}\mid\theta) = \prod_i p(y_i \mid x_i, \theta)$. Viewed as a function of $\theta$ with the data fixed, this is the **likelihood**.
**Maximum likelihood estimation (MLE)** picks the $\theta$ that makes the data most probable. Products of small numbers underflow, and logs turn
products into sums, so in practice we **minimize the negative log-likelihood (NLL)**:

```math
\hat\theta_{\text{MLE}} = \arg\min_\theta \; -\sum_{i=1}^N \log p(y_i \mid x_i, \theta).
```

**Every standard loss is an NLL in disguise.** Here are the two you'll use most.

### B2. Gaussian noise ⟹ mean squared error

Assume $y_i = f_\theta(x_i) + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$. Then

```math
-\log p(y_i\mid x_i,\theta) = \frac{(y_i - f_\theta(x_i))^2}{2\sigma^2} + \tfrac12\log(2\pi\sigma^2).
```

The second term doesn't depend on $\theta$, and $1/(2\sigma^2)$ is just a positive scale. So **minimizing the NLL is the same as minimizing the sum of
squared errors**. MSE *is* the assumption "errors are Gaussian". If your errors have heavy tails, with occasional huge outliers, that assumption is wrong. A Laplace
noise model gives the absolute error instead (MAE), which is more robust to outliers.

### B3. Bernoulli labels ⟹ cross-entropy (log loss)

For binary $y_i \in \lbrace 0, 1\rbrace$ with predicted probability $p_i = \sigma(z_i)$, the Bernoulli likelihood is $p_i^{y_i}(1-p_i)^{1-y_i}$. Take $-\log$:

```math
\ell_i = -\big[y_i \log p_i + (1-y_i)\log(1-p_i)\big].
```

That is **binary cross-entropy**. For $K$ classes with softmax probabilities, it becomes $-\log p_{i,\,y_i}$: "minus the log of the probability you gave to the right answer".

**The simplest MLE, derived by hand.** Flip a coin $n$ times and see $k$ heads. The log-likelihood is
$k\log p + (n-k)\log(1-p)$. Set its derivative to zero:

```math
\frac{k}{p} - \frac{n-k}{1-p} = 0 \;\Rightarrow\; \hat p = \frac{k}{n}.
```

That's the obvious answer, now with a justification. (The same calculation for a Gaussian gives $\hat\mu = \bar x$, the sample mean, and $\hat\sigma^2 = \frac1n\sum(x_i-\bar x)^2$.)

### B4. MAP: priors are regularizers

**Maximum a posteriori (MAP)** estimation maximizes the posterior $p(\theta\mid\mathcal D) \propto p(\mathcal D\mid\theta)\,p(\theta)$, so it adds a
$-\log p(\theta)$ term to the loss:

- Gaussian prior $w_j \sim \mathcal N(0, \tau^2)$: $-\log p(w) = \frac{1}{2\tau^2}\lVert w\rVert_2^2 + c$, which is **ridge / L2 / weight decay**.
- Laplace prior $p(w_j) \propto e^{-\lvert w_j\rvert/b}$: $-\log p(w) = \frac1b\lVert w\rVert_1 + c$, which is the **lasso / L1**.

**A stronger prior means a smaller $\tau$, which means a bigger $\lambda$.** Regularization is a belief, stated before you see the data, that weights are small.
Full Bayesian inference goes one step further: it keeps the whole posterior instead of only its peak ([EL-02 notes](../electives/02-bayesian-probabilistic-ml.md)).

```python
# MLE by brute force: the maximizer of the Bernoulli log-likelihood is k/n
k, n = 7, 20
ps = np.linspace(0.001, 0.999, 9999)
loglik = k * np.log(ps) + (n - k) * np.log(1 - ps)
print("argmax p =", round(ps[loglik.argmax()], 3), " k/n =", k / n)

# Gaussian NLL vs MSE: same minimizer (fit a constant c to data y)
y = rng.normal(3.0, 2.0, size=200)
cs = np.linspace(0, 6, 6001)
mse = [np.mean((y - c) ** 2) for c in cs]
nll = [-np.sum(stats.norm.logpdf(y, loc=c, scale=2.0)) for c in cs]
print("argmin MSE =", cs[np.argmin(mse)], " argmin NLL =", cs[np.argmin(nll)], " mean(y) =", round(y.mean(), 3))
```

---

## Block C: Information theory: entropy, cross-entropy, KL

### C1. Surprise and entropy

The **surprise** of an outcome with probability $p$ is $-\log p$. Certain events carry no surprise, and rare events carry a lot. **Entropy** is the average
surprise under the distribution itself:

```math
H(p) = -\sum_x p(x)\log p(x).
```

A fair coin has $H = \log 2$ (1 bit). A coin with $p=0.99$ has about 0.08 bits. Entropy is the **minimum average code length** for messages drawn from $p$,
and it is a measure of uncertainty.

### C2. Cross-entropy and KL divergence

Suppose the data really comes from $p$, but you encode (or predict) with $q$. Your average surprise is the **cross-entropy**:

```math
H(p, q) = -\sum_x p(x)\log q(x) = H(p) + \underbrace{\sum_x p(x)\log\frac{p(x)}{q(x)}}_{\mathrm{KL}(p\,\Vert\, q)\;\ge\; 0}.
```

- $\mathrm{KL}(p\Vert q)$ is the **extra** surprise you pay for using $q$ instead of the truth. It is $\ge 0$, and $= 0$ only when $q = p$ (Gibbs' inequality).
- In classification, $p$ is the one-hot true label, so $H(p) = 0$ and **cross-entropy loss = KL from the label to the prediction**. Minimizing it
  pushes the predicted distribution toward the truth. Averaged over a data set, minimizing cross-entropy = minimizing NLL = MLE. Three names for one idea.
- **Perplexity**, the LLM metric, is $e^{\text{cross-entropy}}$: "the model is as confused as if it were choosing uniformly among this many tokens" (GEN-01).

### C3. KL is not symmetric, and that matters

$\mathrm{KL}(p\Vert q)$ averages over **$p$**. Wherever $p > 0$ but $q \approx 0$, the term $\log\frac{p}{q}$ explodes. So fitting $q$ by minimizing
$\mathrm{KL}(p\Vert q)$ (*forward KL*) forces $q$ to **cover every region where $p$ has mass**. This is *mass-covering*, and it gives broad, blurry fits.

$\mathrm{KL}(q\Vert p)$ (*reverse KL*) averages over $q$. Now $q$ is punished for putting mass where $p$ is small, but not for **missing** parts of $p$. So $q$ tends to
**lock onto one mode**. This is *mode-seeking*, and it gives sharp but partial fits. Variational inference and VAEs use reverse KL; MLE uses forward KL.
The KL penalty in RLHF (GEN-07) is a reverse KL that keeps the policy close to the reference model.

### C4. Mutual information

$I(X;Y) = \mathrm{KL}\big(p(x,y)\,\Vert\, p(x)p(y)\big)$ is how far the joint distribution is from independence: how much knowing $X$ reduces your uncertainty about $Y$.
Unlike correlation, it catches non-linear relationships. It's used for feature selection (`mutual_info_classif`) and, at heart, by contrastive learning (CLIP, SimCLR).

```python
def H(p): p = np.asarray(p); return -np.sum(p * np.log(p))
def CE(p, q): return -np.sum(np.asarray(p) * np.log(q))
def KL(p, q): p, q = np.asarray(p), np.asarray(q); return np.sum(p * np.log(p / q))
p, q = [0.7, 0.2, 0.1], [0.5, 0.3, 0.2]
print(f"H(p)={H(p):.4f}  CE(p,q)={CE(p,q):.4f}  H+KL={H(p)+KL(p,q):.4f}")
print(f"KL(p||q)={KL(p,q):.4f}  KL(q||p)={KL(q,p):.4f}  <- not symmetric")
print(f"fair coin entropy = {H([.5,.5])/np.log(2):.3f} bits; 99/1 coin = {H([.99,.01])/np.log(2):.3f} bits")
```

---

## Block D: Statistics for evaluating models and running experiments

### D1. Your test score is an estimate, with an error bar

Accuracy on $n$ test examples is the average of $n$ Bernoulli outcomes. By §A3 its **standard error** is

```math
\mathrm{SE} = \sqrt{\frac{\hat a(1-\hat a)}{n}}, \qquad \text{95\% CI} \approx \hat a \pm 1.96\,\mathrm{SE}.
```

90% accuracy on 200 examples gives SE ≈ 2.1%, so the 95% CI is about ±4.2%. **Differences smaller than that are noise.** To halve the error bar you need 4× the data.

### D2. The bootstrap: error bars for anything

There is no simple formula for the SE of an F1 score, an AUC, or a median. The **bootstrap** makes one by simulation:

1. Resample the test set *with replacement* to the same size, many times (say 2,000).
2. Recompute the metric on each resample.
3. The spread of those values approximates the sampling distribution; take the 2.5% and 97.5% percentiles for a 95% interval.

It works because the empirical distribution is our best estimate of the true one. (ISLP §5.2; you'll use it in CORE-04 and CORE-11.)

### D3. Hypothesis tests and p-values, read correctly

- **Null hypothesis $H_0$:** "there is no difference".
- **p-value:** *if $H_0$ were true*, the probability of seeing a result at least this extreme.
- **What a p-value is not:** it is **not** the probability that $H_0$ is true. A small p-value says "this would be surprising under the null". It says nothing about how large or important the effect is.

**Comparing two models on the same test set:** use a **paired** analysis. The examples both models get right (or both get wrong) carry no information
about which model is better; only the **disagreements** do. McNemar's test asks: among the $b$ examples where only A is right and the $c$ where only B is right,
is the split unbalanced beyond what chance would produce? Equivalently, use a **paired bootstrap**: resample examples, and compute B's accuracy minus A's accuracy on each resample.

### D4. Multiple testing: the garden of forking paths

Run 40 comparisons when nothing is real, at $\alpha = 0.05$, and you expect $40 \times 0.05 = 2$ "significant" results by chance. The probability of at least one is $1 - 0.95^{40} \approx 87\%$.
Two fixes:

- **Bonferroni:** test each at $\alpha/m$ ($0.05/40 = 0.00125$). This controls the chance of *any* false positive. It's conservative.
- **Benjamini–Hochberg:** controls the *false discovery rate*, the expected fraction of your "discoveries" that are false. It's less conservative, and the standard choice when you're screening many candidates.

The ML version of this trap: trying 400 hyperparameter configurations and reporting the best validation score. That's why the score on a held-out test set comes out lower (CORE-04, CORE-12).

### D5. A/B tests in one paragraph

To detect a lift of $\delta$ on a conversion rate $p$ with the usual settings (α = 0.05 two-sided, 80% power), you need roughly
$n \approx 16\,p(1-p)/\delta^2$ users **per arm**. Halving the effect you want to detect quadruples the sample size. Decide $n$ **before** starting. "Peeking"
and stopping as soon as $p < 0.05$ inflates false positives, just like multiple testing does.

```python
# Accuracy CI, bootstrap CI, paired comparison, multiple testing
n = 2000
acc_A = 0.850
y_true = np.ones(n, dtype=int)
# Construct per-example correctness for two models with 30 "only A right" and 38 "only B right"
a_right = np.zeros(n, bool); a_right[:int(acc_A * n)] = True        # A: 1700 correct
b_right = a_right.copy()
b_right[:30] = False                                                  # 30 examples only A gets right
b_right[1700:1738] = True                                             # 38 examples only B gets right
print(f"acc A={a_right.mean():.3f}, acc B={b_right.mean():.3f}, diff={b_right.mean()-a_right.mean():+.3f}")
se = np.sqrt(acc_A * (1 - acc_A) / n)
print(f"normal-approx 95% CI for A: {acc_A-1.96*se:.3f} .. {acc_A+1.96*se:.3f}")

bstat = []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    bstat.append(b_right[idx].mean() - a_right[idx].mean())
lo, hi = np.percentile(bstat, [2.5, 97.5])
print(f"paired-bootstrap 95% CI for B-A: {lo:+.4f} .. {hi:+.4f}  (includes 0 -> not convincing)")

b, c = 30, 38
chi2 = (abs(b - c) - 1) ** 2 / (b + c)
print(f"McNemar chi2={chi2:.3f}, p={stats.chi2.sf(chi2, df=1):.3f}")

# 40 tests where nothing is real
pvals = np.array([stats.ttest_ind(rng.normal(size=50), rng.normal(size=50)).pvalue for _ in range(40)])
print("significant at 0.05:", (pvals < 0.05).sum(), "| after Bonferroni:", (pvals < 0.05 / 40).sum(),
      "| P(at least one false positive) =", round(1 - 0.95 ** 40, 2))
```

---

## Pitfalls & misconceptions

- **Confusing $p(A\mid B)$ with $p(B\mid A)$** (the prosecutor's fallacy). "The test is 99% accurate" doesn't mean "a positive is 99% likely to be sick".
- **Reading a p-value as the probability the null is true.**
- **Unpaired tests for paired data.** Two models evaluated on the same examples must be compared with a paired analysis. An unpaired test throws away most of the statistical power.
- **Thinking MSE is "neutral".** It assumes Gaussian noise, and it punishes outliers quadratically.
- **Treating KL as a distance.** It's asymmetric and doesn't satisfy the triangle inequality.

## Cheat sheet

| Idea | Formula |
|---|---|
| Bayes | $p(H\mid D) = p(D\mid H)p(H)/p(D)$ |
| Variance of a mean | $\sigma^2/n$ |
| MLE | $\arg\min_\theta -\sum_i \log p(y_i\mid x_i,\theta)$ |
| Gaussian NLL | MSE + const |
| Bernoulli NLL | $-[y\log p + (1-y)\log(1-p)]$ |
| MAP with Gaussian / Laplace prior | L2 / L1 regularization |
| Cross-entropy | $H(p,q) = H(p) + \mathrm{KL}(p\Vert q)$ |
| Perplexity | $\exp(\text{mean cross-entropy})$ |
| SE of accuracy | $\sqrt{a(1-a)/n}$ |
| Bonferroni | test at $\alpha/m$ |
| A/B sample size (80% power) | $n \approx 16\,p(1-p)/\delta^2$ per arm |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Bayes' rule on a test with 99% sensitivity and 1% prevalence.</summary>

Assuming 99% specificity too: $0.99\cdot0.01 / (0.99\cdot0.01 + 0.01\cdot0.99) = 0.5$. See §A2. With 95% specificity it drops to about 17%.
</details>

<details>
<summary>2. Why does minimizing MSE correspond to MLE under Gaussian noise?</summary>

The Gaussian log-density is $-(y-f(x))^2/(2\sigma^2)$ + const, so the negative log-likelihood of the data set is the sum of squared errors, up to a positive scale and a constant. Same minimizer (§B2).
</details>

<details>
<summary>3. KL(p‖q) vs KL(q‖p): what does each penalize?</summary>

$\mathrm{KL}(p\Vert q)$ heavily penalizes $q \approx 0$ where $p > 0$, so it is mass-covering. $\mathrm{KL}(q\Vert p)$ penalizes $q > 0$ where $p \approx 0$, so it is mode-seeking (§C3).
</details>

<details>
<summary>4. Model B beats A by 0.4% on 2,000 examples. Is it real?</summary>

0.4% is 8 examples. Run a paired analysis on the disagreements. In the code example, 30 vs 38 disagreements gives McNemar p ≈ 0.40, and the paired-bootstrap CI includes 0, so it's not convincing.
You'd need many more test examples, or a bigger gap.
</details>

<details>
<summary>5. You compared 40 feature sets and 3 look "significant" at p < 0.05.</summary>

You'd expect about 2 by chance alone. Correct for multiple comparisons (Bonferroni: p < 0.00125; or BH for the FDR), and confirm on fresh data before believing any of them.
</details>

## Where this leads

- **Next:** [CORE-01 notes](../core-ml/01-ml-workflow-end-to-end.md) start Path 1. The statistics here come back in CORE-03 (log loss), CORE-04 (CV and the bootstrap), and CORE-11 (calibration and conformal prediction).
- **Information theory returns in:** [DL-01 notes](../deep-learning/01-neural-networks-from-scratch.md) (softmax + cross-entropy), [GEN-04 notes](../llms-genai/04-generative-models-diffusion.md) (the ELBO), [GEN-07 notes](../llms-genai/07-post-training-alignment-reasoning.md) (the KL penalty).
