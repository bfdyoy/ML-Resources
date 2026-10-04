# GEN-10 notes: Advanced Diffusion & Flow Matching

[← Lesson GEN-10](../../lessons/llms-genai/10-advanced-diffusion-flow-matching.md) · [All notes](../README.md) · [← GEN-04 notes](04-generative-models-diffusion.md) · Next: [PROD-01 notes →](../production/01-ml-system-design.md)

> **Reading time** ≈ 75 min. **You need:** [GEN-04 notes](04-generative-models-diffusion.md) §3 (DDPM), [MATH-02 notes](../math/02-calculus-optimization.md) (gradients), and [MATH-03 notes](../math/03-probability-statistics.md) §A4 (Gaussians). Comfort with "an ODE is a velocity field you follow" is enough; no SDE theory is assumed.

---

## Where we are

DDPM works, but its derivation (a chain of 1,000 Gaussians) hides a simpler picture. This note rebuilds generation around one object, the **score**, and then around a **velocity field**. The payoffs:

- why DDIM can sample in 20–50 steps with the *same* trained model;
- what classifier-free guidance really does;
- **flow matching**, today's simplest and most popular training recipe, which is just a regression.

---

## 1. Scores, and why they're learnable

The **score** of a density is the gradient of its log, $s(x) = \nabla_x\log p(x)$: an arrow pointing toward higher density.

**The key property: the normalizing constant vanishes.** If $p(x) = \tilde p(x)/Z$, then $\nabla_x\log p = \nabla_x\log\tilde p - \nabla_x\log Z = \nabla_x\log\tilde p$, because $Z$ doesn't depend on $x$. Likelihood-based training must handle $Z$ (often intractable). Score-based training never has to.

**Sampling with a score: Langevin dynamics.** Follow the score uphill while adding noise:

```math
x \leftarrow x + \eta\, s(x) + \sqrt{2\eta}\, z,\qquad z\sim\mathcal N(0, I).
```

For small $\eta$ and many steps, this produces samples from $p$. The noise keeps it from collapsing onto the peaks. **Its weakness:** walkers rarely cross low-density gaps between separated modes, so mixing between modes is extremely slow (the demo below finds both modes but gets their weights wrong). The fix is to learn scores at **many noise levels** and anneal from high noise (where the modes blur together) down to low noise. That's the step from score matching to diffusion.

### 1.1 Denoising score matching: the trick that makes it trainable

We don't know $\nabla\log p_\text{data}$. But we can **add noise** and use the fact that the score of the *noising kernel* is known. If $x_t = x_0 + \sigma\varepsilon$, then $\nabla_{x_t}\log q(x_t\mid x_0) = -\varepsilon/\sigma$. And, remarkably:

```math
\arg\min_\theta\ \mathbb{E}\big\lVert s_\theta(x_t) - \nabla_{x_t}\log q(x_t\mid x_0)\big\rVert^2 \quad\text{also minimizes}\quad \mathbb{E}\big\lVert s_\theta(x_t) - \nabla_{x_t}\log p_\sigma(x_t)\big\rVert^2 ,
```

where $p_\sigma$ is the *noised data distribution*. The regression target is noisy, but its **conditional average** given $x_t$ is the true score. (It's the same reason MSE regression learns $\mathbb{E}[y\mid x]$.)

**Connection to DDPM:** with $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$, the noise predictor and the score are the same thing, up to scale:

```math
s_\theta(x_t, t) = -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}} .
```

**Predicting the noise *is* estimating the score of the noised data.** Tweedie's formula gives the matching denoiser: $\mathbb{E}[x_0\mid x_t] = \big(x_t + (1-\bar\alpha_t)\,s(x_t)\big)/\sqrt{\bar\alpha_t}$.

---

## 2. The continuous-time view: SDEs and the probability-flow ODE

Let the number of steps go to infinity. The forward noising becomes a stochastic differential equation, $dx = f(x,t)\,dt + g(t)\,dW$. A classical result (Anderson, 1982) says it can be **run backwards in time** using the score:

```math
dx = \big[f(x,t) - g(t)^2\,\nabla_x\log p_t(x)\big]\,dt + g(t)\,d\bar W .
```

There's also a **deterministic** process with the *same marginal distributions* $p_t$ at every time, the **probability-flow ODE**:

```math
\frac{dx}{dt} = f(x,t) - \tfrac12 g(t)^2\,\nabla_x\log p_t(x).
```

Why this matters:

- **One trained score network supports many samplers**: stochastic (the reverse SDE, which is DDPM) or deterministic (the ODE).
- **DDIM is essentially an ODE discretization.** The ODE's trajectories are smooth, and there's no injected noise to average out, so large steps with an ODE solver (DDIM, DPM-Solver, Heun) reach good samples in **20–50 steps instead of 1,000**, using the same trained model.
- **A deterministic map from noise to data** enables latent inversion (image → noise), interpolation, and editing.

---

## 3. Classifier-free guidance, as scores

By Bayes, $\nabla\log p(x\mid c) = \nabla\log p(x) + \nabla\log p(c\mid x)$. CFG with weight $w$ samples from a **sharpened** distribution $\propto p(x)\,p(c\mid x)^w$, whose score is

```math
\nabla\log p(x) + w\big(\nabla\log p(x\mid c) - \nabla\log p(x)\big) .
```

In noise-prediction form, that's exactly GEN-04's $\varepsilon_\varnothing + w(\varepsilon_c - \varepsilon_\varnothing)$. A large $w$ over-weights "looks like the prompt" relative to "looks like a natural image": it pushes samples **off the data manifold** (oversaturated colours, artifacts) and collapses diversity. Moderate $w$ (about 3–8) trades a little diversity for strong prompt adherence.

---

## 4. Architectures: from U-Nets to DiT

Latent diffusion (GEN-04 §3.5) originally used a convolutional U-Net with cross-attention to the text. **DiT (Diffusion Transformer)** patchifies the latent into tokens and runs a plain transformer (DL-06). The time step and class or text conditioning enter through **adaptive LayerNorm** (adaLN): the conditioning predicts each block's scale and shift.
DiTs scale predictably with compute, just like LLMs, which is why recent image and video models use them.

---

## 5. Flow matching / rectified flow

### 5.1 The idea: learn a velocity field that carries noise to data

Choose a simple **path** between a noise sample $x_0\sim\mathcal N(0,I)$ and a data sample $x_1$. The straight line is the natural choice:

```math
x_t = (1 - t)\,x_0 + t\,x_1,\qquad t\in[0, 1].
```

Along this line, the velocity is constant: $\frac{dx_t}{dt} = x_1 - x_0$. Train a network $v_\theta(x, t)$ to predict it:

```math
\mathcal{L}_\text{FM} = \mathbb{E}_{t,\ x_0,\ x_1}\big\lVert v_\theta(x_t, t) - (x_1 - x_0)\big\rVert^2 .
```

That's the whole training procedure: sample noise, a data point and a time, interpolate, and regress the difference. **To generate**, integrate the ODE $\frac{dx}{dt} = v_\theta(x,t)$ from $t=0$ (noise) to $t=1$ (data) with any ODE solver.

### 5.2 Why the target is "wrong" but the result is right

Many different $(x_0, x_1)$ pairs pass through the same point $x_t$, with different velocities. The regression learns their **average**, $v^*(x,t) = \mathbb{E}[x_1 - x_0\mid x_t = x]$, the **marginal velocity field**. It's the same "noisy target, correct conditional mean" logic as denoising score matching (§1.1).
That averaged field moves the whole noise distribution onto the data distribution along the chosen probability path. Flow matching with Gaussian paths is **equivalent to diffusion with a particular schedule and parameterization**. It's a cleaner way to write the same family of models, with simpler maths, straighter paths, and no SDE machinery.

### 5.3 Why few-step sampling can still fail, and the fixes

Each *conditional* path is straight, but the *marginal* field isn't: paths from different pairs **cross**, and averaging bends the field. Curved trajectories need many small Euler steps. A 4-step Euler sampler cuts corners and lands off the data manifold. That's the lesson's debug question.
Fixes:

- **a higher-order solver** (Heun, RK), with more accuracy per step;
- **rectified flow / reflow**: generate (noise, sample) pairs with the trained model, then retrain on *those* pairs. They no longer cross, so the new field is straighter, and few-step sampling works;
- **distillation / consistency models**: train a student to jump straight to the endpoint in 1–4 steps;
- a better choice of time discretization (more steps where the curvature is high).

```python
import numpy as np, torch, torch.nn as nn
rng = np.random.default_rng(0); torch.manual_seed(0)

# --- Scores of a 1-D Gaussian mixture, and Langevin sampling from them -----------------------------------
w, m, s = np.array([0.3, 0.7]), np.array([-2.0, 2.0]), np.array([0.5, 0.8])
def pdf(x): return np.sum(w * np.exp(-0.5 * ((x[:, None] - m) / s) ** 2) / (s * np.sqrt(2 * np.pi)), 1)
def score(x):                                    # d/dx log p(x), no normalizing constant needed
    comp = w * np.exp(-0.5 * ((x[:, None] - m) / s) ** 2) / s
    return np.sum(comp * (-(x[:, None] - m) / s**2), 1) / comp.sum(1)
x = rng.normal(size=20000) * 3
for _ in range(2000):
    x = x + 0.01 * score(x) + np.sqrt(2 * 0.01) * rng.normal(size=x.shape)
print(f"Langevin: fraction near +2 = {np.mean(x > 0):.3f} (target 0.7); mean {x.mean():.3f} (target {w @ m:.3f})")
# Both modes are found, but the 30/70 weights are NOT recovered: walkers rarely cross the low-density gap,
# so the split inherited from the starting points persists. That is exactly why score models use MANY noise
# levels (annealed Langevin = diffusion): at high noise the modes merge, and walkers distribute correctly
# before the noise is lowered.

# --- Denoising score matching: the average of -eps/sigma given x_t is the score of the noised density ------------
sigma = 0.5
x0 = np.where(rng.random(400_000) < 0.3, rng.normal(-2, 0.5, 400_000), rng.normal(2, 0.8, 400_000))
eps = rng.normal(size=x0.shape); xt = x0 + sigma * eps
bins = np.linspace(-3, 3, 7)
idx = np.digitize(xt, bins)
for b in [2, 4, 6]:
    sel = idx == b; centre = xt[sel].mean()
    # the score of the noised mixture: same formula with variances s^2 + sigma^2
    comp = w * np.exp(-0.5 * (centre - m) ** 2 / (s**2 + sigma**2)) / np.sqrt(s**2 + sigma**2)
    true_score = np.sum(comp * (-(centre - m) / (s**2 + sigma**2))) / comp.sum()
    print(f"x_t ≈ {centre:+.2f}: mean of -eps/sigma = {np.mean(-eps[sel] / sigma):+.3f}   true noised score = {true_score:+.3f}")
```

```python
# --- Flow matching on 2-D data: train, then sample with 4 vs 100 Euler steps, and Heun ------------------------------
def data(n):
    c = torch.tensor([[2., 2.], [-2., 2.], [2., -2.], [-2., -2.]])
    return c[torch.randint(0, 4, (n,))] + 0.2 * torch.randn(n, 2)
v = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
opt = torch.optim.Adam(v.parameters(), lr=2e-3)
for step in range(3000):
    x1 = data(512); x0 = torch.randn(512, 2); t = torch.rand(512, 1)
    xt = (1 - t) * x0 + t * x1
    loss = ((v(torch.cat([xt, t], 1)) - (x1 - x0)) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

def sample(n_steps, heun=False, n=4000):
    x = torch.randn(n, 2); dt = 1.0 / n_steps
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((n, 1), i * dt)
            k1 = v(torch.cat([x, t], 1))
            if heun:
                k2 = v(torch.cat([x + dt * k1, t + dt], 1)); x = x + dt * (k1 + k2) / 2
            else:
                x = x + dt * k1
    return x
quality = lambda x: (x.abs() - 2).norm(dim=1).mean().item()      # mean distance to the nearest centre
print(f"data: {quality(data(4000)):.3f}")
for n_steps, heun in [(1, False), (4, False), (4, True), (100, False)]:
    print(f"{'Heun' if heun else 'Euler'} {n_steps:3d} steps: mean distance to the nearest mode {quality(sample(n_steps, heun)):.3f}")
```

---

## Pitfalls & misconceptions

- **"Score models and diffusion models are different things."** They're the same model family: noise prediction is score estimation (§1.1).
- **Assuming fewer sampling steps is free.** Discretization error grows with the trajectory's curvature. Use better solvers or distillation.
- **Huge guidance weights** to "fix" a weak model.
- **Thinking flow matching's straight conditional paths imply straight sampling trajectories.** The marginal field curves where paths cross.
- **Mismatching the time convention.** Some codebases use $t = 0$ for data, others for noise.

## Cheat sheet

| Item | Formula |
|---|---|
| Score | $s(x) = \nabla_x\log p(x)$ (no $Z$ needed) |
| Langevin | $x \leftarrow x + \eta s(x) + \sqrt{2\eta}z$ |
| Denoising score matching target | $-\varepsilon/\sigma$ (its conditional mean is the true noised score) |
| Noise ↔ score | $s_\theta = -\varepsilon_\theta/\sqrt{1-\bar\alpha_t}$ |
| Probability-flow ODE | $\dot x = f - \frac12 g^2\nabla\log p_t$ |
| CFG | $\nabla\log p(x) + w(\nabla\log p(x\mid c) - \nabla\log p(x))$ |
| Flow-matching path / target | $x_t = (1-t)x_0 + tx_1$; regress $x_1 - x_0$ |
| Sampling | integrate $\dot x = v_\theta(x,t)$ from noise to data |

## Answer sketches for the lesson's self-check

<details>
<summary>1. What is the score, and why can it be learned without the normalizing constant?</summary>

$\nabla_x\log p(x)$. Since $\log p = \log\tilde p - \log Z$ and $Z$ is constant in $x$, the gradient ignores $Z$. With denoising score matching, the training target $-\varepsilon/\sigma$ is computable from the noise you added yourself.
</details>

<details>
<summary>2. How are DDPM's noise prediction and score estimation related?</summary>

$s_\theta(x_t, t) = -\varepsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$. The noise-prediction loss is denoising score matching, reweighted, and its optimum predicts the conditional mean of the noise, which is proportional to the score of the noised distribution (the demo checks this numerically).
</details>

<details>
<summary>3. Why does DDIM allow far fewer steps with the same model?</summary>

It follows the deterministic probability-flow ODE (same marginals as the diffusion), whose smooth trajectories can be integrated accurately with large steps. The trained score or noise network is unchanged; only the sampler differs.
</details>

<details>
<summary>4. What does CFG combine, and what goes wrong at high scales?</summary>

The unconditional and conditional scores (or noise predictions), extrapolating toward the condition: sampling from $p(x)p(c\mid x)^w$. At high $w$, samples are pushed off the natural-image manifold: oversaturation, artifacts, and lost diversity.
</details>

<details>
<summary>5. Flow matching's regression target on a straight path.</summary>

The constant velocity $x_1 - x_0$ (data minus noise) at the point $x_t = (1-t)x_0 + tx_1$. The network learns its conditional mean, the marginal velocity field.
</details>

<details>
<summary>6. Why does latent space make high resolution feasible?</summary>

The VAE compresses the image about 48×, so the denoiser processes far fewer positions. Convolution and especially attention costs drop dramatically, while perceptual quality is preserved by the autoencoder.
</details>

<details>
<summary>7. 4 Euler steps look terrible, 100 look good.</summary>

The learned marginal velocity field is curved (the conditional straight paths cross, and averaging bends the field), so coarse Euler steps leave the data manifold. Use a higher-order solver (the demo's 4-step Heun beats 4-step Euler), reflow/rectification to straighten the field, or distillation and consistency training for few-step sampling.
</details>

## Where this leads

**Path 3 is complete.** You can build, evaluate, secure, align, serve, and generate. Next: [PROD-01 notes](../production/01-ml-system-design.md) (Path 4) puts all of it into production systems.
For vision, continue with [CV-01 notes](../vision/01-object-detection-segmentation.md).
