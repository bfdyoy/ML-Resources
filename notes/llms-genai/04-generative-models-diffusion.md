# GEN-04 notes: Generative Models: VAEs, GANs & Diffusion

[← Lesson GEN-04](../../lessons/llms-genai/04-generative-models-diffusion.md) · [All notes](../README.md) · [← GEN-08 notes](08-efficient-llm-inference.md) · Next: [GEN-10 notes →](10-advanced-diffusion-flow-matching.md)

> **Reading time** ≈ 75 min. **You need:** [MATH-03 notes](../math/03-probability-statistics.md) §A4 and §B–C (Gaussians, likelihood, KL), [DL-03 notes](../deep-learning/03-training-deep-networks.md) (training), and [DL-04 notes](../deep-learning/04-cnns-computer-vision.md) (convolutions; the U-Net uses them). GEN-04 can follow straight after DL-04.

---

## Where we are

Classifiers learn $p(y\mid x)$. **Generative models** learn $p(x)$ itself, so they can *sample new data*. For text, the chain rule made this easy (next-token prediction). For images, with millions of continuous, correlated pixels, there are three classic strategies:

- **VAEs:** an explicit, approximate likelihood.
- **GANs:** an implicit model trained adversarially.
- **Diffusion:** learn to reverse a gradual noising process.

Diffusion won, but each idea survives in modern systems: Stable Diffusion runs diffusion *inside a VAE's latent space*.

---

## 1. Variational autoencoders

### 1.1 The model

A latent variable $z \sim \mathcal N(0, I)$ (low-dimensional), and a **decoder** $p_\theta(x\mid z)$ that turns $z$ into an image. The likelihood $p_\theta(x) = \int p_\theta(x\mid z)p(z)\,dz$ is intractable: that integral is over all $z$.

### 1.2 The ELBO, derived

Introduce an **encoder** $q_\phi(z\mid x)$ that guesses which $z$ produced $x$. For any $q$:

```math
\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big] - \mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big)}_{\text{ELBO}} + \underbrace{\mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)\big)}_{\ge 0}.
```

(Write $\log p(x) = \mathbb{E}_q[\log p(x,z) - \log q] + \mathbb{E}_q[\log q - \log p(z\mid x)]$ and regroup.)

The last KL is non-negative, so the **ELBO is a lower bound on the log-likelihood**. Maximize it over $\theta$ and $\phi$ together. Its two terms pull in opposite directions:

- **Reconstruction:** $\mathbb{E}_q[\log p(x\mid z)]$. Codes must carry enough information to rebuild $x$. With a Gaussian decoder, this is minus an MSE (MATH-03 §B2).
- **Regularization:** $\mathrm{KL}(q(z\mid x)\Vert\mathcal N(0,I))$. Each code distribution must stay close to the prior, so the latent space is **smooth and filled in**, and samples from $\mathcal N(0,I)$ decode to sensible images.

For a diagonal Gaussian encoder $q = \mathcal N(\mu, \operatorname{diag}\sigma^2)$, the KL has a closed form:

```math
\mathrm{KL}\big(\mathcal N(\mu,\sigma^2)\,\Vert\,\mathcal N(0,1)\big) = \tfrac12\sum_j\big(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\big).
```

### 1.3 The reparameterization trick

We need gradients of $\mathbb{E}_{z\sim q_\phi}[f(z)]$ with respect to $\phi$. But sampling isn't differentiable: $\phi$ decides the *distribution* we sample from. Rewrite the sample as a deterministic function of $\phi$ plus independent noise:

```math
z = \mu_\phi(x) + \sigma_\phi(x)\odot\varepsilon,\qquad \varepsilon\sim\mathcal N(0, I).
```

The randomness now lives in $\varepsilon$, which doesn't depend on $\phi$, so gradients flow through $\mu$ and $\sigma$ by the ordinary chain rule. The same trick is behind diffusion's training, and much of RL and Bayesian deep learning.

**Why VAE samples are blurry:** maximizing likelihood is a *forward-KL* fit (MATH-03 §C3), which is **mass-covering**. When unsure, the decoder averages over plausible images. Averages of sharp images are blurry.

---

## 2. GANs

### 2.1 The game

A **generator** $G$ maps noise to images. A **discriminator** $D$ outputs the probability that an image is real:

```math
\min_G\max_D\ \ \mathbb{E}_{x\sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))].
```

For a fixed $G$, the best discriminator is $D^*(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_g(x)}$. Plug it back in and the generator is minimizing $2\,\mathrm{JSD}(p_\text{data}\Vert p_g) - \log 4$, a symmetric divergence. Its optimum is $p_g = p_\text{data}$.
In practice, the generator maximizes $\log D(G(z))$ instead (the *non-saturating* loss), because the original loss gives vanishing gradients early on, when $D$ easily wins.

### 2.2 Failure modes

- **Mode collapse:** $G$ finds a few outputs that fool $D$ and produces only those. Nothing in the objective directly punishes **missing** modes: $G$ is never asked to explain each real image, only to make each of its own samples look real.
  A VAE *is* asked to explain every training image (through the likelihood), so dropping a mode costs it likelihood. VAEs are blurry but cover the data. GANs are sharp but can drop modes.
- **Instability:** it's a two-player game, not a minimization. The players can oscillate, and gradients vanish or explode. Tricks like spectral normalization, the Wasserstein loss, and careful learning rates are needed.

---

## 3. Diffusion models (DDPM)

### 3.1 The forward process: destroying data, gradually

Add a little Gaussian noise at each of $T$ steps (around 1,000), with a schedule $\beta_t$:

```math
q(x_t\mid x_{t-1}) = \mathcal N\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big).
```

Gaussians compose, so you can **jump straight to any step**. With $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s\le t}\alpha_s$:

```math
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \varepsilon\sim\mathcal N(0, I).
```

As $t\to T$, $\bar\alpha_t\to0$, and $x_T$ is pure noise. The $\sqrt{\cdot}$ factors keep the total variance at 1.

### 3.2 Training: predict the noise

A network $\varepsilon_\theta(x_t, t)$, usually a U-Net or a transformer, sees a noisy image and the step $t$, and predicts the noise that was added:

```math
\mathcal{L} = \mathbb{E}_{x_0,\,t,\,\varepsilon}\Big[\big\lVert \varepsilon - \varepsilon_\theta\big(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon,\ t\big)\big\rVert^2\Big].
```

It's **plain regression**: stable, with no adversary and no mode collapse. (It's a reweighted ELBO, so it's a likelihood method underneath.)

**Noise prediction vs clean-image prediction are equivalent.** Given $x_t$ and $t$, the relation $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon$ is linear, so

```math
\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon}{\sqrt{\bar\alpha_t}} .
```

Predicting one determines the other. They differ only in how the loss is weighted across noise levels. (A third choice, "v-prediction", is a mix of the two.) [GEN-10 notes](10-advanced-diffusion-flow-matching.md) shows that predicting the noise is also estimating the **score** $\nabla_x\log p$.

### 3.3 Sampling: denoise step by step

Start from $x_T\sim\mathcal N(0, I)$, and for $t = T, \dots, 1$:

```math
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t, t)\Big) + \sigma_t z,\qquad z\sim\mathcal N(0,I)\ (z = 0 \text{ at the last step}).
```

Each step removes a bit of the predicted noise and adds back a smaller amount of fresh noise. That takes $T$ network evaluations, which is slow. GEN-10 covers samplers that need far fewer steps.

### 3.4 Conditioning and classifier-free guidance (CFG)

Train one network both **with** the condition $c$ (a text embedding) and **without** it (drop $c$ about 10% of the time). At sampling time, extrapolate away from the unconditional prediction:

```math
\tilde\varepsilon = \varepsilon_\theta(x_t, \varnothing) + w\big(\varepsilon_\theta(x_t, c) - \varepsilon_\theta(x_t, \varnothing)\big).
```

$w = 1$ is plain conditioning. $w > 1$ (around 3–8) gives more prompt adherence and sharper images, with less diversity. A very large $w$ gives oversaturated, artifact-ridden images.

### 3.5 Latent diffusion (Stable Diffusion)

Running diffusion on 512×512×3 pixels (786k values) is expensive, and most bits encode imperceptible detail. So:

1. Train a **VAE** (with perceptual and adversarial losses) that compresses images to a 64×64×4 latent, **48× fewer values**.
2. Run diffusion **in that latent space**, with a U-Net or transformer denoiser and **cross-attention** to the text-encoder tokens.
3. Decode the final latent with the VAE decoder.

The denoiser's compute, and especially attention's $O(T^2)$ over spatial positions, drops dramatically, which makes high resolutions feasible.

**Debug: blurry blobs after training.** Check:

1. **data scaling:** images in $[-1, 1]$, as the noise schedule assumes;
2. **the sampler:** the right $\bar\alpha_t$ indexing (an off-by-one error is classic), the same schedule as training, and enough steps;
3. **parameterization mismatch:** the network trained to predict $\varepsilon$ but sampled as if it predicted $x_0$ (or the reverse);
4. undertraining or a model that's too small, and whether you're sampling from the **EMA** weights;
5. whether the loss actually went down at *all* noise levels (log the loss per $t$ bucket).

```python
import numpy as np, torch, torch.nn as nn
rng = np.random.default_rng(0); torch.manual_seed(0)

# --- VAE pieces: closed-form KL vs Monte Carlo; reparameterization gradients -----------------------
mu, sigma = np.array([0.5, -1.0]), np.array([0.8, 1.5])
kl_closed = 0.5 * np.sum(mu**2 + sigma**2 - np.log(sigma**2) - 1)
z = mu + sigma * rng.normal(size=(200_000, 2))
log_q = -0.5 * np.sum(((z - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma**2), 1)
log_p = -0.5 * np.sum(z**2 + np.log(2 * np.pi), 1)
print(f"KL closed form {kl_closed:.4f}  Monte Carlo {np.mean(log_q - log_p):.4f}")

mu_t = torch.tensor(0.5, requires_grad=True); log_s = torch.tensor(0.0, requires_grad=True)
eps = torch.randn(100_000)
zt = mu_t + torch.exp(log_s) * eps                     # reparameterized sample
(zt ** 2).mean().backward()                            # E[z^2] = mu^2 + sigma^2
print(f"d/dmu E[z^2]: reparam {mu_t.grad.item():.3f} (exact {2*0.5:.3f}); d/dlog_sigma: {log_s.grad.item():.3f} (exact {2.0:.3f})")

# --- Forward diffusion: closed form == iterating the noising steps (in distribution) -----------------
T = 200
betas = np.linspace(1e-4, 0.05, T); alphas = 1 - betas; abar = np.cumprod(alphas)
x0 = np.full(100_000, 2.0)
x = x0.copy()
for t in range(T):
    x = np.sqrt(alphas[t]) * x + np.sqrt(betas[t]) * rng.normal(size=x.shape)
print(f"after {T} steps: iterated mean/std {x.mean():.3f}/{x.std():.3f}  closed form {np.sqrt(abar[-1])*2:.3f}/{np.sqrt(1-abar[-1]):.3f}")
```

```python
# --- A tiny DDPM on 2-D data (a mixture of 4 Gaussians), trained on CPU in seconds ------------------------
def sample_data(n):
    centers = torch.tensor([[2., 2.], [-2., 2.], [2., -2.], [-2., -2.]])
    return centers[torch.randint(0, 4, (n,))] + 0.3 * torch.randn(n, 2)

T = 100
betas = torch.linspace(1e-4, 0.08, T); alphas = 1 - betas; abar = torch.cumprod(alphas, 0)
net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
opt = torch.optim.Adam(net.parameters(), lr=2e-3)
for step in range(3000):
    x0 = sample_data(512); t = torch.randint(0, T, (512,)); eps = torch.randn_like(x0)
    xt = abar[t].sqrt()[:, None] * x0 + (1 - abar[t]).sqrt()[:, None] * eps
    loss = ((net(torch.cat([xt, (t / T)[:, None]], 1)) - eps) ** 2).mean()     # predict the noise
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    x = torch.randn(2000, 2)
    for t in reversed(range(T)):
        e = net(torch.cat([x, torch.full((len(x), 1), t / T)], 1))
        x = (x - betas[t] / (1 - abar[t]).sqrt() * e) / alphas[t].sqrt()
        if t > 0: x += betas[t].sqrt() * torch.randn_like(x)
quadrants = torch.stack([(x[:, 0] > 0) & (x[:, 1] > 0), (x[:, 0] < 0) & (x[:, 1] > 0),
                         (x[:, 0] > 0) & (x[:, 1] < 0), (x[:, 0] < 0) & (x[:, 1] < 0)]).float().mean(1)
print("final training loss:", round(loss.item(), 3))
print("fraction of samples per mode (target 0.25 each):", quadrants.numpy().round(3))
print("mean distance to the nearest centre:", (x.abs() - 2).norm(dim=1).mean().item().__round__(3), "(data: about 0.38)")
```

---

## Pitfalls & misconceptions

- **"GANs are obsolete."** Adversarial losses still live inside VAE decoders and distillation methods.
- **Comparing sample quality by eye on a few images.** Use FID or similar metrics plus human evaluation, over many samples.
- **Pixels not in $[-1,1]$**, or a mismatch between the training and sampling schedules.
- **Too much classifier-free guidance.** It oversaturates and collapses diversity.
- **Forgetting that latent diffusion quality is capped by its VAE.** Fine details lost by the autoencoder can't be recovered.

## Cheat sheet

| Item | Formula |
|---|---|
| ELBO | $\mathbb{E}_q[\log p(x\mid z)] - \mathrm{KL}(q(z\mid x)\Vert p(z))$ |
| Gaussian KL | $\frac12\sum(\mu^2+\sigma^2-\log\sigma^2-1)$ |
| Reparameterization | $z = \mu + \sigma\odot\varepsilon$ |
| Optimal discriminator | $p_\text{data}/(p_\text{data}+p_g)$ |
| Forward diffusion | $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon$ |
| DDPM loss | $\lVert\varepsilon - \varepsilon_\theta(x_t,t)\rVert^2$ |
| $\varepsilon \leftrightarrow x_0$ | $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\hat\varepsilon)/\sqrt{\bar\alpha_t}$ |
| CFG | $\varepsilon_\varnothing + w(\varepsilon_c - \varepsilon_\varnothing)$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why can't you backprop through sampling, and how does reparameterization fix it?</summary>

A sample drawn from $q_\phi$ isn't a differentiable function of $\phi$: the parameters change the distribution, not a value we compute. Writing $z = \mu_\phi + \sigma_\phi\varepsilon$ with parameter-free noise makes $z$ a differentiable function of $\phi$, so ordinary backprop works (the demo matches the exact gradients).
</details>

<details>
<summary>2. What do the two ELBO terms encourage?</summary>

Reconstruction: latents informative enough to rebuild the input. KL to the prior: code distributions close to $\mathcal N(0, I)$, which gives a smooth, well-covered latent space we can sample from.
</details>

<details>
<summary>3. What is mode collapse, and why doesn't a VAE suffer from it the same way?</summary>

The generator produces only a few kinds of output that fool the discriminator, dropping other modes. The GAN objective never makes $G$ account for every real example. A VAE maximizes the likelihood of *each* training example (mass-covering), so missing a mode is penalized. Its failure mode is blur instead.
</details>

<details>
<summary>4. Noise vs clean-image prediction: what is predicted, and why equivalent?</summary>

Typically the added noise $\varepsilon$. Given $x_t$ and $t$, $x_0$ and $\varepsilon$ are linked linearly by $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon$, so each prediction determines the other. The choice only changes the loss weighting across noise levels.
</details>

<details>
<summary>5. Why does Stable Diffusion work in a latent space?</summary>

A perceptual autoencoder compresses the image about 48× while keeping what matters visually. Diffusion over the small latent is far cheaper (fewer positions for the convolutions and attention), which makes high-resolution generation feasible.
</details>

<details>
<summary>6. Blurry blobs after training: three things to check.</summary>

Data scaling to $[-1,1]$. The sampler's schedule and indexing matching training (and enough steps). The parameterization ($\varepsilon$ vs $x_0$) being consistent between training and sampling. Also check undertraining, the EMA weights, and the loss per noise level (§3.5).
</details>

## Where this leads

Next: [GEN-10 notes](10-advanced-diffusion-flow-matching.md). DDPM needs about 1,000 slow steps, and its derivation hides a deeper structure. GEN-10 reveals it (scores, SDEs, ODEs), and arrives at **flow matching**, the simpler formulation behind many current image and video models.
