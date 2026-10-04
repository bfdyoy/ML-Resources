# DL-03 notes: Training Deep Networks Well

[← Lesson DL-03](../../lessons/deep-learning/03-training-deep-networks.md) · [All notes](../README.md) · [← DL-02 notes](02-pytorch-fluency.md) · Next: [DL-04 notes →](04-cnns-computer-vision.md)

> **Reading time** ≈ 70 min. **You need:** [DL-01 notes](01-neural-networks-from-scratch.md) §3 (the backward pass), [MATH-02 notes](../math/02-calculus-optimization.md) Block B (GD, conditioning, momentum), and [MATH-03 notes](../math/03-probability-statistics.md) §A3 (variance rules).

---

## Where we are

A 2-layer MLP trains easily. A 50-layer one, naively set up, doesn't train at all. This lesson is the toolkit that makes depth work. Each tool fixes a specific failure:

| Failure | Fix |
|---|---|
| Signals that shrink or explode with depth | Initialization, residual connections |
| Awkward, badly conditioned loss landscapes | Normalization, adaptive optimizers, schedules |
| Overfitting | Weight decay, dropout, augmentation |
| Bugs | A debugging recipe |

---

## 1. Why deep nets fail: products of many factors

The backward pass multiplies the error by one Jacobian per layer, $\delta_{l} = W_{l+1}^\top\big(\delta_{l+1}\odot\phi'(a_{l+1})\big)$ (DL-01 §3). Over $L$ layers, that's a product of $L$ matrices.
If their typical "gain" is a bit below 1, the gradient **vanishes** exponentially ($0.9^{50} \approx 0.005$). If it's a bit above 1, it **explodes** ($1.1^{50} \approx 117$). The forward activations behave the same way.
**The goal of initialization is a gain of about 1 per layer.**

### 1.1 Deriving Xavier and He initialization

Take one pre-activation, $a = \sum_{j=1}^{n_\text{in}} w_j x_j$, with independent, zero-mean weights and inputs. By the variance rules (MATH-03 §A3):

```math
\operatorname{Var}(a) = n_\text{in}\,\operatorname{Var}(w)\,\mathbb{E}[x^2].
```

- **Linear or tanh (near 0):** $\mathbb{E}[x^2] \approx \operatorname{Var}(a_\text{prev})$. Keeping $\operatorname{Var}(a) = \operatorname{Var}(a_\text{prev})$ requires $\operatorname{Var}(w) = 1/n_\text{in}$.
  Balancing the backward pass as well gives the **Xavier/Glorot** compromise, $\operatorname{Var}(w) = 2/(n_\text{in} + n_\text{out})$.
- **ReLU:** it zeroes half its inputs, so $\mathbb{E}[x^2] = \tfrac12\operatorname{Var}(a_\text{prev})$ (for symmetric $a$). To compensate, double the weight variance:
  $\operatorname{Var}(w) = 2/n_\text{in}$, so the standard deviation is $\sqrt{2/n_\text{in}}$. That's **He (Kaiming) initialization**.

The demo below pushes a signal through 50 ReLU layers. With a too-small init it vanishes, with a too-big one it explodes, and with He init it stays put.

### 1.2 Residual connections: a gradient highway

Make each block learn a *correction*: $x_{l+1} = x_l + F(x_l)$. Then

```math
\frac{\partial x_{L}}{\partial x_l} = \prod_{k=l}^{L-1}\Big(I + \frac{\partial F_k}{\partial x_k}\Big) = I + (\text{other terms}),
```

so there's always an **identity path** that carries the gradient back undiminished, however deep the network. A block that isn't needed can learn $F \approx 0$ and become the identity.
This is what made 100+ layer ResNets (DL-04) and every transformer (DL-06) trainable.

---

## 2. Normalization layers

**BatchNorm**, for each feature over the batch:

```math
\hat x = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}},\qquad y = \gamma\hat x + \beta .
```

- **Why it helps:** it keeps each layer's input distribution stable, and it makes the loss landscape smoother (better conditioned), so larger learning rates work. The learnable $\gamma, \beta$ let the network undo the normalization if that's best.
- **Train vs eval:** in training, $\mu_B$ and $\sigma_B$ come from the current batch, which makes every example's output depend on its batch-mates. That adds noise, and it's a mild regularizer.
  In eval mode it uses **running averages** collected during training, so predictions are deterministic and work for a single example. Forgetting `model.eval()` therefore changes your predictions (DL-02).
- **Weak spots:** small batches give noisy statistics, and sequence models have variable lengths.

**LayerNorm** normalizes over the *features of each example*, with no dependence on the batch, so it behaves the same in training and inference. It's the standard in transformers.
**RMSNorm** drops the mean subtraction ($x/\text{RMS}(x)\cdot\gamma$). It's cheaper and works as well (DL-08).

---

## 3. Optimizers

### 3.1 From SGD to Adam

- **SGD + momentum** (MATH-02 §B4): it averages the gradient direction and damps zig-zags.
- **RMSProp:** it divides each parameter's step by a running RMS of its gradients. Parameters with consistently large gradients get smaller steps, and rare or small ones get bigger steps. It's a cheap, diagonal fix for bad conditioning.
- **Adam = momentum + RMSProp + bias correction.** With gradient $g_t$:

```math
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, & v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\\
\hat m_t &= \frac{m_t}{1-\beta_1^t}, & \hat v_t &= \frac{v_t}{1-\beta_2^t},\\
\theta_t &= \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}. &&
\end{aligned}
```

**Why bias correction?** $m_0 = 0$, so unrolling gives $m_t = (1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} g_i$, whose expectation is $(1-\beta_1^t)\,\mathbb{E}[g]$ for stationary gradients. Early on, that's far too small:
at $t = 1$ with $\beta_1 = 0.9$, it's 10% of the true mean. Dividing by $1 - \beta_1^t$ removes the bias. Without it, the early $\hat v$ would be tiny too, so the early steps would be erratic.

### 3.2 AdamW: why "decoupled" weight decay matters

With **L2 in the loss**, the penalty's gradient $\lambda\theta$ is added to $g_t$, and then **divided by $\sqrt{\hat v_t}$** along with everything else. Parameters with large gradient history get *less* decay than intended,
so the regularization becomes uneven and tied to gradient scale. **AdamW** applies the decay **directly to the weights**, separately from the adaptive step:

```math
\theta_t = \theta_{t-1} - \eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\Big).
```

Every weight then shrinks at the same relative rate. AdamW generalizes better, and it's the default for transformers. (For plain SGD, L2 and weight decay are equivalent. The difference only appears with adaptive methods.)

### 3.3 Learning-rate schedules

- **Warmup** (a linear ramp over the first few hundred to few thousand steps): early on, Adam's $\hat v$ estimates come from very few gradients and are unreliable, and the network is in a chaotic region where large steps can push it somewhere bad that it never recovers from.
  Large batches allow (and need) large learning rates, which makes this worse. Warmup lets the statistics settle first.
- **Cosine decay:** $\eta_t = \eta_\text{min} + \tfrac12(\eta_\text{max} - \eta_\text{min})(1 + \cos(\pi t/T))$. A smooth anneal lets the iterates settle into a minimum, reducing the "bouncing" caused by SGD noise.
- **One-cycle:** warm up to a high learning rate, then anneal to very low. It's fast and robust for CNNs.

---

## 4. Regularization for deep nets

- **Weight decay:** see AdamW above.
- **Dropout:** during training, zero each unit with probability $p$ and scale the survivors by $1/(1-p)$ (*inverted dropout*), so the expected activation is unchanged. At eval time, do nothing.
  Two views: it stops units from **co-adapting** (no unit can rely on a specific other unit), and it trains an implicit **ensemble** of exponentially many thinned networks that eval mode averages.
- **Data augmentation:** encode invariances you *know* hold (a flipped cat is still a cat). It's often the strongest regularizer for vision.
- **Early stopping:** pick the checkpoint with the best validation loss.
- **Label smoothing:** train against $(1-\varepsilon)\,\text{onehot} + \varepsilon/K$. It discourages extreme logits and improves calibration.

---

## 5. The debugging recipe

1. **Check the loss at initialization.** With $K$ classes and near-uniform initial predictions, the cross-entropy should be about $\ln K$: $\ln 10 \approx 2.303$ for 10 classes. Much higher means the initial logits are too confident, so scale down the last layer.
   A wrong value also catches wrong labels or a wrong loss.
2. **Overfit one batch.** A healthy model and pipeline should drive the loss on 32 fixed examples to about 0. **If it can't, there's a bug** (not a regularization problem): the labels don't match the inputs, the learning rate is wrong, a layer is frozen, or `zero_grad` or `step` is missing.
3. **Visualize the activation and gradient statistics per layer.** Look for dead ReLUs (all zeros), saturated tanh units, and gradient norms that differ by orders of magnitude across layers.
4. **Then scale up:** add data, then regularization, then tune.

(This is a compressed form of Karpathy's *Recipe*, which the lesson asks you to read in full.)

```python
import torch, torch.nn as nn, math
torch.manual_seed(0)

# --- Signal propagation through 50 ReLU layers -------------------------------------
x0 = torch.randn(1000, 512)
for name, std_fn in [("too small (0.01)", lambda n: 0.01), ("too big (0.1)", lambda n: 0.1), ("He sqrt(2/n)", lambda n: math.sqrt(2 / n))]:
    h = x0
    for _ in range(50):
        h = torch.relu(h @ (torch.randn(512, 512) * std_fn(512)))
    print(f"{name:17s} activation std after 50 layers: {h.std().item():.3e}")

# --- Adam by hand matches torch.optim.Adam --------------------------------------------
w_ref = torch.tensor([1.0, -2.0], requires_grad=True)
opt = torch.optim.Adam([w_ref], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
w, m, v = torch.tensor([1.0, -2.0]), torch.zeros(2), torch.zeros(2)
f = lambda w: (w[0] - 3) ** 2 + 10 * (w[1] + 1) ** 2
for t in range(1, 51):
    opt.zero_grad(); f(w_ref).backward(); opt.step()
    g = torch.tensor([2 * (w[0] - 3), 20 * (w[1] + 1)])
    m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
    m_hat, v_hat = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
    w = w - 0.1 * m_hat / (v_hat.sqrt() + 1e-8)
print("by hand:", w.tolist(), "\ntorch  :", w_ref.detach().tolist())
```

```python
# --- AdamW vs Adam + L2: different effective decay for big-gradient parameters ----------------
def run(opt_cls, **kw):
    p = torch.tensor([5.0, 5.0], requires_grad=True)
    opt = opt_cls([p], lr=0.01, **kw)
    for _ in range(500):
        opt.zero_grad()
        g_scale = torch.tensor([100.0, 0.01])                # pure-noise "data" gradients of very different sizes
        loss = (p * g_scale * torch.randn(2)).sum()
        loss.backward(); opt.step()
    return p.detach()
torch.manual_seed(1); print("Adam + L2 (weight_decay in Adam):", run(torch.optim.Adam, weight_decay=0.1).round(decimals=3).tolist())
torch.manual_seed(1); print("AdamW (decoupled)               :", run(torch.optim.AdamW, weight_decay=0.1).round(decimals=3).tolist())
print("-> with L2-in-Adam, the large-gradient parameter (index 0) barely decays; AdamW shrinks both at a similar rate")

# --- Loss at init, and overfitting one batch -------------------------------------------------
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
with torch.no_grad(): model[-1].weight.mul_(0.01); model[-1].bias.zero_()   # near-uniform initial predictions
xb, yb = torch.randn(32, 20), torch.randint(0, 10, (32,))
loss_fn = nn.CrossEntropyLoss()
print(f"loss at init {loss_fn(model(xb), yb).item():.3f}  vs ln(10) = {math.log(10):.3f}")
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(300):
    opt.zero_grad(); l = loss_fn(model(xb), yb); l.backward(); opt.step()
print(f"loss after overfitting one batch: {l.item():.4f}")
```

---

## Pitfalls & misconceptions

- **Default init with custom activations.** Match the init to the non-linearity.
- **No warmup with Adam at high learning rates or large batches.** You get early divergence or loss spikes.
- **Adam with `weight_decay`, when you meant AdamW.** In PyTorch, `Adam(weight_decay=...)` is L2-in-the-gradient.
- **Tuning regularization before you can overfit a batch.** Fix the bugs first.
- **BatchNorm with tiny or varying batches.** Use GroupNorm or LayerNorm.

## Cheat sheet

| Item | Formula / value |
|---|---|
| He init | $w \sim \mathcal N(0,\ 2/n_\text{in})$ |
| Xavier init | $\operatorname{Var}(w) = 2/(n_\text{in}+n_\text{out})$ |
| Residual block | $x + F(x)$; its Jacobian contains $I$ |
| BatchNorm | normalize per feature over the batch; running statistics at eval |
| Adam | $\theta \mathrel{-}= \eta\,\hat m/(\sqrt{\hat v}+\epsilon)$, with $\hat m = m/(1-\beta_1^t)$ |
| AdamW | add $\eta\lambda\theta$ outside the adaptive scaling |
| Initial CE loss | $\ln K$ |
| Inverted dropout | scale by $1/(1-p)$ during training |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why sqrt(2/fan_in) for ReLU?</summary>

$\operatorname{Var}(a) = n_\text{in}\operatorname{Var}(w)\mathbb{E}[x^2]$, and ReLU makes $\mathbb{E}[x^2] = \tfrac12\operatorname{Var}(a_\text{prev})$. Preserving the variance layer to layer needs $\operatorname{Var}(w) = 2/n_\text{in}$ (§1.1). The demo shows a stable std after 50 layers.
</details>

<details>
<summary>2. What does BatchNorm solve, and why train vs eval?</summary>

It stabilizes each layer's input statistics and smooths the landscape, allowing higher learning rates and faster training. In training it uses the batch's statistics. At eval it uses running averages, so outputs are deterministic and don't depend on the other examples in the batch.
</details>

<details>
<summary>3. AdamW vs Adam + L2.</summary>

L2 adds $\lambda\theta$ to the gradient, which Adam then divides by $\sqrt{\hat v}$, so parameters with large gradients decay less. AdamW decays the weights directly, uniformly, outside the adaptive scaling. That's better regularization and generalization (see the demo).
</details>

<details>
<summary>4. Why warmup with Adam and large batches?</summary>

Early second-moment estimates are based on few samples and are unreliable, and the initial landscape is chaotic. Large learning rates (typical with large batches) can then take destructive steps. Warmup ramps the learning rate up while the statistics and the network settle.
</details>

<details>
<summary>5. The loss at init for a 10-class softmax.</summary>

About $\ln 10 \approx 2.30$, for near-uniform predictions. Check it to catch overconfident initial logits (fix by scaling down the last layer) and wrong losses or labels.
</details>

<details>
<summary>6. You can't overfit a single batch of 32.</summary>

There's a bug in the model or pipeline, not a generalization problem. Check: the labels align with the inputs, the loss gets logits, `zero_grad`/`backward`/`step` are all called, the parameters are actually in the optimizer, nothing is unintentionally frozen, and the learning rate is reasonable.
</details>

## Where this leads

Next: [DL-04 notes](04-cnns-computer-vision.md). With a reliable training toolkit, we can now add **structure** to networks. The first and most influential is the convolution,
which builds the geometry of images into the architecture itself.
