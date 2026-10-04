# DL-08 notes: Modern Architectures: RoPE, GQA, MoE & State-Space Models

[← Lesson DL-08](../../lessons/deep-learning/08-modern-architectures-moe-ssm.md) · [All notes](../README.md) · [← DL-07 notes](07-performance-gpus-mixed-precision.md) · Next: [DL-09 notes →](09-graph-neural-networks.md)

> **Reading time** ≈ 70 min. **You need:** [DL-06 notes](06-transformers.md) (the block, attention costs) and [DL-07 notes](07-performance-gpus-mixed-precision.md) (memory and compute arithmetic). For RoPE: 2-D rotation matrices.

---

## Where we are

A 2026 LLM block looks like the 2017 one, with a handful of upgrades. Each one fixes a specific weakness:

| Upgrade | Fixes |
|---|---|
| **RoPE** | Positions that generalize poorly |
| **RMSNorm, SwiGLU** | Small efficiency and quality wins |
| **GQA** | A KV cache too big to serve |
| **MoE** | More parameters without more FLOPs per token |
| **State-space models** | Quadratic attention on long sequences |

This note derives each one.

---

## 1. Rotary position embeddings (RoPE)

### 1.1 The idea

We want the attention score between positions $m$ and $n$ to depend on the **relative offset** $n - m$, not on absolute positions. That's what language needs: "the previous word" means the same thing everywhere.

Split $q$ and $k$ into 2-D pairs, and **rotate each pair by an angle proportional to the position**, $m\theta_i$ for pair $i$. For one pair, with the rotation matrix $R_\alpha$:

```math
q_m' = R_{m\theta} q,\quad k_n' = R_{n\theta} k \quad\Longrightarrow\quad q_m'^\top k_n' = q^\top R_{m\theta}^\top R_{n\theta}\, k = q^\top R_{(n-m)\theta}\, k .
```

The last step uses two rotation facts: $R_a^\top = R_{-a}$, and rotations compose by adding angles. **The absolute positions cancel, leaving only the offset.** Yet RoPE is applied to each token independently, so there's no pairwise computation and it's compatible with the KV cache.

### 1.2 Frequencies and context extension

Pair $i$ uses frequency $\theta_i = b^{-2i/d}$ (with base $b = 10{,}000$ or larger). Early pairs rotate fast (fine, local position), and later ones rotate slowly (coarse, long-range position), like the hands of a clock.

A model trained to length $L$ has never seen the angles beyond $L\theta_i$. To **extend the context**:

- **position interpolation** squeezes positions by $L/L'$, so new lengths map into familiar angles;
- **NTK-aware scaling** and **YaRN** instead change the base, or treat different frequencies differently: they keep the fast (local) frequencies intact and stretch the slow ones.

A short fine-tune at the new length then adapts the model.

---

## 2. RMSNorm and SwiGLU

- **RMSNorm:** $\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac1d\sum_j x_j^2 + \epsilon}}\odot\gamma$. It's LayerNorm without the mean subtraction or bias. Cheaper, and empirically just as good.
- **SwiGLU MLP:** $\text{FFN}(x) = W_2\big(\operatorname{SiLU}(W_1x)\odot W_3x\big)$, a gated MLP where one branch decides how much of the other passes. It has three matrices instead of two, so the hidden width is set to about $\tfrac{8}{3}d$ (instead of $4d$) to keep the parameter count equal: $3\cdot\tfrac83 d^2 = 8d^2$.

---

## 3. Attention variants that shrink the KV cache

### 3.1 The KV cache, by the numbers

During generation, every layer stores the keys and values of all previous tokens so they don't have to be recomputed (GEN-08). Its size per sequence is:

```math
\text{KV bytes} = 2 \times n_\text{layers} \times T \times n_\text{kv heads} \times d_\text{head} \times \text{bytes per value}.
```

*Example (a 7B-class model):* 32 layers, 32 heads of dimension 128, 4,096 tokens, BF16: $2\cdot32\cdot4096\cdot32\cdot128\cdot2 \approx 2.1$ GB **per sequence**. A batch of 32 conversations needs 69 GB for the cache alone.

### 3.2 MQA and GQA

- **Multi-query attention (MQA):** all query heads share **one** K/V head. The cache is $n_\text{heads}\times$ smaller, at some cost in quality.
- **Grouped-query attention (GQA):** query heads are split into $g$ groups, and each group shares one K/V head. With 32 query heads and **8 KV heads**, the cache shrinks **4×** (2.1 GB → 0.54 GB above), with quality close to full multi-head attention. It's the standard choice today.

### 3.3 Sliding-window and other sparse patterns

Each token attends only to the last $w$ tokens. The cost is $O(Tw)$ instead of $O(T^2)$, and the KV cache is capped at $w$. Stacking layers still propagates information further back (the receptive field grows by $w$ per layer, as in CNNs).
Models often interleave local and global attention layers.

---

## 4. Mixture of Experts (MoE)

### 4.1 The layer

Replace the single MLP with $E$ expert MLPs plus a small **router**. For each token, the router scores the experts, keeps the **top-$k$** (typically $k = 1$ or 2), and mixes their outputs:

```math
g = \operatorname{softmax}(W_r x),\qquad y = \sum_{i \in \text{TopK}(g)} \frac{g_i}{\sum_{j\in\text{TopK}} g_j}\, E_i(x).
```

Each token passes through only $k$ experts, so **the compute per token stays roughly constant while the parameter count grows with $E$**.

### 4.2 Total vs active parameters

Take Mixtral 8×7B: about **47B total** and about **13B active** parameters per token (attention is shared, and 2 of the 8 experts are used per token).

- **Memory** is set by the **total**: all the experts must be loaded (47B × 2 bytes ≈ 94 GB in BF16).
- **Compute per token** (FLOPs, and so latency at small batch sizes) is set by the **active** count (13B).
- **Quality** typically lands between a dense 13B and a dense 47B model: more parameters store more knowledge, at a fraction of the FLOPs.

### 4.3 Load balancing: why routers collapse

The router learns from the experts' outputs, and the experts improve only on the tokens they receive. An expert that gets slightly more tokens early becomes better, so it gets *more* tokens: **rich-get-richer collapse**. A few experts take everything, and the rest are dead weight.
Expert-parallel hardware is wasted, and tokens get dropped when an expert exceeds its *capacity*. The Switch Transformer's **auxiliary loss** pushes toward uniform use:

```math
\mathcal{L}_\text{aux} = \alpha\, E \sum_{i=1}^{E} f_i\, P_i ,
```

where $f_i$ is the fraction of tokens routed to expert $i$, and $P_i$ is the mean router probability for expert $i$. It's minimized (at the value $\alpha$) when both are uniform, at $1/E$.
Newer models also use a **router z-loss** (penalizing large router logits, for stability), or bias-based balancing without an auxiliary loss.

*Debug (the lesson's question 6): spiky loss and a few experts hogging the tokens.*

- Log the per-expert token fractions.
- Check that the auxiliary loss is actually added, with a sensible $\alpha$ (around $10^{-2}$).
- Add the router z-loss, and keep the router in FP32.
- Check the capacity factor and dropped-token rate.
- Lower the learning rate, or add warmup.

---

## 5. State-space models (S4, Mamba)

### 5.1 A linear recurrence

A state-space model maps an input sequence $x_t$ to outputs $y_t$ through a hidden state $h_t \in \mathbb{R}^N$:

```math
h_t = \bar A\, h_{t-1} + \bar B\, x_t,\qquad y_t = C\, h_t .
```

(This is the discretization of the continuous system $h'(t) = Ah + Bx$ with step size $\Delta$, for example $\bar A = e^{\Delta A}$.)

At **inference**, it's an RNN: constant memory and $O(1)$ work per token, with no growing KV cache. For **training**, when $\bar A, \bar B, C$ are fixed (time-invariant, as in S4), unrolling gives

```math
y_t = \sum_{j=0}^{t} C\bar A^{j}\bar B\, x_{t-j} = (K * x)_t,\qquad K = (C\bar B,\ C\bar A\bar B,\ C\bar A^2\bar B,\dots),
```

a **convolution** with a long kernel $K$, computable in parallel with FFTs. S4's contribution was a structured $A$ that makes this stable and remembers far back.

### 5.2 Mamba: making the SSM selective

A fixed (time-invariant) SSM treats every token the same way. It can't decide "remember this name, skip these filler words". **Mamba makes $\bar B$, $C$, and the step $\Delta$ functions of the current input $x_t$.**
A large $\Delta$ means "reset and focus on this token", and a small $\Delta$ means "ignore it and keep the state". That content-based selection is what language needs (copying, recall, ignoring noise).

The price: with input-dependent parameters, the convolution trick no longer applies. Mamba computes the recurrence with a **hardware-aware parallel scan** (an associative-scan algorithm, run in fast on-chip memory).

**When SSMs win:** very long sequences (genomics, audio, long documents), where attention's $O(T^2)$ cost and the growing KV cache hurt. **When attention wins:** exact recall and copying of arbitrary earlier tokens.
That's why many 2025–26 models are **hybrids**: mostly SSM or linear-attention layers, plus a few full attention layers.

```python
import torch, math
torch.manual_seed(0)

# --- RoPE: the score depends only on the relative offset --------------------------------------
def rope(x, pos, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2) / d)                 # one frequency per pair
    ang = pos * theta
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * torch.cos(ang) - x2 * torch.sin(ang)
    out[..., 1::2] = x1 * torch.sin(ang) + x2 * torch.cos(ang)
    return out
q, k = torch.randn(64), torch.randn(64)
s1 = rope(q, 5) @ rope(k, 12)          # offset 7
s2 = rope(q, 105) @ rope(k, 112)       # same offset 7, far away
s3 = rope(q, 5) @ rope(k, 13)          # offset 8
print(f"score(5,12)={s1:.4f}  score(105,112)={s2:.4f}  score(5,13)={s3:.4f}")

# --- KV cache sizes ---------------------------------------------------------------------------------
kv = lambda layers, T, kv_heads, d_head, bytes_=2: 2 * layers * T * kv_heads * d_head * bytes_ / 1e9
print(f"KV cache, 7B-class, 4k tokens: MHA(32 kv heads) {kv(32, 4096, 32, 128):.2f} GB | GQA(8) {kv(32, 4096, 8, 128):.2f} GB | MQA(1) {kv(32, 4096, 1, 128):.3f} GB")

# --- SwiGLU with hidden 8/3 d has the same parameters as a 4d GELU MLP ------------------------------------
d = 768
print("4d MLP params:", 2 * d * 4 * d, "  SwiGLU (8/3 d) params:", 3 * d * int(8 * d / 3))
```

```python
# --- A tiny top-2 MoE router: balance with vs without the auxiliary loss ------------------------------
E, d, T = 8, 16, 4096
def train_router(aux_coef, steps=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    Wr = (torch.randn(d, E, generator=g) * 0.01).requires_grad_()
    expert_bias = torch.linspace(0, 1, E)                          # experts differ slightly in "quality"
    opt = torch.optim.Adam([Wr], lr=0.05)
    for _ in range(steps):
        x = torch.randn(T, d, generator=g)
        probs = torch.softmax(x @ Wr, -1)
        top = probs.topk(2, -1).indices
        # the "task" reward pushes the router toward better experts (the rich-get-richer pressure)
        task_loss = -(probs * expert_bias).sum(-1).mean()
        f = torch.zeros(E).scatter_add_(0, top.flatten(), torch.ones(top.numel())) / top.numel()
        P = probs.mean(0)
        aux = E * (f * P).sum()
        loss = task_loss + aux_coef * aux
        opt.zero_grad(); loss.backward(); opt.step()
    return f
for c in [0.0, 1.0]:
    f = train_router(c)
    print(f"aux coef {c}: token fraction per expert = {[round(v, 3) for v in f.tolist()]}")
```

```python
# --- An LTI state-space model: recurrence == convolution; Mamba-style selective step ----------------------
N, Tn = 4, 50
A = torch.diag(torch.tensor([0.9, 0.7, 0.5, 0.95])); B = torch.randn(N, 1); C = torch.randn(1, N)
x = torch.randn(Tn)
h, y_rec = torch.zeros(N, 1), []
for t in range(Tn):
    h = A @ h + B * x[t]; y_rec.append((C @ h).item())
K = torch.tensor([(C @ torch.matrix_power(A, j) @ B).item() for j in range(Tn)])
y_conv = [sum(K[j] * x[t - j] for j in range(t + 1)).item() for t in range(Tn)]
print("recurrence == convolution:", torch.allclose(torch.tensor(y_rec), torch.tensor(y_conv), atol=1e-4))

# Selectivity: a large step Delta (input-dependent) resets the state toward the current token
def selective(xs, deltas, a=-1.0):
    h, out = 0.0, []
    for x_t, dt in zip(xs, deltas):
        Abar = math.exp(dt * a); h = Abar * h + (1 - Abar) * x_t; out.append(round(h, 3))
    return out
xs = [5.0, 0.1, 0.1, 0.1, 0.1]
print("small Delta on filler tokens keeps the 5.0:", selective(xs, [5.0, 0.01, 0.01, 0.01, 0.01]))
print("fixed Delta forgets it                    :", selective(xs, [5.0, 1.0, 1.0, 1.0, 1.0]))
```

---

## Pitfalls & misconceptions

- **"An MoE with 47B parameters runs like a 13B model."** That's true for compute per token, not memory: you still need to load 47B.
- **Comparing models by parameter count across dense and MoE architectures.**
- **Extending RoPE context with no fine-tuning or scaling.** Quality collapses beyond the trained length.
- **"SSMs replace attention."** They trade exact recall for linear cost. That's why hybrids are common.
- **Forgetting the KV-cache cost** when choosing an attention variant for serving.

## Cheat sheet

| Item | Formula / number |
|---|---|
| RoPE | $\langle R_{m\theta}q, R_{n\theta}k\rangle = \langle q, R_{(n-m)\theta}k\rangle$, $\theta_i = b^{-2i/d}$ |
| KV cache | $2\cdot L\cdot T\cdot n_\text{kv}\cdot d_\text{head}\cdot b$ ($b$ = bytes per value) |
| GQA saving | $n_\text{heads}/n_\text{kv}$ (32/8 = 4×) |
| SwiGLU | $W_2(\operatorname{SiLU}(W_1x)\odot W_3x)$, hidden $\approx\frac83d$ |
| MoE output | $\sum_{\text{top-}k} \tilde g_i E_i(x)$ |
| MoE balance loss | $\alpha E\sum_i f_iP_i$ |
| SSM | $h_t = \bar Ah_{t-1} + \bar Bx_t$, $y_t = Ch_t$; time-invariant ⇒ a convolution |
| Mamba | $\bar B, C, \Delta$ depend on $x_t$ (selective) → parallel scan |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why does RoPE encode relative position though applied per token?</summary>

Rotating $q$ by $m\theta$ and $k$ by $n\theta$ makes their dot product $q^\top R_{(n-m)\theta}k$, because rotations compose and transpose to their inverses. The absolute angles cancel, leaving the offset. The demo shows equal scores at equal offsets.
</details>

<details>
<summary>2. How much does GQA with 8 KV heads (of 32) shrink the KV cache?</summary>

4×: the cache scales with the number of KV heads (32 → 8). In the example, 2.1 GB becomes 0.54 GB per 4k-token sequence.
</details>

<details>
<summary>3. 47B total, 13B active: what does each determine?</summary>

Total sets the memory (all the experts must be resident). Active sets the FLOPs per token, and so the compute cost and latency. Quality usually lands between the dense models of each size.
</details>

<details>
<summary>4. Why do MoE models need a load-balancing loss?</summary>

Routing is self-reinforcing: experts that get more tokens improve and attract more. Without balancing, the router collapses onto a few experts, the rest are wasted, and tokens are dropped at capacity limits. The auxiliary loss $E\sum f_iP_i$ pushes toward uniform use (the demo compares token fractions with and without it).
</details>

<details>
<summary>5. What makes Mamba's selective SSM different from S4, and why does it matter?</summary>

S4's parameters are fixed (time-invariant), so it's a convolution, but it can't adapt to content. Mamba makes $\bar B$, $C$ and $\Delta$ depend on the input, so it can choose to store or ignore tokens: content-based selection, needed for language. It's computed with a parallel scan instead of a convolution.
</details>

<details>
<summary>6. MoE: spiky loss, a few experts get all the tokens.</summary>

Check that the auxiliary balance loss is applied with a sensible coefficient, log the per-expert utilization, add the router z-loss and keep the router in FP32, review the capacity factor and dropped tokens, and lower the learning rate or add warmup.
</details>

## Where this leads

Next: [DL-09 notes](09-graph-neural-networks.md). Transformers treat a sequence as a fully connected set of tokens. Graph neural networks generalize that: tokens become nodes, and "who attends to whom" becomes the graph's edges.
After DL-09, Path 2 is complete. Head to [Path 3 (GEN-01 notes)](../llms-genai/01-how-llms-are-built.md) or [Path 5 (CV-01 notes)](../vision/01-object-detection-segmentation.md).
