# EL-07 notes: Mechanistic Interpretability

[← Lesson EL-07](../../lessons/electives/07-mechanistic-interpretability.md) · [All notes](../README.md) · [← EL-06 notes](06-anomaly-detection.md) · Next: [EL-08 notes →](08-gpu-programming.md)

> **Reading time** ≈ 65 min. **You need:** [DL-06 notes](../deep-learning/06-transformers.md) (especially §2, the residual stream) and [MATH-01 notes](../math/01-linear-algebra.md) (projections, low rank). Helpful: [CORE-08 notes](../core-ml/08-interpretability-and-responsible-ml.md) (the limits of attribution methods).

---

## Where we are

CORE-08's tools (SHAP, PDPs) treat a model as a black box: they ask how its *output* depends on its *input*. Mechanistic interpretability opens the box. It asks **which internal computations** implement a behaviour, the way you'd reverse-engineer a compiled program.
The transformer's architecture makes this tractable, because almost everything is linear except a few known places.

---

## 1. The residual stream as a communication channel

In a pre-LN transformer (DL-06 §2), the representation at each position is a running **sum**:

```math
x_L = x_0 + \sum_{\text{layers } l}\Big(\text{attn}_l(\cdot) + \text{mlp}_l(\cdot)\Big),\qquad \text{logits} = W_U\,\text{LN}(x_L).
```

- Every component **reads** from the stream through a linear map (after a LayerNorm), and **writes** to it by **adding** its output.
- Nothing is overwritten. Information persists unless a later component writes something that cancels it.
- Different components can use **different subspaces** of the stream, like separate channels on a shared bus. A later head can read what an earlier head wrote, *if* its read matrices are aligned with that subspace.

**Direct logit attribution:** if you treat the final LayerNorm's scale as a constant (freeze it for the input you're studying), the logits are **linear** in the stream. So the final logit splits **exactly** into a sum of contributions, one per component. It tells you which heads and MLPs directly pushed toward the answer.
(It misses *indirect* effects, where a component helps another component that then writes the answer. That's what patching (§3) captures.)

---

## 2. Attention heads as two circuits

For one head, the attention score between a destination token $i$ and a source token $j$ is $x_i^\top W_Q^\top W_K\,x_j$, and the value written back is $W_O W_V\,x_j$. Group the weights into two low-rank matrices of size $d_\text{model}\times d_\text{model}$, each of rank at most $d_\text{head}$:

- **QK circuit**, $W_Q^\top W_K$: **where to attend.** A bilinear form deciding which source tokens each destination token looks at.
- **OV circuit**, $W_O W_V$: **what to move.** Given that a source is attended to, what gets written into the destination's stream.

They're independent: a head can attend based on *position* (QK) while copying *token identity* (OV).

### 2.1 Induction heads

The sequence is `… A B … A` → predict `B`. Two heads in different layers do this together:

| Step | Head | QK (where to look) | OV (what to write) |
|---|---|---|---|
| 1 | **Previous-token head** (an earlier layer) | each position attends to the position just before it | writes "the previous token was A" into the stream at B's position |
| 2 | **Induction head** (a later layer) | at the second A, its query ("I am A") matches **keys built from step 1's output** ("my previous token was A"), so it attends to B's position | copies B's identity → boosts the logit for B |

That's **K-composition**: the second head's keys read what the first head wrote. Induction heads form abruptly during training, at the same moment as a jump in **in-context learning** ability. That's the evidence that they're a key mechanism for copying and continuing patterns seen earlier in the context.

---

## 3. Causal methods: activation patching

Correlation inside a network isn't causation either. **Activation patching** tests whether a component *matters*:

1. A **clean** run: "The Eiffel Tower is in" → Paris.
2. A **corrupted** run with a minimal change: "The Colosseum is in" → Rome.
3. Rerun the corrupted input, but **paste in one component's activation from the clean run**.
4. Measure how much of the clean behaviour is restored, for example with the logit difference $\text{logit}(\text{Paris}) - \text{logit}(\text{Rome})$:

```math
\text{restored} = \frac{m_\text{patched} - m_\text{corrupted}}{m_\text{clean} - m_\text{corrupted}} .
```

**Why patch clean into corrupted?** It asks whether the component is **sufficient** to carry the information that makes the difference: if this one activation alone restores "Paris", the relevant information flows through it. (Patching the other way, corrupted into clean, tests *necessity*.)
Use **minimal** corruptions, so you isolate one piece of information rather than breaking everything. Path patching and attribution patching (a fast gradient approximation) extend the idea.

---

## 4. Superposition: why neurons are hard to read

You'd like each neuron to mean one thing. In practice, many are **polysemantic**: they fire for unrelated concepts. The reason is **superposition**: models represent **more features than they have dimensions**, by giving each feature a *direction*. The directions are nearly orthogonal, but not exactly.

- In high dimensions you can fit **exponentially many** almost-orthogonal directions. Random unit vectors in $d$ dimensions have cosines of about $\pm 1/\sqrt d$ with each other.
- If features are **sparse** (rarely active at the same time), the interference between their overlapping directions rarely matters, and a ReLU can filter the small interference out.
- So the network packs features in, and any **neuron** (a single basis direction) overlaps with *many* features: polysemanticity. The *basis* isn't the meaningful unit; the feature directions are.

**Sparse autoencoders (SAEs)** recover those directions. Train an overcomplete dictionary on the model's activations $x$:

```math
f = \operatorname{ReLU}(W_e x + b_e),\qquad \hat x = W_d f + b_d,\qquad \mathcal L = \lVert x - \hat x\rVert^2 + \lambda\lVert f\rVert_1 .
```

The L1 penalty forces each activation to be explained by **few** active features (sparsity). The many dictionary columns of $W_d$ become candidate monosemantic features: "Golden Gate Bridge", "code comments", "deception".
They come with caveats: dead features, features split into finer and finer variants as the dictionary grows, and reconstruction error that hides what's missing.

---

## 5. Healthy skepticism about saliency

Gradient-based saliency maps (CORE-08, Integrated Gradients) can look convincing while explaining very little. The **sanity check** (Adebayo et al.): **randomize the model's weights** (or train on random labels) and recompute the map.
If the map looks the same, it was driven by the **input** (edges, textures, roughly an edge detector), not by what the model learned. It explains nothing about the model's decision. That's the lesson's debug question. Prefer methods that pass these checks, and causal interventions (patching, ablations) over pure attribution.

```python
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
torch.manual_seed(0)

# --- Direct logit attribution: with a frozen final-LN scale, the logit splits exactly into component terms --------
d, vocab = 16, 10
x0 = torch.randn(d)
components = {"embed": x0, "attn0": torch.randn(d), "mlp0": torch.randn(d), "attn1": torch.randn(d), "mlp1": torch.randn(d)}
resid = sum(components.values())
W_U = torch.randn(vocab, d)
scale = resid.std()                                           # freeze the LayerNorm scale for this input
center = lambda v: v - v.mean()
logit_total = (W_U @ (center(resid) / scale))[3]
contribs = {k: (W_U @ (center(v) / scale))[3].item() for k, v in components.items()}
print({k: round(v, 2) for k, v in contribs.items()})
print("sum of contributions == total logit:", np.isclose(sum(contribs.values()), logit_total.item()))

# --- Almost-orthogonal directions: how many features fit in d dimensions -------------------------------------------
for dim in [16, 128, 1024]:
    V = F.normalize(torch.randn(2000, dim), dim=1)
    cos = (V @ V.T).fill_diagonal_(0).abs()
    print(f"d={dim:5d}: 2000 random directions, mean |cos| {cos.mean():.3f} (of order 1/sqrt(d) = {1/np.sqrt(dim):.3f})")
```

```python
# --- Toy model of superposition, then a sparse autoencoder that recovers the features ---------------------------------------
n_feat, d_hidden = 20, 5
def sample(batch, p_active=0.05):
    mask = (torch.rand(batch, n_feat) < p_active).float()
    return mask * torch.rand(batch, n_feat)
W = nn.Parameter(torch.randn(d_hidden, n_feat) * 0.1); b = nn.Parameter(torch.zeros(n_feat))
opt = torch.optim.Adam([W, b], lr=1e-2)
for _ in range(4000):
    x = sample(1024)
    x_hat = F.relu(x @ W.T @ W + b)                          # compress 20 features into 5 dims, then decode
    loss = ((x_hat - x) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
norms = W.norm(dim=0)
print(f"features with a sizeable direction in 5 dims: {(norms > 0.5).sum().item()} of {n_feat}  (more than 5 means superposition)")

acts = sample(20000) @ W.detach().T                          # the 5-d hidden activations
enc, dec = nn.Linear(d_hidden, 40), nn.Linear(40, d_hidden)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=3e-3)
for _ in range(3000):
    f = F.relu(enc(acts)); rec = dec(f)
    loss = ((rec - acts) ** 2).mean() + 3e-3 * f.abs().sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
true_dirs = F.normalize(W.detach().T[norms > 0.5], dim=1)    # the represented feature directions
learned = F.normalize(dec.weight.detach().T, dim=1)
best = (true_dirs @ learned.T).max(1).values
print(f"SAE: each represented feature's best-matching dictionary direction has cosine {best.mean():.2f} on average")
```

---

## Pitfalls & misconceptions

- **Reading attention patterns as explanations.** Where a head looks ≠ what it does (that's the OV circuit) or whether it matters (that's patching).
- **Interpreting individual neurons** as if they were features.
- **Direct logit attribution as the whole story.** It misses indirect effects.
- **Corruptions that change too much** in patching experiments.
- **Saliency maps without sanity checks.**

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Residual stream | $x_L = x_0 + \sum(\text{attn} + \text{mlp})$; read linearly, write by adding |
| Direct logit attribution | logit $= \sum_c W_U\,\text{LN}(\text{out}_c)$ with the LN scale frozen |
| QK / OV | $W_Q^\top W_K$ (where) / $W_OW_V$ (what) |
| Induction | previous-token head + induction head (K-composition): `A B … A → B` |
| Patching | restored = $(m_p - m_c)/(m_\text{clean} - m_c)$ |
| SAE | $\lVert x - W_d\operatorname{ReLU}(W_ex)\rVert^2 + \lambda\lVert f\rVert_1$ |
| Saliency sanity check | randomize the weights: is the map unchanged? Then it's uninformative |

## Answer sketches for the lesson's self-check

<details>
<summary>1. "The residual stream is a communication channel."</summary>

Each layer's output is added to a shared running vector. Later layers read from it through linear projections (after LN), and write by addition. Different components use different subspaces to pass information forward, like channels on a shared bus.
</details>

<details>
<summary>2. What do the QK and OV circuits determine?</summary>

QK ($W_Q^\top W_K$): the attention pattern, meaning which source positions each destination attends to. OV ($W_OW_V$): what information from an attended source is written into the destination's residual stream.
</details>

<details>
<summary>3. How does an induction head use a previous-token head?</summary>

The previous-token head writes "my previous token was A" at B's position. At a later A, the induction head's query matches keys built from that information (K-composition), so it attends to B's position, and its OV circuit copies B, predicting B next.
</details>

<details>
<summary>4. What does activation patching measure; why patch clean into corrupted?</summary>

The causal contribution of a component: how much of the clean behaviour is restored when only that activation comes from the clean run. Clean-into-corrupted tests whether the component is sufficient to carry the decisive information. The reverse tests necessity.
</details>

<details>
<summary>5. Why does superposition make neurons hard to interpret?</summary>

Models encode more (sparse) features than dimensions as nearly orthogonal directions. Each neuron's axis overlaps with many feature directions, so it responds to several unrelated concepts. Features live in directions, not in single neurons. SAEs try to recover those directions.
</details>

<details>
<summary>6. A saliency map is unchanged after randomizing the weights.</summary>

The map depends on the input, not on what the model learned (it's effectively an edge detector), so it doesn't explain the model's decisions. Use methods that pass the sanity checks, and causal tests.
</details>

## Where this leads

Next: [EL-08 notes](08-gpu-programming.md). Interpretability opened up the model's computation. GPU programming opens up the hardware that runs it: kernels, memory, and why FlashAttention is fast.
