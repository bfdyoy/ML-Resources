# DL-06 notes: Transformers

[← Lesson DL-06](../../lessons/deep-learning/06-transformers.md) · [All notes](../README.md) · [← DL-05 notes](05-embeddings-sequences-attention.md) · Next: [DL-07 notes →](07-performance-gpus-mixed-precision.md)

> **Reading time** ≈ 75 min. This is the most important note in Path 2, so read it twice. **You need:** [DL-05 notes](05-embeddings-sequences-attention.md) §2 and §4 (language modeling, attention), [DL-03 notes](03-training-deep-networks.md) §1.2 and §2 (residuals, LayerNorm), and [MATH-01 notes](../math/01-linear-algebra.md) §A2 and §B1 (matrix products, dot products).

---

## Where we are

DL-05 ended with attention as a helper bolted onto an RNN. The transformer removes the RNN entirely:

- **every token looks at every other token, in parallel**, through attention;
- small MLPs process each token;
- residual connections and LayerNorm keep it trainable at depth.

This one architecture now powers language, vision (CV-02), audio (EL-09), and more. We'll build it from the inside out, deriving each design choice.

---

## 1. Self-attention as soft dictionary lookup

Every token at position $i$ produces three vectors from its representation $x_i$, using learned matrices:

- a **query** $q_i = W_Q x_i$: "what am I looking for?"
- a **key** $k_j = W_K x_j$: "what do I contain?"
- a **value** $v_j = W_V x_j$: "what do I hand over if selected?"

Token $i$'s output is a weighted average of all the values, with weights given by how well its query matches each key. Stack the tokens as the rows of $X \in \mathbb{R}^{T\times d}$:

```math
Q = XW_Q,\ K = XW_K,\ V = XW_V,\qquad \operatorname{Attention}(Q,K,V) = \operatorname{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}} + M\Big)V .
```

- $QK^\top$ is the $T\times T$ matrix of **all pairwise dot products** (MATH-01 §B1: alignment).
- The softmax is taken **row by row**, so each token's weights sum to 1.
- $M$ is the mask (§1.2).

Compared with a Python dict, this is a lookup where the key match is *soft* and *learned*, and the result blends the values.

### 1.1 Why divide by $\sqrt{d_k}$

Suppose the entries of $q$ and $k$ are independent with mean 0 and variance 1. Then $q^\top k = \sum_{m=1}^{d_k} q_m k_m$ is a sum of $d_k$ terms, each with variance 1, so

```math
\operatorname{Var}(q^\top k) = d_k \quad\Longrightarrow\quad \operatorname{std}(q^\top k) = \sqrt{d_k}.
```

With $d_k = 64$, the raw scores have a standard deviation of about 8. A softmax over scores that far apart is **nearly one-hot** (saturated), and its gradient is nearly zero, so learning stalls.
Dividing by $\sqrt{d_k}$ brings the scores back to unit variance, keeping the softmax in its responsive range. The demo below measures the saturation directly.

### 1.2 The causal mask, and parallel training

A language model must predict token $t$ from tokens $\lt t$ only. Set $M_{ij} = -\infty$ for $j > i$ (and 0 otherwise). After the softmax, those weights are exactly 0, so **no token can see the future**.

The payoff is huge. In **one forward pass** over a sequence of length $T$, the model makes $T$ next-token predictions at once, each correctly conditioned only on its prefix. Training on all positions in parallel is the main reason transformers train so much faster than RNNs,
which must step through the tokens one by one.

### 1.3 Multi-head attention

One attention pattern per layer is limiting: a token may need to look at its syntactic head, at the previous token, and at a coreferent noun, all at once. **Multi-head attention** runs $h$ attention operations in parallel,
each on its own $d_k = d_\text{model}/h$ dimensional projections. It concatenates their outputs and mixes them with $W_O$:

```math
\operatorname{MHA}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\,W_O,\qquad \text{head}_r = \operatorname{Attention}(XW_Q^{(r)}, XW_K^{(r)}, XW_V^{(r)}).
```

The cost is the same as one big head, but you get $h$ different relational patterns.

### 1.4 Permutation equivariance, and why we need positions

Permute the input tokens with a permutation matrix $P$. Then $Q, K, V$ are permuted the same way, the score matrix becomes $P(QK^\top)P^\top$, the row-wise softmax commutes with this reordering, and so

```math
\operatorname{Attention}(PX) = P\operatorname{Attention}(X).
```

**Attention has no notion of order.** "dog bites man" and "man bites dog" contain the same multiset of tokens. So order must be injected:

- **learned absolute** position embeddings (GPT-2);
- **sinusoidal** ones, $PE(p, 2i) = \sin(p/10000^{2i/d})$ and $PE(p, 2i+1) = \cos(p/10000^{2i/d})$, which are smooth and multi-frequency, like the hands of a clock;
- **rotary (RoPE)**, which rotates $q$ and $k$ by position-dependent angles so that $q^\top k$ depends only on the **relative** offset. Used by almost every modern LLM; derived in [DL-08 notes](08-modern-architectures-moe-ssm.md).

(The causal mask breaks the symmetry partly on its own, but explicit positions work far better.)

---

## 2. The transformer block

A modern (pre-LN) decoder block:

```math
\begin{aligned}
x &\leftarrow x + \operatorname{MHA}\big(\operatorname{LN}(x)\big) &&\text{(tokens exchange information)}\\
x &\leftarrow x + \operatorname{MLP}\big(\operatorname{LN}(x)\big) &&\text{(each token is processed on its own)}
\end{aligned}
```

The MLP is usually $d \to 4d \to d$ with a GELU (or SwiGLU) in between.

**The residual stream view.** Think of $x$ as a **shared communication channel** that runs through the whole network:

- each attention or MLP sub-layer *reads* from it (after a LayerNorm) and *writes* by adding its output;
- **attention moves information between positions**;
- **MLPs transform information within a position** (they store a lot of the "knowledge").

Without the residual connections, every layer would have to re-encode everything, and the gradient would pass through a deep product of Jacobians (DL-03 §1). With them, there's an identity path from the loss straight back to the embeddings.
Pre-LN (normalize *before* each sub-layer, as above) keeps that path clean, and it's more stable than the original post-LN. Mechanistic interpretability (EL-07) is built on this view.

**The whole model:** token embedding + positions → $N$ blocks → final LN → linear "unembedding" to vocabulary logits (often **tied** to the embedding matrix) → softmax.

---

## 3. Costs: why long context is hard

For sequence length $T$ and width $d$:

| Part | Compute | Memory (activations) |
|---|---|---|
| Attention scores $QK^\top$ and $AV$ | $O(T^2 d)$ | $O(T^2)$ per head: the $T\times T$ matrix |
| Projections + MLP | $O(T d^2)$ | $O(Td)$ |

For short sequences, the MLPs dominate. As $T$ grows, the **quadratic** attention term takes over: doubling the context quadruples the attention cost. That's why long context is expensive, and why FlashAttention (which never materializes the $T\times T$ matrix in slow memory), sparse or sliding-window attention (DL-08), and state-space models exist.
At inference, cached keys and values (the **KV cache**) avoid recomputation, but their memory grows linearly with $T$ (GEN-08).

### 3.1 Counting parameters

Per block, ignoring the small biases and LayerNorms:

- attention has $W_Q, W_K, W_V, W_O$, which is $4d^2$;
- the MLP has $d\cdot4d + 4d\cdot d = 8d^2$.

So it's about **$12d^2$ per block**. For GPT-2 small ($d = 768$, 12 blocks, vocabulary 50,257, context 1,024), that's $12\cdot12\cdot768^2 \approx 85\text{M}$ in the blocks, plus $50{,}257\cdot768 \approx 38.6\text{M}$ token embeddings (tied with the output layer), plus 0.8M position embeddings. Total: **about 124M**, matching the published size. The code checks the formula against a real module.

---

## 4. Encoder, decoder, encoder-decoder

| Type | Attention | Pretraining objective | Good at | Examples |
|---|---|---|---|---|
| **Encoder-only** | Bidirectional (no mask) | Masked LM: predict hidden tokens from both sides | Understanding: classification, retrieval embeddings, NER | BERT, RoBERTa |
| **Decoder-only** | Causal | Next-token prediction | Generation, and (at scale) almost everything | GPT, Llama |
| **Encoder-decoder** | Encoder bidirectional; decoder causal + **cross-attention** to the encoder | Span corruption / seq2seq | Input→output transformations: translation, summarization | T5, BART, Whisper |

**Cross-attention** is attention whose queries come from the decoder and whose keys and values come from the encoder's outputs. It's exactly the DL-05 §4.2 attention, now inside a transformer.

---

## 5. Overfitting a small GPT (the lesson's debug question)

Training loss drops fast, samples are gibberish, and validation loss rises: the model is **memorizing** a small corpus. It has too many parameters for the data (Tiny Shakespeare is about 1M characters), or it's been trained too long.
Fixes: early stopping on validation loss, dropout, a smaller model, more data, weight decay. Also check that the validation split isn't accidentally the same text.

```python
import torch, torch.nn as nn, torch.nn.functional as F, math
torch.manual_seed(0)

# --- Attention from scratch with a causal mask == PyTorch's fused kernel -------------
T, d_k = 6, 16
Q, K, V = torch.randn(T, d_k), torch.randn(T, d_k), torch.randn(T, d_k)
scores = Q @ K.T / math.sqrt(d_k)
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)      # True above the diagonal = future
scores = scores.masked_fill(mask, float("-inf"))
A = torch.softmax(scores, dim=-1)
out = A @ V
ref = F.scaled_dot_product_attention(Q[None], K[None], V[None], is_causal=True)[0]
print("matches F.scaled_dot_product_attention:", torch.allclose(out, ref, atol=1e-6))
print("row 2 of A (only positions 0..2 nonzero):", [round(v, 3) for v in A[2].tolist()])

# --- Why sqrt(d_k): score spread and softmax saturation -----------------------------------
for dk in [4, 64, 512]:
    q, k = torch.randn(10000, dk), torch.randn(10000, dk)
    s = (q * k).sum(1)
    logits = torch.randn(1000, 8, dk) @ torch.randn(1000, dk, 1)       # 8 candidate keys per query
    p_raw = torch.softmax(logits.squeeze(-1), -1).max(-1).values.mean()
    p_scaled = torch.softmax(logits.squeeze(-1) / math.sqrt(dk), -1).max(-1).values.mean()
    print(f"d_k={dk:3d}: std(q.k)={s.std():6.2f} (sqrt d_k={math.sqrt(dk):5.2f}); mean max-softmax raw={p_raw:.3f}, scaled={p_scaled:.3f}")

# --- Permutation equivariance (no mask, no positions) --------------------------------------
X = torch.randn(T, 8); Wq, Wk, Wv = (torch.randn(8, 8) for _ in range(3))
attn = lambda X: torch.softmax((X @ Wq) @ (X @ Wk).T / math.sqrt(8), -1) @ (X @ Wv)
perm = torch.randperm(T)
print("Attention(PX) == P Attention(X):", torch.allclose(attn(X[perm]), attn(X)[perm], atol=1e-5))
```

```python
# --- A minimal pre-LN GPT; check the 12 d^2 parameter formula --------------------------------
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        T = x.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=causal, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, vocab, ctx, d, h, n_layers):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight                       # weight tying
    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks: x = b(x)
        return self.head(self.ln(x))

d, L = 128, 4
small = GPT(vocab=1000, ctx=256, d=d, h=4, n_layers=L)
block_params = sum(p.numel() for p in small.blocks.parameters())
print(f"block params (actual) {block_params:,} vs 12*d^2*L = {12 * d * d * L:,}  (the rest is biases and LayerNorms)")
print("logits shape:", tuple(small(torch.randint(0, 1000, (2, 10))).shape))
gpt2 = 12 * 12 * 768**2 + 50257 * 768 + 1024 * 768
print(f"GPT-2 small estimate: {gpt2/1e6:.1f}M parameters (published: 124M)")
```

---

## Pitfalls & misconceptions

- **A mask with the wrong orientation** (or none at all) during training. The model peeks at the future, the loss looks amazing, and generation is garbage.
- **Forgetting positional information.** The model becomes a bag-of-words.
- **Thinking attention weights are explanations.** They show where information *was routed*, not why a decision was made (EL-07).
- **Quadratic memory surprises:** going from 2k to 32k context multiplies the attention-matrix memory by 256.
- **Post-LN with a large learning rate and no warmup.** It's unstable. Pre-LN plus warmup is the safe default.

## Cheat sheet

| Item | Formula |
|---|---|
| Attention | $\operatorname{softmax}(QK^\top/\sqrt{d_k} + M)V$ |
| Score variance | $\operatorname{Var}(q^\top k) = d_k$, hence the $\sqrt{d_k}$ |
| Causal mask | $M_{ij} = -\infty$ for $j > i$ |
| Equivariance | $\operatorname{Attn}(PX) = P\operatorname{Attn}(X)$, so add positions |
| Pre-LN block | $x \mathrel{+}= \operatorname{MHA}(\operatorname{LN}x)$; $x \mathrel{+}= \operatorname{MLP}(\operatorname{LN}x)$ |
| Parameters per block | $\approx 12d^2$ |
| Attention cost | $O(T^2d)$ compute, $O(T^2)$ memory |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why divide by sqrt(d_k)?</summary>

Dot products of $d_k$-dimensional vectors with unit-variance entries have variance $d_k$. Unscaled, the softmax saturates (nearly one-hot) and its gradients vanish. Scaling restores unit variance (§1.1). The demo shows the max-softmax near 1 for raw scores at large $d_k$.
</details>

<details>
<summary>2. What does the causal mask do, and why does it enable parallel training?</summary>

It sets the weights to future positions to 0, so each position conditions only on its prefix. Then one forward pass yields correct next-token predictions for all $T$ positions at once (teacher forcing in parallel), instead of $T$ sequential steps.
</details>

<details>
<summary>3. Why does permutation equivariance force positional information?</summary>

Attention treats its input as a set: permuting the tokens just permutes the outputs, so word order is invisible. Positions (learned, sinusoidal, RoPE) are needed to tell "dog bites man" from "man bites dog".
</details>

<details>
<summary>4. Time and memory cost of attention in sequence length.</summary>

$O(T^2 d)$ time and $O(T^2)$ memory per head for the score matrix. Long contexts become expensive quadratically, which motivates FlashAttention, sparse or sliding windows, linear attention, and SSMs, and makes the KV cache the inference bottleneck.
</details>

<details>
<summary>5. The role of the residual stream.</summary>

It's a shared channel that every sub-layer reads from and adds to, giving an identity path for gradients and letting layers make incremental edits. Without it, deep transformers suffer vanishing gradients and every layer must re-encode everything.
</details>

<details>
<summary>6. Loss drops fast, samples gibberish, validation loss rising.</summary>

Overfitting: memorizing a small corpus. Use early stopping on validation loss, dropout, weight decay, a smaller model or more data, and check the train/validation split for leakage.
</details>

## Where this leads

Next: [DL-07 notes](07-performance-gpus-mixed-precision.md). You can now build a GPT. Next comes making it **fast**: where the time actually goes on a GPU, mixed precision, compilation, and memory accounting.
After that, [DL-08](08-modern-architectures-moe-ssm.md) upgrades the 2017 block to a 2026 one (RoPE, GQA, MoE). [GEN-01](../llms-genai/01-how-llms-are-built.md) scales it into an LLM.
