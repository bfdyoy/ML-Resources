# GEN-02 notes: Adapting LLMs: Prompting, Fine-tuning, LoRA & RAG

[← Lesson GEN-02](../../lessons/llms-genai/02-adapting-llms-finetuning-rag.md) · [All notes](../README.md) · [← GEN-01 notes](01-how-llms-are-built.md) · Next: [GEN-03 notes →](03-evaluating-llm-apps.md)

> **Reading time** ≈ 60 min. **You need:** [GEN-01 notes](01-how-llms-are-built.md), [DL-07 notes](../deep-learning/07-performance-gpus-mixed-precision.md) §3 (training memory), and [MATH-01 notes](../math/01-linear-algebra.md) §A4 and §C5 (rank, low-rank approximation).

---

## Where we are

Pretraining is for a handful of labs. Everyone else **adapts** a pretrained model. There are three levers, in increasing order of cost:

1. **Prompting:** change the input.
2. **Retrieval-augmented generation (RAG):** give the model the right documents at query time.
3. **Fine-tuning:** change the weights, usually with **LoRA**, which changes only a tiny low-rank slice of them.

Knowing what each lever *can* and *can't* change is the most important practical skill in this path.

---

## 1. Prompting and in-context learning

A pretrained model can pick up a task from **examples in the prompt** (few-shot) with no weight updates. Useful habits:

- clear instructions and an explicit output format;
- a few diverse examples that cover the edge cases;
- asking for reasoning before the answer when the task is multi-step;
- putting reference material *in* the prompt.

Prompting is fast to iterate on and costs nothing to "train". Its limits: context length, per-call cost (long prompts repeated on every call), and behaviours that are hard to specify in words.

---

## 2. Fine-tuning, and why full fine-tuning is expensive

Full fine-tuning updates every weight. With AdamW and mixed precision, that costs about **16 bytes per parameter** (DL-07 §3.1) plus activations. A 7B model needs more than 112 GB, which is more than one GPU.
It also produces a full copy of the weights for each task you fine-tune.

### 2.1 LoRA: learn a low-rank update

Freeze the pretrained weight $W_0 \in \mathbb{R}^{d\times k}$. Learn an update that is the product of two thin matrices:

```math
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A,\qquad B \in \mathbb{R}^{d\times r},\ A \in \mathbb{R}^{r\times k},\ r \ll \min(d, k).
```

- **Trainable parameters:** $r(d + k)$ instead of $dk$. For $d = k = 4096$ and $r = 16$, that's 131k instead of 16.8M per matrix, about **0.8%**.
- **Initialization:** $A$ is random and $B = 0$, so $\Delta W = 0$ at the start, and the fine-tuned model **begins exactly as the base model**. (The code checks this.)
- **The $\alpha/r$ scaling:** it keeps the update's magnitude roughly independent of $r$, so when you change the rank, you don't have to re-tune the learning rate. A common choice is $\alpha = 2r$ or $\alpha = r$.
- **Merging:** after training, compute $W_0 + \frac{\alpha}{r}BA$ once. Inference then has **zero extra latency**. Alternatively, keep several adapters (a few MB each) and swap them on one shared base model.

### 2.2 Where the memory saving comes from

| Item | Full fine-tuning | LoRA |
|---|---|---|
| Base weights | Trainable (2 bytes BF16 + 4 bytes FP32 master) | **Frozen**, 2 bytes (or 0.5 bytes with 4-bit **QLoRA**) |
| Gradients | For all parameters | Only for the adapters |
| Adam states ($m, v$) | 8 bytes × all parameters | 8 bytes × adapter parameters (under 1%) |
| Activations | Needed | Still needed (you still backprop *through* the frozen layers) |

So the savings come from **not storing gradients and optimizer state for the frozen weights**. A 7B model drops from more than 112 GB to about 14 GB (BF16 base) or about 4 GB (4-bit base, QLoRA), plus activations.

### 2.3 Why low rank is enough

Empirically, the weight changes that fine-tuning needs have low **intrinsic rank**. Their singular values decay quickly, so by the Eckart–Young theorem (MATH-01 §C5) a rank-$r$ update captures most of them.
Adapting to a task uses a small subspace of what the model can already do. The **rank** caps how much the update can express. The **target modules** decide where it acts: attention projections only, or all linear layers, which usually works better.

### 2.4 Catastrophic forgetting

Fine-tune hard on a narrow dataset (a high learning rate, many epochs) and the weights move far from the pretrained solution. Format and task skills improve, while general knowledge and other skills degrade. The model may also start parroting the training answers (overfitting).
Mitigations:

- a lower learning rate and fewer epochs, with early stopping on a held-out set;
- LoRA, whose small, constrained updates forget less;
- **mixing general-purpose data** into the fine-tuning set;
- evaluating on *both* the target task and a general benchmark.

---

## 3. Retrieval-augmented generation (RAG)

### 3.1 The pipeline

**Offline:** split documents into chunks → embed each chunk → store them in an index.
**Online:** embed the query → retrieve the top-$k$ chunks → (optionally) rerank → put them in the prompt → generate an answer that cites them.

In probabilistic terms, RAG approximates $p(y\mid x) \approx \sum_{z \in \text{top-}k} p(z\mid x)\, p(y\mid x, z)$: the answer is conditioned on retrieved evidence $z$.
**Retrieval decides what the model can know. Generation decides whether it uses that faithfully.**

### 3.2 Debugging "confident wrong answers": test each stage

| Stage | What fails | How to test it |
|---|---|---|
| Ingestion / chunking | Answers split across chunks, tables mangled, stale documents | Inspect the chunks for known answers |
| Embedding / retrieval | The right chunk isn't in the top-$k$ | **Recall@k on a labeled query set** (GEN-05) |
| Ranking / packing | The right chunk is retrieved but buried, or truncated out of the context | Check its rank and position; try a reranker |
| Generation | The right context is present but ignored or contradicted | **Oracle test:** give the gold chunk directly. If the model still fails, it's the generator or the prompt. |
| Question type | The answer needs aggregation across many docs, or isn't in the corpus at all | Categorize the failures; add "I don't know" handling |

The **oracle test** is the most useful single diagnostic: it cleanly separates retrieval failures from generation failures.

### 3.3 Chunk size trade-off

- **Small chunks:** precise embeddings (one idea per chunk), so a good **retrieval match**. But each chunk may lack the surrounding context needed to *answer*, and an answer may span several chunks.
- **Large chunks:** more context per hit. But the embedding averages many topics, so retrieval gets fuzzier, and fewer chunks fit in the prompt.

Common remedies: moderate chunks with overlap, structure-aware splitting (headings, paragraphs), and "small-to-big" retrieval (retrieve small chunks, then expand them to their parent section).

---

## 4. Choosing: prompting vs RAG vs fine-tuning

| Need | Best lever | Why |
|---|---|---|
| New or changing **knowledge** (docs, policies, prices) | **RAG** | Update the index, not the weights. Answers can cite sources. Fine-tuning is unreliable at injecting facts. |
| **Behaviour**: format, style, tone, a narrow skill, a smaller or faster model | **Fine-tuning** (LoRA) | Teaches *how* to respond. Can distil a big model's behaviour into a small one. |
| Quick prototype, small volume | **Prompting** | No training. Iterate in minutes. |
| Domain assistant with house style | **RAG + light fine-tuning** | Each fixes what the other can't. |

Always start with prompting plus a baseline eval (GEN-03). Move to RAG or fine-tuning only when the error analysis says so.

```python
import torch, torch.nn as nn
torch.manual_seed(0)

# --- A LoRA linear layer ---------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)          # freeze W0
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))          # B = 0 -> no change at start
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
    def merged(self):
        W = self.base.weight + self.scale * self.B @ self.A
        lin = nn.Linear(self.base.in_features, self.base.out_features); lin.weight.data, lin.bias.data = W.detach(), self.base.bias.detach()
        return lin

d = 256
base = nn.Linear(d, d)
lora = LoRALinear(base, r=4, alpha=8)
x = torch.randn(32, d)
print("identical to base at init:", torch.allclose(lora(x), base(x)))
n_train = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(f"trainable params: {n_train:,} vs full {d*d + d:,} ({100*n_train/(d*d):.1f}%)")

# Task: the "fine-tuned" target differs from the base by a rank-2 update
U, V = torch.randn(d, 2), torch.randn(2, d)
W_target = base.weight.detach() + 0.05 * U @ V
X = torch.randn(4096, d); Y = X @ W_target.T + base.bias.detach()
opt = torch.optim.Adam([lora.A, lora.B], lr=1e-2)
for step in range(500):
    loss = ((lora(X) - Y) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
print(f"LoRA (r=4) fits a rank-2 change: final MSE {loss.item():.2e}")
m = lora.merged()
print("merged layer == adapter forward:", torch.allclose(m(x), lora(x), atol=1e-5))

# --- Memory: full fine-tuning vs LoRA vs QLoRA for a 7B model (static state only) ------------
P, frac = 7e9, 0.005                          # adapters are ~0.5% of the parameters
full = 16 * P
lora_bf16 = 2 * P + 16 * frac * P
qlora = 0.5 * P + 16 * frac * P
print(f"7B: full FT ~{full/1e9:.0f} GB | LoRA (bf16 base) ~{lora_bf16/1e9:.1f} GB | QLoRA (4-bit base) ~{qlora/1e9:.1f} GB  (+ activations)")
```

```python
# --- A minimal RAG retrieval check: recall@k and the oracle idea ----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
docs = [
    "Refunds are available within 30 days of purchase with a receipt.",
    "Our office is open Monday to Friday from 9am to 5pm.",
    "Premium members get free shipping on all orders over 20 euros.",
    "Passwords must be at least 12 characters and include a number.",
    "Support tickets are answered within two business days.",
]
queries = [("how long do I have to return an item", 0), ("when can I visit the office", 1),
           ("is shipping free for premium", 2), ("password length requirement", 3),
           ("how fast does support reply", 4)]
vec = TfidfVectorizer().fit(docs)
D = vec.transform(docs)
def recall_at_k(k):
    hits = 0
    for q, gold in queries:
        scores = (vec.transform([q]) @ D.T).toarray()[0]
        hits += gold in np.argsort(-scores)[:k]
    return hits / len(queries)
print("recall@1 =", recall_at_k(1), " recall@3 =", recall_at_k(3))
print("-> retrieval misses are visible here, before any LLM is involved. GEN-05 covers the fixes: BM25, dense embeddings, hybrid search, reranking.")
```

---

## Pitfalls & misconceptions

- **Fine-tuning to teach facts.** It's unreliable and hard to update. Use RAG for knowledge.
- **Evaluating RAG only end-to-end.** Measure retrieval separately, with recall@k, and run the oracle test.
- **A LoRA learning rate copied from full fine-tuning.** LoRA usually wants a *higher* learning rate (around 1e-4 to 2e-4 is common).
- **Training on the prompt tokens** in instruction tuning. Mask them: compute the loss on the responses only.
- **No general-capability eval after fine-tuning**, so forgetting goes unnoticed.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| LoRA | $W = W_0 + \frac{\alpha}{r}BA$; $B = 0$ at init; $r(d+k)$ parameters |
| Memory | full ≈ 16 B/param; LoRA ≈ 2 B/param (base) + a tiny adapter state; QLoRA ≈ 0.5 B/param |
| RAG | $p(y\mid x) \approx \sum_{z\in\text{top-}k}p(z\mid x)p(y\mid x,z)$ |
| RAG debug | recall@k for retrieval; oracle context for generation |
| Choose | knowledge → RAG; behaviour/format → fine-tune; start with prompting + evals |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why does LoRA use much less GPU memory?</summary>

The base weights are frozen, so there are no gradients, no FP32 master copy, and no Adam states for them. Only the tiny adapters (under 1% of the parameters) carry the 16 bytes/param of training state. The base can even be stored in 4 bits (QLoRA). Activations are still needed.
</details>

<details>
<summary>2. Tiny dataset + high learning rate on an instruct model?</summary>

The model overfits and drifts far from the pretrained weights: it parrots the training answers, loses general skills (catastrophic forgetting), and can lose its instruction-following or safety behaviours. Use a lower learning rate, fewer epochs, LoRA, mixed-in general data, and early stopping.
</details>

<details>
<summary>3. RAG gives confident wrong answers: stages and tests.</summary>

Chunking (inspect the chunks), retrieval (recall@k on labeled queries), ranking and packing (the gold chunk's rank and whether it was truncated), and generation (the oracle test with the gold context). Also check whether the question is answerable from the corpus at all, and whether the index is stale (§3.2).
</details>

<details>
<summary>4. When is fine-tuning right over RAG, and vice versa?</summary>

Fine-tune for behaviour: format, style, tone, a narrow skill, latency or cost (a smaller model). Use RAG for knowledge that is large, private, or changing, or that needs citations. Often both.
</details>

<details>
<summary>5. What does chunk size trade off?</summary>

Small chunks give precise retrieval but little context per hit (answers can be split up). Large chunks carry more context but give blurrier embeddings and fewer chunks per prompt. Tune it on retrieval recall *and* answer quality.
</details>

<details>
<summary>6. Perfect format, forgotten general knowledge.</summary>

Catastrophic forgetting from aggressive, narrow fine-tuning. Lower the learning rate and the number of epochs, use LoRA with a modest rank, mix in general data, regularize toward the base model, and evaluate general benchmarks alongside the task.
</details>

## Where this leads

Next: [GEN-03 notes](03-evaluating-llm-apps.md). Every choice in this lesson (prompt vs RAG vs fine-tune, chunk size, rank) should be decided by **measurement**, and LLM outputs are notoriously hard to measure. GEN-03 is about building evals you can trust.
