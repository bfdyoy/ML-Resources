# GEN-01 notes: How LLMs Are Built

[← Lesson GEN-01](../../lessons/llms-genai/01-how-llms-are-built.md) · [All notes](../README.md) · [← DL-09 notes](../deep-learning/09-graph-neural-networks.md) · Next: [GEN-02 notes →](02-adapting-llms-finetuning-rag.md)

> **Reading time** ≈ 60 min. **You need:** [DL-06 notes](../deep-learning/06-transformers.md) (the transformer), [DL-05 notes](../deep-learning/05-embeddings-sequences-attention.md) §2 (language modeling, perplexity, temperature), and [DL-07 notes](../deep-learning/07-performance-gpus-mixed-precision.md) (compute and memory arithmetic).

---

## Where we are

You built a small GPT in DL-06. A frontier LLM is *the same architecture* (plus the DL-08 upgrades), taken through four stages:

1. **Tokenization:** text → integers.
2. **Pretraining:** next-token prediction on trillions of tokens.
3. **Post-training:** turning a text-completer into a helpful assistant (and, more recently, a reasoner).
4. **Inference:** sampling text out of it, fast.

This note explains the math and the reasoning behind each stage. The later lessons go deeper: post-training in GEN-07, inference in GEN-08.

---

## 1. Tokenization: byte-pair encoding (BPE)

### 1.1 Why not characters or words?

- **Characters:** sequences get very long, which is expensive given attention's $O(T^2)$ cost, and each step carries little meaning.
- **Words:** the vocabulary explodes, and any unseen word ("ChatGPT-ish") breaks the model.
- **Subwords** split the difference: frequent words become one token, and rare words are built from pieces.

### 1.2 The BPE algorithm

1. Start with a vocabulary of single bytes (or characters).
2. Count every adjacent pair of symbols in the training corpus.
3. **Merge the most frequent pair** into a new symbol, and add it to the vocabulary.
4. Repeat until the vocabulary reaches its target size (e.g. 50k–200k tokens).

To **encode** new text, apply the learned merges in the order they were learned. The code below trains BPE from scratch on a tiny corpus and shows the merges it picks.

### 1.3 The failure modes it causes

- **Spelling and counting:** "strawberry" might be tokens like `str` + `awberry`. The model never sees individual letters, so "how many r's?" asks about information *inside* a token, which it has to have memorized indirectly.
- **Arithmetic:** numbers are chunked inconsistently ("1234" might be `123` + `4`, and "1235" might be `12` + `35`), so digit-aligned algorithms are hard to learn. Many tokenizers now split digits individually.
- **Non-English text:** merges are learned mostly from English, so other languages are split into many more, shorter tokens (a higher **fertility**, meaning tokens per word). The consequences:
  - the same text costs 2–3× more tokens (more money, and a smaller effective context window);
  - each token carries less meaning;
  - the model saw fewer of them during training, so quality drops.

  That's the answer to the lesson's Romanian debug question.
- **Leading spaces:** " hello" and "hello" are different tokens, so stray whitespace in prompts changes behaviour.

---

## 2. Pretraining

### 2.1 Objective and data

The objective is exactly DL-05's: minimize the average next-token cross-entropy over a huge corpus. The craft is in the **data**:

1. web crawl (Common Crawl);
2. text extraction;
3. language identification;
4. quality filtering (heuristics plus classifiers trained to spot "educational" text);
5. **deduplication** (exact and near-duplicate removal: duplicates waste compute and cause memorization);
6. PII and toxicity filtering;
7. a **mixture** with code, math, books, and papers.

The FineWeb write-up (lesson Go deeper) documents every step and measures each step's effect.

### 2.2 How much compute? $C \approx 6ND$

For a model with $N$ parameters trained on $D$ tokens:

- the **forward pass** costs about 2 FLOPs per parameter per token (one multiply and one add in each matmul);
- the **backward pass** costs about twice that (DL-01 §3.1: one matmul for the weight gradient, one to pass the error down).

So:

```math
C \approx 6ND \ \text{FLOPs}.
```

*Example:* a 7B model on 2T tokens needs $6 \times 7\times10^9 \times 2\times10^{12} \approx 8.4\times10^{22}$ FLOPs. A GPU sustaining $4\times10^{14}$ FLOP/s (about 40% of a $10^{15}$ peak) would take $2.1\times10^8$ seconds, which is about **6.7 GPU-years**, or about 2.5 days on 1,000 GPUs.

### 2.3 Scaling laws

The pretraining loss falls as a smooth **power law** in model size and data. The Chinchilla paper fit:

```math
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} .
```

- $E$ is the irreducible entropy of text.
- The other two terms are the penalties for a model that is too small, and for too little data.
- For a fixed budget $C = 6ND$, there's an optimal split. Chinchilla's headline rule of thumb is **about 20 training tokens per parameter**, with $N$ and $D$ growing roughly equally as the compute grows.

Two caveats:

- Plugging the paper's published parametric fit into an optimizer gives *larger* ratios than 20 (the code below shows about 50–80 at these budgets). The fit and the rule of thumb were derived differently, and a later replication found inconsistencies. Treat "20×" as an order of magnitude, not a law.
- "Compute-optimal" ignores **inference cost**. A model that will serve billions of requests should be smaller and trained far longer: modern open models use hundreds or thousands of tokens per parameter.

### 2.4 What the base model is

A **base model** continues text. Ask it a question and it may continue with more questions, because that's what text on the web often does. It has absorbed knowledge and skills, but it hasn't been trained to *follow instructions* or to behave as an assistant.

---

## 3. Post-training

### 3.1 Supervised fine-tuning (SFT)

Train on (prompt, ideal response) pairs, formatted with a **chat template** (special tokens marking the system, user, and assistant turns). The loss is computed **only on the assistant tokens**: you don't want to train the model to produce users' prompts.
SFT teaches format, tone, and how to follow instructions, using tens of thousands to millions of examples.

**What SFT can't fix by itself:** it only shows good answers, never *which of two answers is better*. And it imitates its demonstrations, so it can't exceed them.

### 3.2 Preference optimization (RLHF, DPO)

Collect pairs where humans (or AI judges) prefer response A over response B. Then either:

- train a **reward model** on those pairs and optimize the policy against it with RL, with a KL penalty that keeps it close to the SFT model (RLHF/PPO); or
- optimize the preference directly, with no explicit reward model (DPO).

This optimizes *relative quality* (helpfulness, harmlessness, style), which SFT can't express. The math is in [GEN-07 notes](07-post-training-alignment-reasoning.md).

### 3.3 RL with verifiable rewards: reasoning models

For math problems with checkable answers, code with unit tests, and puzzles with verifiers, the reward is **computed automatically and objectively**. You don't need human labels, and it's hard to game, because the answer is either right or wrong.
Sample many attempts, reward the correct ones, and update (GRPO). Over training, models learn to generate **long chains of thought** that check, backtrack, and try alternatives, because longer careful reasoning raises the reward.
That's what produced the "reasoning model" generation.

---

## 4. Inference: getting text out

### 4.1 Sampling knobs

The model outputs logits $z$ over the vocabulary. Then:

- **Temperature:** $p_i \propto e^{z_i/T}$. $T \to 0$ is greedy (argmax), and $T > 1$ flattens the distribution (more diverse, and more errors). Use $T \approx 0$ for extraction, classification, and code where there's one right answer and reproducibility matters. Use higher values for brainstorming.
- **Top-k:** sample only from the $k$ most likely tokens.
- **Top-p (nucleus):** sample from the smallest set of tokens whose probabilities sum to at least $p$. It adapts: a confident distribution gives a small set, and a flat one gives a big set.

### 4.2 KV cache and context limits

Generating token $t$ needs attention over every previous token. Recomputing their keys and values each step would make generation quadratic, so they're **cached**. Each new step computes only the new token's $q, k, v$.
The cache grows linearly with context (DL-08 §3.1). The **context limit** comes from the trained positional range (RoPE, DL-08 §1.2), plus attention and KV memory costs, plus how well the model actually *uses* far-away tokens. Those are three different limits.

```python
from collections import Counter
import numpy as np

# --- Byte-pair encoding from scratch ----------------------------------------------------
def train_bpe(text, n_merges):
    words = Counter(text.split())
    vocab = {tuple(w) + ("</w>",): c for w, c in words.items()}     # words as symbol tuples
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for sym, c in vocab.items():
            for a, b in zip(sym, sym[1:]): pairs[(a, b)] += c
        if not pairs: break
        best = max(pairs, key=pairs.get); merges.append(best)
        new_vocab = {}
        for sym, c in vocab.items():
            out, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best:
                    out.append(sym[i] + sym[i + 1]); i += 2
                else:
                    out.append(sym[i]); i += 1
            new_vocab[tuple(out)] = c
        vocab = new_vocab
    return merges

def encode(word, merges):
    sym = list(word) + ["</w>"]
    for a, b in merges:
        i, out = 0, []
        while i < len(sym):
            if i < len(sym) - 1 and (sym[i], sym[i + 1]) == (a, b):
                out.append(a + b); i += 2
            else:
                out.append(sym[i]); i += 1
        sym = out
    return sym

english = ("the model reads the text and the model predicts the next token in the text " * 30 +
           "training a language model on lots of text makes the model better at predicting text " * 30)
merges = train_bpe(english, 60)
print("first merges:", merges[:8])
for w in ["the", "model", "predicting", "strawberry"]:
    print(f"{w!r:13} -> {encode(w, merges)}")

# Fertility: tokens per word for English vs Romanian under an English-trained BPE
en = "the model predicts the next token in the text".split()
ro = "modelul prezice urmatorul cuvant din textul dat".split()
fert = lambda ws: sum(len(encode(w, merges)) for w in ws) / len(ws)
print(f"tokens per word: English {fert(en):.2f}, Romanian {fert(ro):.2f}")
```

```python
# --- Compute budget, and scaling-law trade-offs ---------------------------------------------------
N, D = 7e9, 2e12
C = 6 * N * D
print(f"C = 6ND = {C:.2e} FLOPs; at 4e14 FLOP/s that's {C / 4e14 / 3.15e7:.1f} GPU-years")

E, A, B, a, b = 1.69, 406.4, 410.7, 0.34, 0.28        # Chinchilla's published parametric fit
for budget in [1e21, 1e23]:
    Ns = np.logspace(8, 12, 20000); Ds = budget / (6 * Ns)
    L = E + A / Ns**a + B / Ds**b; i = L.argmin()
    print(f"budget {budget:.0e}: loss-optimal N={Ns[i]:.2e}, D={Ds[i]:.2e} ({Ds[i]/Ns[i]:.0f} tokens/param), loss {L[i]:.3f}")

# --- Temperature and top-p on a next-token distribution ---------------------------------------------
logits = np.array([4.0, 3.5, 2.0, 1.0, 0.5, 0.0, -1.0])
def probs(z, T): p = np.exp((z - z.max()) / T); return p / p.sum()
for T in [0.2, 1.0, 2.0]:
    print(f"T={T}: {np.round(probs(logits, T), 3)}")
p = probs(logits, 1.0); order = np.argsort(-p); cum = np.cumsum(p[order])
nucleus = order[: np.searchsorted(cum, 0.9) + 1]
print("top-p=0.9 keeps token ids", sorted(nucleus.tolist()), "covering", round(p[nucleus].sum(), 3))
```

---

## Pitfalls & misconceptions

- **"The model sees letters."** It sees token IDs. Reason about tokens when a model fails at character-level tasks.
- **Comparing costs or context lengths across languages without counting tokens.**
- **"Bigger is always better."** At a fixed compute budget there's an optimal size, and inference cost pushes toward smaller models trained longer.
- **Temperature 0 = deterministic.** It's greedy, but GPU nondeterminism and batching can still change outputs slightly.
- **Confusing the base and instruct versions** of a model when fine-tuning or evaluating.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| BPE | repeatedly merge the most frequent adjacent pair |
| Training compute | $C \approx 6ND$ |
| Chinchilla loss | $E + A/N^\alpha + B/D^\beta$; rule of thumb about 20 tokens/param (an order of magnitude) |
| SFT loss | cross-entropy on the assistant tokens only |
| Temperature | $p \propto e^{z/T}$ |
| Top-p | smallest set with cumulative probability $\ge p$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why do LLMs struggle to count letters in "strawberry"?</summary>

The model sees a few multi-character tokens, not letters. Counting r's needs information hidden inside the tokens, which the model only knows indirectly (if at all). Spelling it out letter by letter first (one token per letter) helps.
</details>

<details>
<summary>2. Base vs instruction-tuned model.</summary>

Base: next-token prediction on raw web-scale text, so it continues documents. Instruct: further trained with SFT on chat-formatted (instruction, response) data, and usually preference-optimized, so it follows instructions and behaves like an assistant.
</details>

<details>
<summary>3. What does preference optimization optimize that SFT doesn't?</summary>

Relative quality: which of two responses is better according to human (or AI) judgment. SFT only imitates single demonstrations, so it can't express "this answer is better than that one", and it can't exceed its demonstrations.
</details>

<details>
<summary>4. What is a verifiable reward, and why did it enable reasoning models?</summary>

A reward computed by an automatic, reliable check (exact math answer, passing unit tests). It's cheap, scalable, and hard to game, so RL can run at scale. Models discover that longer, self-checking chains of thought earn more reward, which produces reasoning behaviour.
</details>

<details>
<summary>5. How does temperature change the distribution; when T = 0?</summary>

It divides the logits: low $T$ sharpens toward the argmax, and high $T$ flattens toward uniform. Use $T \approx 0$ for tasks with a single correct output (extraction, classification, code, evals) where you want reproducibility.
</details>

<details>
<summary>6. Poor Romanian and 3× more tokens: the connection.</summary>

An English-dominated BPE splits Romanian into many short tokens (high fertility). The cost and effective context length suffer, each token carries less meaning, and Romanian made up little of the pretraining data, so the model is both less efficient and less skilled there (see the fertility demo).
</details>

## Where this leads

Next: [GEN-02 notes](02-adapting-llms-finetuning-rag.md). You know how a model is built. Most practitioners don't pretrain: they **adapt** existing models. GEN-02 covers the three adaptation levers (prompting, fine-tuning with LoRA, and retrieval-augmented generation), with the math of LoRA.
