# Playbook 4: LLM & retrieval tricks

[← Playbook](README.md) · Previous: [Deep learning tricks](03-deep-learning-tricks.md) · Next: [Outside the box](05-outside-the-box.md)

> Tricks for building with LLMs that hold up under measurement. Each one says *how to check* that it helped on **your** eval set
> ([GEN-03](../lessons/llms-genai/03-evaluating-llm-apps.md)), because LLM tricks transfer across tasks less reliably than classical ones. **You need:** GEN-02, GEN-03, GEN-05.

---

## 1. Sample several answers and vote (self-consistency)

**When:** tasks with a checkable final answer (math, classification, extraction, multiple choice), where one sample is right more often than not.
**How:** sample k answers at temperature > 0 (each with its own reasoning), extract the final answers, and return the most common one.
This is [self-consistency](https://arxiv.org/abs/2203.11171), applied on top of [chain-of-thought](https://arxiv.org/abs/2201.11903).
**Why:** it's Condorcet's jury theorem. If each sample is independently right with probability p > 0.5, a majority vote is right more often than any single sample, and the gap widens with k.

```python
import math
import numpy as np

def majority_correct(p, k):
    """P(a strict majority of k independent samples is correct), k odd."""
    return sum(math.comb(k, i) * p**i * (1 - p)**(k - i) for i in range(k // 2 + 1, k + 1))

for p in [0.6, 0.7]:
    print(f"single-sample accuracy {p}: " + ", ".join(f"k={k}: {majority_correct(p, k):.3f}" for k in [1, 5, 11, 21]))

rng = np.random.default_rng(0)                         # now with CORRELATED errors: each question has its own difficulty
p_question = rng.beta(1.2, 0.8, size=20000)            # mean accuracy is still 0.6, but spread across questions
votes = rng.random((20000, 11)) < p_question[:, None]
print(f"correlated case: single sample {votes[:, 0].mean():.3f}, majority of 11: {(votes.sum(1) >= 6).mean():.3f}")
```

With independent samples at p = 0.6, voting over 11 samples lifts accuracy from 0.600 to **0.753**, and over 21 samples to **0.826**.
But real samples are **correlated**: a question the model fundamentally misunderstands stays wrong however many times you ask.
In the simulation, where each question has its own difficulty but the mean accuracy is still about 0.6, voting over 11 samples gives only **0.635** (from a single-sample 0.604).
On real tasks, expect the correlated picture, and measure it.

**The trap:** voting needs a canonical final answer to compare. For free-form text, use an LLM to cluster equivalent answers, or pick the answer most similar to the others.

## 2. Use an LLM as a labeler, then measure it like one

**When:** you need thousands of labels (relevance judgments, classification, eval grades) and humans are slow.
**How:** write a rubric, have the LLM label a sample that **you have also labeled** (50–200 items), and measure agreement **corrected for chance** with Cohen's κ, not raw agreement.
Iterate on the rubric with the disagreements in front of you. Only then label at scale, and re-audit a random sample regularly.

```python
def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    p_obs = np.mean(a == b)
    p_chance = sum(np.mean(a == c) * np.mean(b == c) for c in np.union1d(a, b))
    return (p_obs - p_chance) / (1 - p_chance)

rng = np.random.default_rng(1)
human = (rng.random(1000) < 0.9).astype(int)           # 90% of answers are "correct"
lazy_judge = np.ones(1000, int)                        # a judge that always says "correct"
noisy_judge = np.where(rng.random(1000) < 0.92, human, 1 - human)
for name, j in [("always-correct judge", lazy_judge), ("real judge, 92% raw agreement", noisy_judge)]:
    print(f"{name:30s} raw agreement {np.mean(j == human):.3f}   kappa {cohen_kappa(human, j):.3f}")
```

A judge that *always* says "correct" agrees with you **89.2%** of the time on this data, and has κ = **0.000**: it carries no information at all.
The noisy judge, with 92% raw agreement, has κ = **0.677** ("substantial", on the usual Landis–Koch scale).
Report κ, and look at the confusion matrix: a judge that misses most *failures* is useless for catching regressions, whatever its accuracy.

**The traps:** LLM judges prefer **longer** answers, answers in the **first position** of a pairwise comparison, and answers from **their own model family**.
Randomize the order, compare at similar lengths, and use a judge from a different family when you can.

## 3. Ask for reasoning first, and the answer in a fixed format

**When:** any non-trivial task.
**Why:** a model generates left to right. If the answer comes first, the "reasoning" after it is a justification, not a computation. [Chain-of-thought](https://arxiv.org/abs/2201.11903)
works because the intermediate tokens are computation that later tokens can condition on.
**How:** "Think step by step inside `<reasoning>` tags, then give the final answer as JSON `{"label": ...}`." Parse only the JSON. Use structured outputs (constrained decoding) where your API supports them.
For reasoning models, which already think internally, just ask for the final format.

## 4. Retrieval: hybrid first, rerank second, and contextualize chunks

**When:** any RAG system. Retrieval failures cause most bad RAG answers, so fix retrieval before the prompt.
**How, in order of value for effort** (measure each step on a labeled query set with recall@k and NDCG, as in [Lab 11](../labs/11-retrieval-metrics/README.md)):
1. **Hybrid search:** BM25 + dense embeddings, fused with reciprocal rank fusion. BM25 catches exact identifiers, names and rare terms that embeddings blur. Embeddings catch paraphrases.
2. **Rerank** the top 50–100 with a cross-encoder. It reads the query and the chunk *together*, which a bi-encoder can't.
3. **Contextualize chunks** before indexing. Prepend each chunk with a short, generated description of where it sits in its document ("This chunk is from the 2024 annual report, section on revenue…").
   Anthropic's [contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval) write-up combines this with BM25 and reranking.
4. **Query rewriting / HyDE** (GEN-05) for short or vague queries.

**The trap:** tuning chunk size, k, and the embedding model by eye on three queries. Build the labeled query set first, even 30 queries.

## 5. Put the stable part of the prompt first

**When:** high-volume LLM calls with a long shared prefix (system prompt, tool definitions, few-shot examples, a shared document).
**Why:** inference servers and APIs **cache the KV states of prompt prefixes** ([GEN-08](../lessons/llms-genai/08-efficient-llm-inference.md)). A cache hit makes the cached part much cheaper and faster.
But a cache matches only an *identical prefix*. One changed token early in the prompt invalidates everything after it.

```python
def cached_tokens(prev, cur):
    """Tokens of `cur` served from a prefix cache holding `prev` (the longest common prefix)."""
    n = 0
    for a, b in zip(prev, cur):
        if a != b:
            break
        n += 1
    return n

system = ["SYS"] * 1500                       # 1,500 tokens of instructions + tool definitions
docs = ["DOC"] * 3000                         # 3,000 tokens of a shared reference document
def prompt(order, user_q, timestamp):
    parts = {"ts": [f"TIME{timestamp}"] * 10, "sys": system, "docs": docs, "q": [f"Q{user_q}"] * 50}
    return [tok for key in order for tok in parts[key]]

for order in [("ts", "sys", "docs", "q"), ("sys", "docs", "ts", "q")]:
    hits = [cached_tokens(prompt(order, i - 1, i - 1), prompt(order, i, i)) for i in range(1, 101)]
    total = len(prompt(order, 0, 0))
    print(f"order {'/'.join(order):17s} cached fraction: {np.mean(hits) / total:.3f}")
```

Putting a timestamp at the top of the prompt makes the cached fraction **0.000**. Moving it down next to the user's question raises it to **0.987**, with the same content.
Order prompts as: static instructions → tools → long shared context → per-request details → the question.

## 6. Mind the middle of the context

**When:** you pass many retrieved chunks.
**Why:** models use information at the **start and end** of a long context better than in the middle ([Lost in the Middle](https://arxiv.org/abs/2307.03172)).
Newer long-context models suffer less from this, but it's cheap to guard against.
**How:** retrieve fewer, better chunks (that's §4), and place the best-ranked chunks at the edges:

```python
def edges_first(chunks_best_first):
    """Order chunks so ranks 1, 3, 5, ... fill from the front and 2, 4, 6, ... from the back."""
    front, back = chunks_best_first[0::2], chunks_best_first[1::2]
    return front + back[::-1]

print(edges_first(["r1", "r2", "r3", "r4", "r5", "r6"]))
```

This places the two best chunks first and last, `['r1', 'r3', 'r5', 'r6', 'r4', 'r2']`, and the weakest in the middle. Measure it on your own eval set, as with every trick on this page.

## 7. Distill the expensive pipeline into a cheap model

**When:** a big model with a long prompt works well on a narrow, high-volume task (classification, extraction, routing).
**How:** run the big pipeline on a few thousand real inputs, keep the outputs that pass your checks, and fine-tune a small model (or train a classical classifier on embeddings) on them.
Same idea as [knowledge distillation](https://arxiv.org/abs/1503.02531). Evaluate the student against held-out *human* labels, not just its agreement with the teacher.

## 8. Few-shot examples from your failures

**When:** the model makes the *same kind* of mistake repeatedly.
**How:** take your eval's failure cases, write the correct output for 3–5 of them, and add them as few-shot examples. Re-run the whole eval: few-shot examples fix their own category and can quietly break others.
For a large pool of examples, *retrieve* the most similar few for each query (dynamic few-shot).

## 9. Make the eval set first, and make it adversarial

**When:** always, before any of the above.
**How:** 30–100 real or realistic inputs, including the hard cases: ambiguous questions, unanswerable ones, long inputs, inputs in another language, and prompt-injection attempts ([GEN-09](../lessons/llms-genai/09-llm-security-safety.md)).
Every trick on this page is a hypothesis until it moves *that* number. It's the LLM version of [*Machine Learning Yearning*](https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf)'s
"set up a dev set and a single-number metric first".

---

## Cheat sheet

| Situation | Trick |
|---|---|
| Checkable answers, and accuracy matters more than cost | Self-consistency voting (§1), measured; correlated errors cap the gain |
| Need many labels | LLM labeler validated with κ against your labels (§2) |
| Wrong answers that sound right | Reasoning before the answer, in a structured format (§3) |
| Bad RAG answers | Hybrid retrieval → rerank → contextual chunks (§4) |
| Cost or latency | Stable prefix first, for cache hits (§5); distill (§7) |
| Many chunks in the context | Fewer chunks; best at the edges (§6) |
| A repeated failure type | Few-shot from failures, then re-run the full eval (§8) |
