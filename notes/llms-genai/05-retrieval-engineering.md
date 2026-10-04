# GEN-05 notes: Retrieval Engineering for RAG

[← Lesson GEN-05](../../lessons/llms-genai/05-retrieval-engineering.md) · [All notes](../README.md) · [← GEN-03 notes](03-evaluating-llm-apps.md) · Next: [GEN-06 notes →](06-agents-tool-use.md)

> **Reading time** ≈ 65 min. **You need:** [CORE-07 notes](../core-ml/07-feature-engineering-pipelines-leakage.md) §3 (TF-IDF), [DL-05 notes](../deep-learning/05-embeddings-sequences-attention.md) §1 (embeddings, contrastive training), [CORE-06 notes](../core-ml/06-unsupervised-learning.md) §2 (k-means), and [GEN-02 notes](02-adapting-llms-finetuning-rag.md) §3 (the RAG pipeline).

---

## Where we are

In RAG, **the generator can only be as good as what the retriever hands it**. This lesson treats retrieval as its own ML system, with its own metrics, models, and indexes. Each component gets measured in isolation before you look at end-to-end answers.

---

## 1. Measuring retrieval

You need a labeled set of (query, relevant documents) pairs: hand-written, mined from logs, or LLM-generated and then human-checked.

| Metric | Definition | Reads as |
|---|---|---|
| **Recall@k** | Fraction of a query's relevant docs that appear in the top $k$ | "Did the right thing make it into the context?" (the key RAG metric) |
| **Precision@k** | Fraction of the top $k$ that are relevant | How much noise is in the context |
| **MRR** | Mean of $1/\text{rank}$ of the first relevant hit | "How high is the first good result?" |
| **NDCG@k** | Graded relevance, discounted by position, normalized by the ideal ordering | Ranking quality when relevance has degrees |

```math
\mathrm{DCG@}k = \sum_{i=1}^{k}\frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)},\qquad \mathrm{NDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}.
```

*Worked MRR:* first relevant hit at ranks 1, 3, and "not found" for three queries gives $(1 + 1/3 + 0)/3 \approx 0.444$.

---

## 2. Lexical retrieval: from TF-IDF to BM25

TF-IDF (CORE-07 §3) has two problems for search:

1. A term appearing 20 times counts 20× as much as one appearing once. But relevance **saturates**: the 20th mention adds little.
2. Long documents contain more terms, so they win unfairly.

**BM25** fixes both. For query terms $t$, document $d$, term frequency $f(t,d)$, document length $\lvert d\rvert$, and average length $\overline{\lvert d\rvert}$:

```math
\text{BM25}(q, d) = \sum_{t\in q} \text{IDF}(t)\cdot\frac{f(t,d)\,(k_1 + 1)}{f(t,d) + k_1\Big(1 - b + b\,\frac{\lvert d\rvert}{\overline{\lvert d\rvert}}\Big)},\qquad \text{IDF}(t) = \ln\Big(\frac{N - n_t + 0.5}{n_t + 0.5} + 1\Big).
```

- $k_1$ (about 1.2–2) sets **saturation**: as $f \to \infty$, the term's contribution tends to $\text{IDF}\cdot(k_1 + 1)$ instead of growing without bound.
- $b$ (about 0.75) sets **length normalization**: $b = 0$ ignores length, and $b = 1$ fully normalizes.

**When lexical beats dense:** exact identifiers (error codes, SKUs, function names), rare proper nouns, very domain-specific jargon the embedding model never saw, and queries where a keyword *must* match. It's also cheap, needs no training, and is easy to explain.

---

## 3. Dense retrieval: bi-encoders

Encode queries and documents **separately** into vectors and score them by cosine similarity: $s(q, d) = \cos(E(q), E(d))$. Document vectors are computed **once, offline**, so search is a nearest-neighbour lookup.

**Training (contrastive, InfoNCE).** For a batch of (query, positive doc) pairs, treat the *other* docs in the batch as negatives:

```math
\mathcal{L} = -\log\frac{\exp\big(s(q, d^+)/\tau\big)}{\sum_{j}\exp\big(s(q, d_j)/\tau\big)} .
```

It's a softmax classification: "which doc in this batch belongs to this query?" (The same loss as CLIP, [CV-03](../vision/03-clip-vision-language-models.md).) Bigger batches give more negatives. **Hard negatives** (similar but wrong docs, mined with BM25) teach finer distinctions. $\tau$ is a temperature.

**Choosing a model:** the MTEB leaderboard ranks models across many tasks. Look at the **retrieval** tasks closest to your domain, and weigh size, latency, vector dimension, context length, and licence. Then **evaluate on your own labeled queries**, because leaderboard rank often doesn't transfer to your domain.

## 4. Cross-encoders: rerank the shortlist

A cross-encoder feeds `[query; document]` *together* through a transformer and outputs one relevance score. Every query token can attend to every document token, so it's **much more accurate** than comparing two independently computed vectors.
But nothing can be precomputed: scoring $N$ documents takes $N$ forward passes. Hence the standard **two-stage** design:

1. A cheap retriever (BM25 / bi-encoder / hybrid) gets the top 50–200 candidates.
2. A cross-encoder reranks them, and the top 5–10 go to the LLM.

---

## 5. Approximate nearest-neighbour (ANN) indexes

Exact search over $N$ vectors of dimension $d$ costs $O(Nd)$ per query. Memory is $N\cdot d\cdot4$ bytes: 1M × 768 FP32 ≈ **3.1 GB**. ANN indexes trade a little recall for big speed and memory gains:

- **IVF (inverted file):** k-means the vectors into $n_\text{list}$ clusters. At query time, search only the `nprobe` nearest clusters. Raising `nprobe` gives more recall and slower queries.
- **PQ (product quantization):** split each vector into $m$ sub-vectors, and replace each one with the ID of its nearest of 256 sub-centroids (1 byte). The vector shrinks from $4d$ bytes to $m$ bytes: 768 FP32 dimensions (3,072 B) with $m = 96$ becomes 96 B, **32× smaller**. Distances are approximated with lookup tables. They're rough, so real systems **re-score a PQ shortlist with full-precision vectors** (in the demo below, recall@10 rises from 0.16 to 0.87).
- **HNSW:** a multi-layer **proximity graph**. Search starts at the sparse top layer and greedily walks toward the query, descending layer by layer.
  - `M` (links per node): more recall, more memory.
  - `efConstruction`: build quality, at the cost of build time.
  - `efSearch` (the candidate list size at query time): **the main recall/latency knob**.
  - HNSW is fast with high recall, but memory-hungry (full vectors plus the graph).

Always measure **recall@k against exact search** on your data when tuning these knobs.

---

## 6. Hybrid search and fusion

Lexical and dense retrieval fail on *different* queries, so combine them. Their scores live on incomparable scales (BM25 is unbounded, cosine is in $[-1,1]$), so fuse **ranks** instead of scores, with **reciprocal rank fusion (RRF)**:

```math
\text{RRF}(d) = \sum_{\text{rankers } r} \frac{1}{k + \text{rank}_r(d)},\qquad k \approx 60 .
```

It's robust because it needs no score calibration and no training. A document that ranks well in *either* list gets a good score, one that ranks well in *both* gets the best, and $k$ damps the dominance of the very top positions.

## 7. Query-side tricks

- **Query rewriting / expansion:** fix typos, expand acronyms, split multi-part questions.
- **HyDE:** ask an LLM to write a *hypothetical answer*, then embed that instead of the question. Answers look more like documents than questions do, so the embedding lands closer to the relevant passages.
  **When it hurts:** when the LLM's guess is wrong or off-topic (it drags the retrieval toward the hallucination), when the query is a precise keyword or ID lookup, and when latency or cost is tight (it adds an LLM call).

## 8. Chunking, revisited

Retrieval recall favours **small, focused chunks** (one idea, one sharp embedding). Answer quality favours **enough context** around the hit. Use structure-aware splitting, overlap, or small-to-big expansion (GEN-02 §3.3), and tune by measuring **both** recall@k and end-to-end answer quality.

---

## 9. "Recall@10 is 0.9, but answers are still wrong"

Retrieval is fine *on your eval set*. Look further down the pipeline, and at the eval set itself:

1. **Ranking and packing:** is the gold chunk at position 9, buried in the middle of a long context ("lost in the middle"), or truncated? Add a reranker, send fewer and better chunks, and put the best ones first.
2. **Generation:** run the oracle test (GEN-02 §3.2). Is the model ignoring the context, or mixing it with prior knowledge? Fix the prompt (require citations, "answer only from the context"), or use a better model.
3. **The eval set:** does it represent real user queries? Sample production queries, and label the failures.
4. **Answerability:** many questions need several chunks, aggregation, or information that isn't in the corpus. Add multi-hop handling or "I don't know".

```python
import numpy as np, math
from collections import Counter
rng = np.random.default_rng(0)

# --- BM25 from scratch: saturation and length normalization ---------------------------------------
docs = ["error E1234 occurs when the disk is full",
        "the disk the disk the disk the disk is a storage device and the disk spins",
        "to fix error codes check the logs and the disk usage",
        "a long manual about many topics including disks networks printers screens keyboards and an error"]
tok = [d.split() for d in docs]; N = len(tok); avgdl = np.mean([len(t) for t in tok])
df = Counter(w for t in tok for w in set(t))
def bm25(q, k1=1.5, b=0.75):
    s = np.zeros(N)
    for w in q.split():
        idf = math.log((N - df[w] + 0.5) / (df[w] + 0.5) + 1)
        for i, t in enumerate(tok):
            f = t.count(w)
            s[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avgdl)) if f else 0
    return s
def raw_tf(q):
    return np.array([sum(t.count(w) for w in q.split()) for t in tok], float)
q = "disk error"
print("raw term counts ranks:", np.argsort(-raw_tf(q)).tolist(), " BM25 ranks:", np.argsort(-bm25(q)).tolist())
print("BM25 picks the exact-ID doc for 'E1234':", int(np.argmax(bm25("E1234"))) == 0)

# --- Ranking metrics ----------------------------------------------------------------------------------
def mrr(ranks): return np.mean([1 / r if r else 0 for r in ranks])
def ndcg(rels, k):
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sorted(rels, reverse=True)
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
    return dcg / idcg
print("MRR for first hits at ranks 1, 3, none:", round(mrr([1, 3, None]), 3))
print("NDCG@5 of graded relevances [0, 2, 3, 0, 1]:", round(ndcg([0, 2, 3, 0, 1], 5), 3))

# --- Reciprocal rank fusion --------------------------------------------------------------------------------
def rrf(rankings, k=60):
    scores = Counter()
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1): scores[d] += 1 / (k + rank)
    return [d for d, _ in scores.most_common()]
lexical = ["d7", "d2", "d9", "d4"]; dense = ["d2", "d5", "d7", "d1"]
print("RRF fused:", rrf([lexical, dense]))
```

```python
# --- IVF and PQ by hand vs exact search: the recall / memory trade-off ----------------------------------------
from sklearn.cluster import KMeans
Nv, d = 20000, 64
centers = rng.normal(size=(50, d))
X = centers[rng.integers(0, 50, Nv)] + 0.5 * rng.normal(size=(Nv, d))
Q = centers[rng.integers(0, 50, 200)] + 0.5 * rng.normal(size=(200, d))
exact = np.argsort(((Q[:, None] - X[None]) ** 2).sum(-1), axis=1)[:, :10]

km = KMeans(100, n_init=1, random_state=0).fit(X)
lists = [np.where(km.labels_ == c)[0] for c in range(100)]
def ivf_search(q, nprobe):
    near = np.argsort(((km.cluster_centers_ - q) ** 2).sum(1))[:nprobe]
    cand = np.concatenate([lists[c] for c in near])
    return cand[np.argsort(((X[cand] - q) ** 2).sum(1))[:10]], len(cand)
for nprobe in [1, 4, 16]:
    res = [ivf_search(q, nprobe) for q in Q]
    rec = np.mean([len(set(r) & set(e)) / 10 for (r, _), e in zip(res, exact)])
    print(f"IVF nprobe={nprobe:2d}: recall@10={rec:.3f}, scanned {np.mean([c for _, c in res]) / Nv:.1%} of vectors")

m = 8; sub = d // m                                          # PQ: 8 sub-vectors x 256 centroids = 8 bytes/vector
codebooks, codes = [], []
for j in range(m):
    kmj = KMeans(256, n_init=1, random_state=0).fit(X[:5000, j*sub:(j+1)*sub])
    codebooks.append(kmj.cluster_centers_); codes.append(kmj.predict(X[:, j*sub:(j+1)*sub]))
codes = np.stack(codes, 1)
def pq_search(q, shortlist=10):
    tables = [((codebooks[j] - q[j*sub:(j+1)*sub]) ** 2).sum(1) for j in range(m)]   # lookup tables
    dist = sum(tables[j][codes[:, j]] for j in range(m))
    return np.argsort(dist)[:shortlist]
def pq_refine(q, shortlist=200):                              # re-score a PQ shortlist with exact distances
    cand = pq_search(q, shortlist)
    return cand[np.argsort(((X[cand] - q) ** 2).sum(1))[:10]]
rec_pq = np.mean([len(set(pq_search(q)) & set(e)) / 10 for q, e in zip(Q, exact)])
rec_ref = np.mean([len(set(pq_refine(q)) & set(e)) / 10 for q, e in zip(Q, exact)])
print(f"PQ: {d*4} bytes -> {m} bytes per vector ({d*4//m}x smaller); recall@10 = {rec_pq:.3f} alone, {rec_ref:.3f} with exact re-scoring of the top 200")
```

---

## Pitfalls & misconceptions

- **Judging retrieval only through end-to-end answers.** Measure recall@k directly.
- **Picking an embedding model from the leaderboard alone.**
- **Adding scores from different retrievers.** Fuse ranks (RRF) or calibrate first.
- **Tuning HNSW or IVF without measuring recall against exact search.**
- **Embedding queries and documents with different models, or different prefixes.** Many models require "query:" and "passage:" prefixes.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Recall@k / MRR | coverage of the relevant docs / mean $1/\text{rank}$ of the first hit |
| NDCG | $\sum(2^{rel}-1)/\log_2(i+1)$, normalized by the ideal |
| BM25 | saturation via $k_1$, length normalization via $b$ |
| InfoNCE | $-\log\frac{e^{s(q,d^+)/\tau}}{\sum_j e^{s(q,d_j)/\tau}}$ |
| Two-stage | bi-encoder/BM25 recall → cross-encoder precision |
| PQ size | $m$ bytes per vector (vs $4d$) |
| RRF | $\sum_r 1/(k + \text{rank}_r)$, $k \approx 60$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. What does BM25 add over tf-idf; when does it beat dense retrieval?</summary>

Term-frequency saturation ($k_1$) and document-length normalization ($b$). It wins on exact identifiers, rare names, out-of-domain jargon, and must-match keywords, where embeddings blur the exact tokens.
</details>

<details>
<summary>2. Bi-encoder vs cross-encoder roles.</summary>

A bi-encoder embeds the query and the docs separately, so doc vectors are precomputed and search is fast ANN: good for recall over millions of docs. A cross-encoder reads the query and doc jointly (full attention between them), which is more accurate but needs one forward pass per pair: good for reranking a short list.
</details>

<details>
<summary>3. HNSW parameters for recall vs speed and memory.</summary>

`M` (links per node: recall vs memory), `efConstruction` (build quality vs build time), and `efSearch` (query-time beam width: recall vs latency, the main knob).
</details>

<details>
<summary>4. How chunk size pulls recall and answer quality in opposite directions.</summary>

Smaller chunks give sharper embeddings and better retrieval matching, but less context for answering. Larger chunks give more context per hit, but fuzzier embeddings and fewer chunks fit. Tune both metrics, or use small-to-big retrieval.
</details>

<details>
<summary>5. Why does HyDE help, and when could it hurt?</summary>

A hypothetical answer resembles real documents more than a short question does, so its embedding lands nearer the relevant passages. It hurts when the generated answer is wrong or off-topic, for exact keyword or ID lookups, and when the extra LLM call's latency or cost matters.
</details>

<details>
<summary>6. What is RRF, and why is it robust?</summary>

It sums $1/(k + \text{rank})$ across rankers. It uses only ranks, so it needs no score calibration between lexical and dense systems, it rewards documents found by either, and it has essentially one forgiving hyperparameter.
</details>

<details>
<summary>7. Recall@10 = 0.9, but wrong answers.</summary>

Look past retrieval: ranking and packing (the gold chunk buried or truncated; add a reranker and put the best chunks first), generation faithfulness (the oracle test; prompt for citations), eval-set representativeness (sample real queries), and answerability (multi-hop or missing information) (§9).
</details>

## Where this leads

Next: [GEN-06 notes](06-agents-tool-use.md). Retrieval is one *tool* an LLM can use. GEN-06 generalizes this to any tool (search, code, APIs) and to multi-step plans. That brings new failure modes, and new ways to evaluate them.
