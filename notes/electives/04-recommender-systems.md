# EL-04 notes: Recommender Systems

[← Lesson EL-04](../../lessons/electives/04-recommender-systems.md) · [All notes](../README.md) · [← EL-03 notes](03-reinforcement-learning.md) · Next: [EL-05 notes →](05-causal-inference-uplift.md)

> **Reading time** ≈ 60 min. **You need:** [MATH-01 notes](../math/01-linear-algebra.md) §C5 (low-rank approximation), [CORE-04 notes](../core-ml/04-generalization-validation-regularization.md) §3 (ridge), [GEN-05 notes](../llms-genai/05-retrieval-engineering.md) (retrieval metrics, two-stage design, ANN), and [DL-06 notes](../deep-learning/06-transformers.md) (for sequential recommenders).

---

## Where we are

Recommenders rank items for users, from millions of candidates, in milliseconds, using feedback that is **biased by what the system showed before**. They combine ideas from all over the curriculum:

- low-rank matrices (MATH-01);
- contrastive retrieval (GEN-05);
- feature-rich ranking (CORE-05, DL);
- transformers (DL-06);
- exploration and feedback loops (EL-03, PROD-01).

---

## 1. The architecture: retrieve, then rank

| Stage | Input | Model | Output |
|---|---|---|---|
| **Candidate generation** | millions of items | cheap: embeddings + ANN, co-occurrence, popularity, rules | hundreds of candidates |
| **Ranking** | hundreds | expensive: GBM or deep network with rich user × item × context features | scored list |
| **Re-ranking** | the scored list | business logic: diversity, freshness, dedup, fairness, filters | final slate |

**Why split?** A rich ranking model costs, say, 1 ms per item. Scoring 10M items per request would take hours. Retrieval must be **sublinear** (ANN over precomputed embeddings, GEN-05 §5). Ranking can afford to be precise because it sees only a few hundred items.

---

## 2. Collaborative filtering with matrix factorization

### 2.1 The model

The user–item matrix $R$ (ratings, or interactions) is huge and mostly empty. Assume it's approximately **low-rank**: each user $u$ and item $i$ gets a $k$-dimensional vector, and

```math
\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i .
```

$\mu$ is the global mean, $b_u$ and $b_i$ are biases (a generous user, a popular item), and the dot product captures **taste × attributes**. The learned dimensions often line up with interpretable axes ("action vs romance", "mainstream vs niche"), without anyone defining them. Fit on the *observed* entries only:

```math
\min\sum_{(u,i)\in\text{obs}}\big(r_{ui} - \hat r_{ui}\big)^2 + \lambda\big(\lVert p_u\rVert^2 + \lVert q_i\rVert^2 + b_u^2 + b_i^2\big).
```

**Alternating least squares (ALS):** with the item vectors fixed, each user's problem is a **ridge regression** with a closed form (CORE-04 §3.1), $p_u = (Q_u^\top Q_u + \lambda I)^{-1}Q_u^\top r_u$, over the items that user rated. Then alternate. It parallelizes trivially. SGD works too.

### 2.2 Implicit feedback

Most data is **implicit**: clicks, views, purchases. There are no explicit dislikes. **A missing interaction isn't a negative.** The user may never have *seen* the item. Two standard treatments:

- **Weighted MF:** treat every entry as "preference $\in\lbrace0,1\rbrace$", with **confidence** $c_{ui} = 1 + \alpha\cdot\text{count}_{ui}$. Observed interactions get high weight, and the missing ones a low weight.
- **Pairwise ranking (BPR):** for a user, an item they interacted with should score above a *sampled* item they didn't. Loss: $-\log\sigma(\hat r_{ui} - \hat r_{uj})$.

**Negative sampling:** uniform sampling mostly picks obscure items (easy negatives). Popularity-weighted sampling gives harder negatives. In-batch negatives (as in CLIP) are efficient, but they over-sample popular items, so apply a **logQ correction**: subtract $\log$ (the item's sampling probability) from its score, so popular items aren't unfairly penalized.

### 2.3 Cold start

A new user or item has no interactions, so it has no learned vector. The options:

- **content features:** item text and image embeddings, attributes; user demographics and context, through a two-tower model with feature inputs (§3);
- **popularity** or trending items, as a fallback;
- **onboarding questions;**
- **exploration** (bandits, EL-03 §2.3) to gather signal quickly.

---

## 3. Neural recommenders

- **Two-tower (dual encoder):** a user tower (history, features, context) and an item tower (ID embedding plus content features), with the score $\langle f(u), g(i)\rangle$. It's trained with a sampled softmax / InfoNCE over in-batch negatives (GEN-05 §3).
  Item embeddings are precomputed, so serving is an ANN lookup. This is the **retrieval** model, and its feature inputs solve cold start.
- **Wide & Deep:** the **wide** part is a linear model on **cross-product features** ("user installed app A AND is shown app B"). It **memorizes** specific, sparse co-occurrences that are reliably predictive.
  The **deep** part learns dense embeddings that **generalize** to unseen combinations. Embeddings tend to over-generalize rare but important exact patterns, and the wide part pins them down. **DeepFM** replaces hand-crafted crosses with automatic factorization-machine interactions.
- **Sequential (SASRec, BERT4Rec):** treat the user's history as a sequence of item tokens, and use causal self-attention to predict the next item (DL-06, with items instead of words). It captures order and recency ("bought a camera → wants a lens").

---

## 4. Evaluation, and why offline wins can lose online

**Offline metrics:** recall@k, NDCG@k, and MRR (GEN-05 §1), computed on held-out interactions, ideally with a **time-based split** (train on the past, test on the future).
Also measure **coverage** (the fraction of the catalogue ever recommended), **diversity** (dissimilarity within a slate), **novelty**, and the metrics *per popularity bucket*.

**Why offline NDCG can mislead:**

- **Exposure (popularity) bias:** the logged data only contains interactions with items the *old* system showed, and those are mostly popular items. A model that recommends blockbusters "predicts" those logs well. The metric rewards matching past exposure, not discovering what users would actually like.
- **Feedback loops:** deploy it, and it shows more blockbusters, which collect more clicks, which train the next model to push them harder. Diversity collapses, and the long tail starves.
- **Missing counterfactuals:** you never observe how a user would have reacted to items that were never shown.

**Debug: excellent offline recall@10, but every recommendation is a blockbuster.**

- *The metric:* evaluate per popularity bucket, add coverage, diversity, and novelty metrics, use **unbiased evaluation** (inverse-propensity weighting by exposure probability, or a small randomized-exposure dataset), and **always compare against a popularity baseline**. If the gap is tiny, your model is a popularity model.
- *The model:* the logQ / popularity correction in training, de-biased negatives, re-ranking for diversity (MMR, calibrated recommendations), exploration slots, and ultimately an **online A/B test** on long-term engagement, not just CTR.

```python
import numpy as np
rng = np.random.default_rng(0)

# --- Synthetic implicit data with popularity skew ------------------------------------------------------
n_users, n_items, k = 500, 300, 8
U_true, V_true = rng.normal(size=(n_users, k)), rng.normal(size=(n_items, k))
popularity = np.exp(rng.normal(0, 1.2, n_items))                     # a few items are much more popular
logits = U_true @ V_true.T / np.sqrt(k) + np.log(popularity)
P = 1 / (1 + np.exp(-(logits - 4)))
R = (rng.random((n_users, n_items)) < P).astype(float)               # observed interactions
test = np.zeros_like(R)                                               # hold out one interaction per user
for u in range(n_users):
    items = np.where(R[u])[0]
    if len(items) > 1:
        i = rng.choice(items); test[u, i] = 1; R[u, i] = 0
print(f"density {R.mean():.3f}; top-10% items get {np.sort(R.sum(0))[::-1][:30].sum() / R.sum():.0%} of interactions")

def recall_at_k(scores, k=10):
    scores = scores.copy(); scores[R > 0] = -np.inf                    # don't recommend already-seen items
    top = np.argsort(-scores, axis=1)[:, :k]
    users = np.where(test.sum(1) > 0)[0]
    hits = [test[u, top[u]].sum() > 0 for u in users]
    return np.mean(hits), len(np.unique(top[users])) / n_items        # recall@k, catalogue coverage

# Baseline 1: popularity
pop_scores = np.tile(R.sum(0), (n_users, 1))
# Model: weighted ALS for implicit feedback (confidence 1 + alpha * r)
def implicit_als(R, k=16, lam=5.0, alpha=10.0, iters=10):
    X, Y = rng.normal(0, 0.1, (n_users, k)), rng.normal(0, 0.1, (n_items, k))
    C = 1 + alpha * R
    for _ in range(iters):
        for u in range(n_users):
            Cu = C[u]; A = (Y.T * Cu) @ Y + lam * np.eye(k); X[u] = np.linalg.solve(A, (Y.T * Cu) @ R[u])
        for i in range(n_items):
            Ci = C[:, i]; A = (X.T * Ci) @ X + lam * np.eye(k); Y[i] = np.linalg.solve(A, (X.T * Ci) @ R[:, i])
    return X @ Y.T
mf_heavy = implicit_als(R, k=8, lam=20.0, alpha=2.0)      # strongly regularized
mf_scores = implicit_als(R, k=8, lam=5.0, alpha=2.0)
mf_light = implicit_als(R, k=32, lam=5.0, alpha=40.0)     # more capacity, more confidence in interactions
for name, sc in [("popularity", pop_scores), ("ALS, lam=20", mf_heavy), ("ALS, lam=5", mf_scores), ("ALS, k=32 a=40", mf_light)]:
    rec, cov = recall_at_k(sc)
    print(f"{name:15s}: recall@10 {rec:.3f}, catalogue coverage {cov:.0%}")
# The offline-best settings sit closest to "recommend what's popular". Tuning only on recall@10
# quietly turns the model into a popularity model; coverage shows the price.
```

```python
# --- A diversity re-ranker (MMR): trade relevance for less redundancy in the slate --------------------------
item_vecs = V_true / np.linalg.norm(V_true, axis=1, keepdims=True)
def mmr(scores_u, lam=0.7, k=10):
    chosen, cand = [], list(np.argsort(-scores_u)[:100])
    while len(chosen) < k:
        def gain(i):
            red = max((item_vecs[i] @ item_vecs[j] for j in chosen), default=0)
            return lam * scores_u[i] - (1 - lam) * red
        best = max(cand, key=gain); chosen.append(best); cand.remove(best)
    return chosen
u = 0
s = mf_scores[u].copy(); s[R[u] > 0] = -np.inf
s_norm = (s - s[np.isfinite(s)].min()) / (s[np.isfinite(s)].max() - s[np.isfinite(s)].min())
plain = list(np.argsort(-s_norm)[:10]); diverse = mmr(np.nan_to_num(s_norm, neginf=0))
avg_sim = lambda L: np.mean([item_vecs[a] @ item_vecs[b] for i, a in enumerate(L) for b in L[i + 1:]])
print(f"mean pairwise similarity in the slate: plain top-10 {avg_sim(plain):.3f}  MMR {avg_sim(diverse):.3f}")
```

---

## Pitfalls & misconceptions

- **Treating unobserved interactions as dislikes.**
- **Random train/test splits** on interaction logs. Use time-based splits.
- **No popularity baseline.**
- **Optimizing CTR only.** It invites clickbait, filter bubbles, and long-term disengagement.
- **Ignoring exposure bias** in both training and evaluation.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| MF prediction | $\mu + b_u + b_i + p_u^\top q_i$ |
| ALS step | $p_u = (Q_u^\top Q_u + \lambda I)^{-1}Q_u^\top r_u$ (a ridge per user) |
| Implicit confidence | $c_{ui} = 1 + \alpha\cdot\text{count}$ |
| BPR | $-\log\sigma(\hat r_{ui} - \hat r_{uj})$ |
| logQ correction | score − $\log P(\text{item sampled})$ |
| Wide vs deep | memorize crosses vs generalize through embeddings |
| Evaluate | time split; recall/NDCG + coverage/diversity/novelty + popularity buckets; then an A/B test |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why split retrieval from ranking?</summary>

Scoring every item with a rich model is far too slow. Cheap retrieval (ANN over embeddings, sublinear) narrows millions of items to hundreds, which an expensive, feature-rich ranker can then score precisely, within the latency budget.
</details>

<details>
<summary>2. What does MF learn, and how do you handle cold start?</summary>

Low-dimensional user-taste and item-attribute vectors (plus biases) whose dot products reconstruct the interactions. For new users or items: content and context features (two-tower models), popularity fallbacks, onboarding, and exploration.
</details>

<details>
<summary>3. Implicit feedback: why missing ≠ negative; negative sampling.</summary>

Absence usually means "never seen", not "disliked". Use confidence-weighted objectives, or pairwise losses with sampled negatives (uniform, popularity-aware, in-batch with a logQ correction), and tune how hard the negatives are.
</details>

<details>
<summary>4. What does the wide part memorize that the deep part can't?</summary>

Specific sparse feature crosses (exact co-occurrences such as "installed A × shown B") that are reliably predictive but rare. Dense embeddings tend to over-generalize these exceptions, and the linear cross terms memorize them exactly.
</details>

<details>
<summary>5. Why can optimizing offline NDCG produce a worse product?</summary>

The logs reflect past exposure, so models that mimic it (popular items) score well offline while users get less novelty and diversity. Deployment then reinforces the bias through feedback loops, and long-term satisfaction can fall.
</details>

<details>
<summary>6. Great offline recall, all blockbusters.</summary>

Popularity bias in the data and the metric. Fix the metric (popularity-stratified recall, coverage, diversity, novelty, IPS or randomized evaluation, comparison with a popularity baseline) and the model (logQ correction, de-biased negatives, diversity re-ranking, exploration), then validate online.
</details>

## Where this leads

Next: [EL-05 notes](05-causal-inference-uplift.md). Recommenders showed that logged data reflects past *decisions*. Causal inference is the general toolkit for asking "what would have happened if we had acted differently?", which is the question behind every A/B test and every uplift model.
