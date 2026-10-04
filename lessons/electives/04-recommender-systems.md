# EL-04: Recommender Systems

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Elective | ~10 h | L2 | CORE-04, DL-02 (DL-06 for the sequential part) |

## Why this matters
Recommenders drive most of what people see online: feeds, shops, video, music. They combine embeddings, ranking, large-scale retrieval,
and tricky evaluation (offline metrics vs real user behaviour), which makes them a great capstone for classical ML, deep learning, and ML systems.

## Learning goals
By the end you can:
- Describe the standard architecture: candidate generation → scoring → re-ranking.
- Implement collaborative filtering with matrix factorization, and explain implicit vs explicit feedback.
- Build two-tower (dual-encoder) retrieval and a feature-rich ranking model (Wide & Deep / DeepFM style).
- Build a sequential recommender (SASRec-style) for next-item prediction.
- Evaluate with ranking metrics (recall@k, NDCG, MRR), and explain why offline gains may not survive an A/B test.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Recommender Systems](../../notes/electives/04-recommender-systems.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [Google: Recommendation Systems course](https://developers.google.com/machine-learning/recommendation) | The whole course: overview, candidate generation, matrix factorization, DNN models, retrieval/scoring/re-ranking | 2.5 h |
| 2 | **Read + Build** | [fastbook](https://github.com/fastai/fastbook) | Ch. 8 "Collaborative Filtering" (dot product → embeddings → neural CF) | 2 h |
| 3 | **Intuition** | [Jay Alammar: Skip-gram for recommendations](https://jalammar.github.io/skipgram-recommender-talk/) | The talk write-up: word2vec ideas applied to items | 30 min |
| 4 | **Build** | [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) | Run 3 notebooks: an MF/ALS baseline, a deep model, and the evaluation notebook. Compare them on MovieLens. | 2.5 h |
| 5 | **Read** | [SASRec](https://arxiv.org/abs/1808.09781) | §1–3 (the model) | 1 h |
| 6 | *Read (optional)* | [Evidently system-design case studies](https://www.evidentlyai.com/ml-system-design) | 2 recommender case studies from real companies | 45 min |

## Check your understanding
1. Why do recommenders split retrieval (fast, approximate) from ranking (slow, precise)?
2. What does matrix factorization learn? How do you handle a brand-new user or item (cold start)?
3. Implicit feedback: why are missing interactions not negatives, and how do you sample negatives?
4. What does the "wide" part of Wide & Deep memorize that the "deep" part can't?
5. Why can optimizing offline NDCG produce a worse product (popularity bias, feedback loops)?
6. *(debug)* Your model's recall@10 is excellent offline, but recommendations are all blockbusters. What's happening, and how do you fix both the metric and the model?

## Mini-project
**Task:** On MovieLens, build popularity → MF → two-tower → SASRec, using a time-based split. Report recall@10/NDCG@10, plus
coverage/diversity of the recommended items.
**Deliverable:** A leaderboard table, and a short discussion of accuracy vs diversity.

## Go deeper
- Papers: [Wide & Deep](https://arxiv.org/abs/1606.07792) · [DeepFM](https://arxiv.org/abs/1703.04247) · [Neural CF](https://arxiv.org/abs/1708.05031) · [BERT4Rec](https://arxiv.org/abs/1904.06690) · [DLRM](https://arxiv.org/abs/1906.00091)
- [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) Ch. 8: evaluation metrics shared with search.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 12: Specialized topics](../../toolbox/12-specialized-topics.md) (recommender systems).
- **Papers:** [Tabular, time series, recsys & causal](../../papers/09-tabular-timeseries-recsys-causal.md) (recommender systems section).
- **Implement it yourself:** MF with SGD in NumPy, then [from-scratch ladder](../../exercises/from-scratch-ladder.md) rung 39 (ranking metrics).
- **Drills:** more in [exercises/](../../exercises/README.md).
