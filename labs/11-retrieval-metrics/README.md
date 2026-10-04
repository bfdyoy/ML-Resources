# Lab 11: BM25, ranking metrics & reciprocal rank fusion

[← Labs](../README.md) · Lesson: [GEN-05 Retrieval engineering](../../lessons/llms-genai/05-retrieval-engineering.md) · Notes: [GEN-05 notes](../../notes/llms-genai/05-retrieval-engineering.md)

**Time** ≈ 1.5 h · **You'll practise:** the BM25 formula's two knobs (`k1` saturation, `b` length normalization), the three ranking metrics every RAG leaderboard uses, and rank-based fusion.

| Function | Checked against |
|---|---|
| `bm25_scores` | a hand computation; saturation in tf; shorter docs win at equal tf; `b = 0` |
| `recall_at_k`, `mean_reciprocal_rank` | hand values |
| `ndcg_at_k` | `sklearn.metrics.ndcg_score` |
| `reciprocal_rank_fusion` | a hand-fused example |

```bash
pytest labs/11-retrieval-metrics
```

**Bonus:** use your `bm25_scores` and any sentence-embedding model on 50 documents of your own, with 20 labeled queries. Is RRF of the two
better than either alone on recall@5? Then try `k = 1` vs `k = 60` in RRF. What does `k` control?
