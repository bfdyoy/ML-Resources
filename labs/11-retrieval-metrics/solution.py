"""Lab 11: BM25 scoring, ranking metrics, and reciprocal rank fusion.

Documents are lists of tokens. A ranking is a list of doc ids, best first. Relevance labels are dicts {doc_id: grade}.
"""
import math


def bm25_scores(query, docs, k1=1.5, b=0.75):
    """BM25 score of every doc for the query tokens (a repeated query term counts each time it appears):
        score(d) = sum over t in query of idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
        idf(t) = ln((N - df + 0.5) / (df + 0.5) + 1)       (Lucene's always-positive variant)
    where tf is the count of t in d, df the number of docs containing t, N = len(docs). Return a list of floats."""
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / N
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    scores = []
    for d in docs:
        s = 0.0
        for t in query:
            tf = d.count(t)
            if tf:
                idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
                s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return scores


def recall_at_k(ranking, relevant, k):
    """Fraction of the relevant ids (a set) that appear in the top k. 0.0 if `relevant` is empty."""
    return len(set(ranking[:k]) & relevant) / len(relevant) if relevant else 0.0


def mean_reciprocal_rank(rankings, relevants):
    """Mean over queries of 1 / (rank of the first relevant id), counting ranks from 1; 0 if none is retrieved."""
    total = 0.0
    for ranking, rel in zip(rankings, relevants):
        total += next((1.0 / (i + 1) for i, d in enumerate(ranking) if d in rel), 0.0)
    return total / len(rankings)


def ndcg_at_k(ranking, grades, k):
    """NDCG@k with linear gains: DCG = sum_i grades.get(doc_i, 0) / log2(i + 2) over the top k (i from 0),
    divided by the DCG of the ideal ordering of ALL graded docs (truncated to k). 0.0 if the ideal DCG is 0."""
    dcg = sum(grades.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranking[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank_fusion(rankings, k=60):
    """Fuse several rankings: score(d) = sum over rankings containing d of 1 / (k + rank), ranks from 1.
    Return the fused list of ids sorted by descending score; ties broken by id (ascending) for determinism.
    Why it works: it uses only ranks, so BM25 scores and cosine similarities never need to be put on one scale."""
    scores = {}
    for ranking in rankings:
        for r, d in enumerate(ranking, start=1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + r)
    return sorted(scores, key=lambda d: (-scores[d], d))
