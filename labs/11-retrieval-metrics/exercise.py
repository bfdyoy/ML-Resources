# EXERCISE: generated from solution.py by scripts/make_lab_stubs.py.
# Replace each `raise NotImplementedError` with your code, then run the tests (see README.md).
"""Lab 11: BM25 scoring, ranking metrics, and reciprocal rank fusion.

Documents are lists of tokens. A ranking is a list of doc ids, best first. Relevance labels are dicts {doc_id: grade}.
"""
import math


def bm25_scores(query, docs, k1=1.5, b=0.75):
    """BM25 score of every doc for the query tokens (a repeated query term counts each time it appears):
        score(d) = sum over t in query of idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
        idf(t) = ln((N - df + 0.5) / (df + 0.5) + 1)       (Lucene's always-positive variant)
    where tf is the count of t in d, df the number of docs containing t, N = len(docs). Return a list of floats."""
    raise NotImplementedError("your code here")


def recall_at_k(ranking, relevant, k):
    """Fraction of the relevant ids (a set) that appear in the top k. 0.0 if `relevant` is empty."""
    raise NotImplementedError("your code here")


def mean_reciprocal_rank(rankings, relevants):
    """Mean over queries of 1 / (rank of the first relevant id), counting ranks from 1; 0 if none is retrieved."""
    raise NotImplementedError("your code here")


def ndcg_at_k(ranking, grades, k):
    """NDCG@k with linear gains: DCG = sum_i grades.get(doc_i, 0) / log2(i + 2) over the top k (i from 0),
    divided by the DCG of the ideal ordering of ALL graded docs (truncated to k). 0.0 if the ideal DCG is 0."""
    raise NotImplementedError("your code here")


def reciprocal_rank_fusion(rankings, k=60):
    """Fuse several rankings: score(d) = sum over rankings containing d of 1 / (k + rank), ranks from 1.
    Return the fused list of ids sorted by descending score; ties broken by id (ascending) for determinism.
    Why it works: it uses only ranks, so BM25 scores and cosine similarities never need to be put on one scale."""
    raise NotImplementedError("your code here")
