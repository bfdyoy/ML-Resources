import math

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

DOCS = [
    "the cat sat on the mat".split(),
    "dogs and cats are pets".split(),
    "the cat chased the cat".split(),
    "stock markets fell sharply today as markets reacted".split(),
]


def test_bm25_by_hand(impl):
    s = impl.bm25_scores(["cat"], DOCS)
    N, avgdl, df = 4, (6 + 5 + 5 + 8) / 4, 2
    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
    expect0 = idf * 1 * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 6 / avgdl))
    assert math.isclose(s[0], expect0)
    assert s[1] == 0.0 and s[3] == 0.0
    assert s[2] > s[0]                                   # tf = 2 beats tf = 1


def test_bm25_saturation_and_length(impl):
    docs = [["x"] * 1 + ["pad"] * 9, ["x"] * 10, ["x"] + ["pad"], ["y"] * 10]
    s = impl.bm25_scores(["x"], docs)
    assert s[1] < 10 * s[0]                              # term frequency saturates
    assert s[2] > s[0]                                   # same tf, shorter doc scores higher
    s_b0 = impl.bm25_scores(["x"], docs, b=0.0)
    assert math.isclose(s_b0[2], s_b0[0])                # b = 0 turns length normalization off


def test_recall_and_mrr(impl):
    assert impl.recall_at_k([3, 1, 2, 0], {1, 0}, 2) == 0.5
    assert impl.recall_at_k([3, 1, 2, 0], {1, 0}, 4) == 1.0
    assert impl.recall_at_k([3], set(), 1) == 0.0
    mrr = impl.mean_reciprocal_rank([[3, 1, 2], [0, 2, 1], [5, 6, 7]], [{1}, {0}, {9}])
    assert math.isclose(mrr, (1 / 2 + 1 + 0) / 3)


def test_ndcg_matches_sklearn(impl):
    rng = np.random.default_rng(0)
    for _ in range(5):
        grades = rng.integers(0, 4, size=10)
        scores = rng.normal(size=10)
        ranking = list(np.argsort(-scores))
        for k in [3, 5, 10]:
            ours = impl.ndcg_at_k(ranking, {i: int(g) for i, g in enumerate(grades)}, k)
            assert math.isclose(ours, ndcg_score([grades], [scores], k=k))
    assert impl.ndcg_at_k([0, 1], {}, 2) == 0.0


def test_rrf(impl):
    bm25 = ["a", "b", "c"]
    dense = ["c", "a", "d"]
    fused = impl.reciprocal_rank_fusion([bm25, dense], k=60)
    assert fused == ["a", "c", "b", "d"]
    assert impl.reciprocal_rank_fusion([["x", "y"], ["y", "x"]]) == ["x", "y"]   # tie broken by id
