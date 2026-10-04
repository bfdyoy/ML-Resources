import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier


def test_kfold_partition(impl):
    folds = impl.kfold_indices(23, 5, seed=1)
    assert len(folds) == 5
    vals = np.concatenate([v for _, v in folds])
    assert sorted(vals.tolist()) == list(range(23))
    assert {len(v) for _, v in folds} <= {4, 5}
    for tr, va in folds:
        assert set(tr).isdisjoint(va) and len(tr) + len(va) == 23
    assert not np.array_equal(folds[0][1], np.arange(len(folds[0][1])))    # shuffled
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, impl.kfold_indices(23, 5, seed=1)))


def test_group_kfold(impl):
    groups = np.array(list("aaaabbbccdddddeef"))
    folds = impl.group_kfold_indices(groups, 3)
    vals = np.concatenate([v for _, v in folds])
    assert sorted(vals.tolist()) == list(range(len(groups)))
    for tr, va in folds:
        assert set(groups[tr]).isdisjoint(groups[va])
    sizes = sorted(len(v) for _, v in folds)
    assert sizes == [5, 6, 6]                                     # d=5 | a=4,f=1 -> ... greedy balance
    assert all(np.all(np.diff(v) > 0) for _, v in folds)


def test_gini(impl):
    assert impl.gini(np.array([0, 0, 0])) == 0.0
    assert np.isclose(impl.gini(np.array([0, 1, 0, 1])), 0.5)
    assert np.isclose(impl.gini(np.array([0, 1, 2])), 2 / 3)
    assert impl.gini(np.array([], dtype=int)) == 0.0


def test_best_split_matches_a_depth1_tree(impl):
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.normal(size=60).round(1)
        y = (x + rng.normal(scale=0.7, size=60) > 0.2).astype(int)
        t, score = impl.best_split(x, y)
        stump = DecisionTreeClassifier(max_depth=1).fit(x[:, None], y)
        tree = stump.tree_
        n_l, n_r = tree.n_node_samples[1], tree.n_node_samples[2]
        ref = (n_l * tree.impurity[1] + n_r * tree.impurity[2]) / 60
        assert np.isclose(score, ref)
    assert impl.best_split(np.array([1.0, 1.0]), np.array([0, 1])) == (None, 0.5)
    t, s = impl.best_split(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1]))
    assert t == 2.5 and s == 0.0


def test_oof_target_encoding_never_sees_own_label(impl):
    rng = np.random.default_rng(0)
    cats = rng.integers(0, 30, size=300)
    y = rng.integers(0, 2, size=300)
    enc = impl.oof_target_encode(cats, y, k=5, seed=0, smoothing=5.0)
    assert enc.shape == (300,) and np.all((enc > 0) & (enc < 1))
    for i in [0, 17, 123, 299]:                                   # flipping row i's label must not change row i's encoding
        y2 = y.copy()
        y2[i] = 1 - y2[i]
        assert np.isclose(impl.oof_target_encode(cats, y2, k=5, seed=0, smoothing=5.0)[i], enc[i])


def test_oof_target_encoding_values(impl):
    cats = np.array(["a", "a", "b", "b", "c", "c"])
    y = np.array([1, 1, 0, 0, 1, 0])
    enc = impl.oof_target_encode(cats, y, k=2, seed=0, smoothing=0.0)
    for tr, va in impl.kfold_indices(6, 2, seed=0):
        for i in va:
            same = [j for j in tr if cats[j] == cats[i]]
            expect = y[same].mean() if same else y[tr].mean()
            assert np.isclose(enc[i], expect)
