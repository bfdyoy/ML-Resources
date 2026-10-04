"""Lab 05: cross-validation splitters, a tree split, and leak-free target encoding.

Part A (CORE-04): k-fold and group k-fold index generators.
Part B (CORE-05): Gini impurity and the best threshold split on one feature.
Part C (CORE-07): out-of-fold target encoding, where no row's encoding ever sees its own label.
"""
import numpy as np


# ---------- Part A ----------
def kfold_indices(n, k, seed=0):
    """Shuffle 0..n-1 with np.random.default_rng(seed), cut it into k folds whose sizes differ by at most 1,
    and return a list of k (train_idx, val_idx) pairs. Every index is in exactly one val fold."""
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, k)
    return [(np.concatenate(folds[:i] + folds[i + 1:]), folds[i]) for i in range(k)]


def group_kfold_indices(groups, k):
    """Like k-fold, but all rows of a group land in the same val fold (no group on both sides).
    Balance fold sizes greedily: take groups from largest to smallest (ties: by group label),
    and put each one into the fold with the fewest rows so far (ties: lowest fold number).
    Return a list of k (train_idx, val_idx) pairs with sorted index arrays."""
    groups = np.asarray(groups)
    labels, counts = np.unique(groups, return_counts=True)
    order = sorted(range(len(labels)), key=lambda i: (-counts[i], labels[i]))
    fold_of, sizes = {}, np.zeros(k, int)
    for i in order:
        f = int(np.argmin(sizes))
        fold_of[labels[i]] = f
        sizes[f] += counts[i]
    assign = np.array([fold_of[g] for g in groups])
    idx = np.arange(len(groups))
    return [(idx[assign != f], idx[assign == f]) for f in range(k)]


# ---------- Part B ----------
def gini(y):
    """Gini impurity 1 - sum_c p_c^2 of an integer label array (0.0 for an empty array)."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1.0 - np.sum(p ** 2)


def best_split(x, y):
    """Best threshold for the rule `x <= t` on one numeric feature, minimizing the size-weighted Gini
    of the two children. Candidates are midpoints between consecutive distinct sorted values of x.
    Return (t, weighted_gini). If x has a single distinct value, return (None, gini(y)).
    Ties: the smallest t wins."""
    xs = np.unique(x)
    best = (None, gini(y))
    if len(xs) == 1:
        return best
    best = (None, np.inf)
    n = len(y)
    for t in (xs[:-1] + xs[1:]) / 2:
        left = x <= t
        score = (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / n
        if score < best[1]:
            best = (t, score)
    return best


# ---------- Part C ----------
def oof_target_encode(categories, y, k=5, seed=0, smoothing=10.0):
    """Out-of-fold, smoothed target encoding. For each fold of kfold_indices(n, k, seed):
    fit on the train part, then encode the val part with
        enc(c) = (sum_y(c) + smoothing * prior) / (count(c) + smoothing),   prior = mean of y on the train part
    Categories unseen in the train part get the prior. Return a float array of length n.

    Why: encoding a row with a statistic that includes its own label leaks the target, and high-cardinality
    categories turn that leak into a near-perfect (and fake) training score.
    """
    categories, y = np.asarray(categories), np.asarray(y, float)
    out = np.empty(len(y))
    for tr, va in kfold_indices(len(y), k, seed):
        prior = y[tr].mean()
        cats, inv = np.unique(categories[tr], return_inverse=True)
        sums = np.bincount(inv, weights=y[tr])
        counts = np.bincount(inv)
        table = dict(zip(cats, (sums + smoothing * prior) / (counts + smoothing)))
        out[va] = [table.get(c, prior) for c in categories[va]]
    return out
