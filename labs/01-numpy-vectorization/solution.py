"""Lab 01: NumPy vectorization.

Every function here must work WITHOUT a Python loop over rows: use broadcasting, ufuncs, reductions,
fancy indexing, `np.cumsum`, and `np.einsum`. The tests compare against slow loop versions.
"""
import numpy as np


def pairwise_sq_dists(X, Y):
    """Squared Euclidean distances D[i, j] = ||X[i] - Y[j]||^2, shape (n, m).

    Subgoals:
      1. Expand ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
      2. Row norms as (n, 1) and (1, m) arrays, the cross term as one matrix product
      3. Clip tiny negative values (floating-point round-off) to 0
    """
    x2 = (X ** 2).sum(axis=1)[:, None]
    y2 = (Y ** 2).sum(axis=1)[None, :]
    return np.maximum(x2 + y2 - 2.0 * X @ Y.T, 0.0)


def softmax(z, axis=-1):
    """Numerically stable softmax along `axis`.

    Subgoals:
      1. Subtract the max along `axis` (keepdims=True) so exp never overflows
      2. Exponentiate and divide by the sum along the same axis
    Hint: softmax(z) == softmax(z - c) for any constant c.
    """
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def one_hot(y, num_classes):
    """Integer labels (n,) -> one-hot matrix (n, num_classes) of floats.

    Hint: fancy indexing with np.arange(n) and y, or np.eye(num_classes)[y].
    """
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def moving_average(x, w):
    """Mean of each length-w window ('valid' mode): output length len(x) - w + 1.

    Subgoals:
      1. c = cumulative sum of x with a 0 prepended
      2. window sums are c[w:] - c[:-w]
    """
    c = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
    return (c[w:] - c[:-w]) / w


def standardize(X):
    """Column-wise z-scores (ddof=0). Constant columns become all zeros (no division by zero)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return (X - mu) / np.where(sd == 0, 1.0, sd)


def batched_quadratic_form(X, A):
    """q[i] = X[i] @ A @ X[i] for every row, shape (n,). Use np.einsum ('ij,jk,ik->i')."""
    return np.einsum("ij,jk,ik->i", X, A, X)


def knn_predict(X_train, y_train, X_test, k):
    """k-nearest-neighbour majority vote (Euclidean). Ties go to the smallest label.

    Subgoals:
      1. D = pairwise_sq_dists(X_test, X_train)
      2. Indices of the k smallest per row: np.argpartition(D, k - 1, axis=1)[:, :k]
      3. Their labels -> one-hot -> sum over the k neighbours -> argmax (argmax picks the first max = smallest label)
    """
    D = pairwise_sq_dists(X_test, X_train)
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    votes = one_hot(y_train[idx].ravel(), y_train.max() + 1).reshape(len(X_test), k, -1).sum(axis=1)
    return votes.argmax(axis=1)
