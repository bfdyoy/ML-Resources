# EXERCISE: generated from solution.py by scripts/make_lab_stubs.py.
# Replace each `raise NotImplementedError` with your code, then run the tests (see README.md).
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
    raise NotImplementedError("your code here")


def softmax(z, axis=-1):
    """Numerically stable softmax along `axis`.

    Subgoals:
      1. Subtract the max along `axis` (keepdims=True) so exp never overflows
      2. Exponentiate and divide by the sum along the same axis
    Hint: softmax(z) == softmax(z - c) for any constant c.
    """
    raise NotImplementedError("your code here")


def one_hot(y, num_classes):
    """Integer labels (n,) -> one-hot matrix (n, num_classes) of floats.

    Hint: fancy indexing with np.arange(n) and y, or np.eye(num_classes)[y].
    """
    raise NotImplementedError("your code here")


def moving_average(x, w):
    """Mean of each length-w window ('valid' mode): output length len(x) - w + 1.

    Subgoals:
      1. c = cumulative sum of x with a 0 prepended
      2. window sums are c[w:] - c[:-w]
    """
    raise NotImplementedError("your code here")


def standardize(X):
    """Column-wise z-scores (ddof=0). Constant columns become all zeros (no division by zero)."""
    raise NotImplementedError("your code here")


def batched_quadratic_form(X, A):
    """q[i] = X[i] @ A @ X[i] for every row, shape (n,). Use np.einsum ('ij,jk,ik->i')."""
    raise NotImplementedError("your code here")


def knn_predict(X_train, y_train, X_test, k):
    """k-nearest-neighbour majority vote (Euclidean). Ties go to the smallest label.

    Subgoals:
      1. D = pairwise_sq_dists(X_test, X_train)
      2. Indices of the k smallest per row: np.argpartition(D, k - 1, axis=1)[:, :k]
      3. Their labels -> one-hot -> sum over the k neighbours -> argmax (argmax picks the first max = smallest label)
    """
    raise NotImplementedError("your code here")
