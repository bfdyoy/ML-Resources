# EXERCISE: generated from solution.py by scripts/make_lab_stubs.py.
# Replace each `raise NotImplementedError` with your code, then run the tests (see README.md).
"""Lab 03: linear regression three ways (closed form, gradient descent, ridge).

Conventions: X has shape (n, d) WITHOUT a bias column; every fit returns (w, b) with w of shape (d,).
Prediction is X @ w + b. Loss is the mean squared error (1/n) * sum (X w + b - y)^2.
"""
import numpy as np


def fit_closed_form(X, y):
    """Ordinary least squares via np.linalg.lstsq on the design matrix [1, X].

    Subgoals: 1. prepend a column of ones  2. solve least squares  3. split theta into (b, w)
    Why lstsq and not inv(X^T X): it uses an SVD, so it stays accurate when X^T X is ill-conditioned.
    """
    raise NotImplementedError("your code here")


def mse_and_grad(X, y, w, b):
    """Return (loss, dw, db) for the mean squared error.

    Derivation hint: with residual r = X w + b - y, loss = mean(r^2), dloss/dw = (2/n) X^T r, dloss/db = (2/n) sum(r).
    """
    raise NotImplementedError("your code here")


def fit_gd(X, y, lr=0.1, n_steps=500):
    """Batch gradient descent from w = 0, b = 0. Return (w, b, losses) where losses[t] is the loss
    BEFORE update t (so len(losses) == n_steps and losses[0] is the loss of the zero model)."""
    raise NotImplementedError("your code here")


def fit_ridge(X, y, alpha):
    """Ridge regression minimizing ||X w + b - y||^2 + alpha ||w||^2 (the intercept is NOT penalized),
    the same objective as sklearn.linear_model.Ridge(alpha).

    Subgoals:
      1. Center X and y (this removes b from the problem)
      2. Solve (Xc^T Xc + alpha I) w = Xc^T yc with np.linalg.solve (not inv)
      3. Recover b = mean(y) - mean(X) @ w
    """
    raise NotImplementedError("your code here")
