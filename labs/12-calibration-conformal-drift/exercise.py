# EXERCISE: generated from solution.py by scripts/make_lab_stubs.py.
# Replace each `raise NotImplementedError` with your code, then run the tests (see README.md).
"""Lab 12: expected calibration error, split conformal prediction, and the population stability index.

Part A (CORE-11): ECE.   Part B (CORE-11): split conformal for classification and regression.   Part C (PROD-03): PSI.
"""
import numpy as np


# ---------- Part A ----------
def expected_calibration_error(y_true, proba, n_bins=10):
    """Binary ECE: split [0, 1] into n_bins equal-width bins by the predicted P(y=1) (the last bin includes 1.0),
    and return sum_b (n_b / N) * |mean(y in b) - mean(proba in b)| over non-empty bins."""
    raise NotImplementedError("your code here")


# ---------- Part B ----------
def conformal_quantile(scores, alpha):
    """The finite-sample-corrected quantile used by split conformal:
    the ceil((n + 1)(1 - alpha))-th smallest calibration score (np.inf if that rank exceeds n)."""
    raise NotImplementedError("your code here")


def conformal_prediction_sets(proba_cal, y_cal, proba_test, alpha):
    """Split conformal classification with the score 1 - p(true class).
    Return a boolean (n_test, C) matrix: class c is in the set iff 1 - proba_test[:, c] <= qhat."""
    raise NotImplementedError("your code here")


def conformal_interval(residuals_cal, pred_test, alpha):
    """Split conformal regression with the score |y - y_hat|: return (lower, upper) = pred -+ qhat."""
    raise NotImplementedError("your code here")


# ---------- Part C ----------
def psi(expected, actual, n_bins=10, eps=1e-6):
    """Population stability index of `actual` against the reference sample `expected`.
    Bin edges: the quantiles of `expected` at 0, 1/n_bins, ..., 1, with the outer edges replaced by -inf and +inf
    (so out-of-range values still land in a bin). Proportions are clipped below at eps.
        PSI = sum_b (a_b - e_b) * ln(a_b / e_b)
    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift."""
    raise NotImplementedError("your code here")
