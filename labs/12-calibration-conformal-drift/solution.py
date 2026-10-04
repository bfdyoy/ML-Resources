"""Lab 12: expected calibration error, split conformal prediction, and the population stability index.

Part A (CORE-11): ECE.   Part B (CORE-11): split conformal for classification and regression.   Part C (PROD-03): PSI.
"""
import numpy as np


# ---------- Part A ----------
def expected_calibration_error(y_true, proba, n_bins=10):
    """Binary ECE: split [0, 1] into n_bins equal-width bins by the predicted P(y=1) (the last bin includes 1.0),
    and return sum_b (n_b / N) * |mean(y in b) - mean(proba in b)| over non-empty bins."""
    y_true, proba = np.asarray(y_true, float), np.asarray(proba, float)
    bins = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - proba[m].mean())
    return ece


# ---------- Part B ----------
def conformal_quantile(scores, alpha):
    """The finite-sample-corrected quantile used by split conformal:
    the ceil((n + 1)(1 - alpha))-th smallest calibration score (np.inf if that rank exceeds n)."""
    s = np.sort(np.asarray(scores))
    r = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return np.inf if r > len(s) else s[r - 1]


def conformal_prediction_sets(proba_cal, y_cal, proba_test, alpha):
    """Split conformal classification with the score 1 - p(true class).
    Return a boolean (n_test, C) matrix: class c is in the set iff 1 - proba_test[:, c] <= qhat."""
    scores = 1 - proba_cal[np.arange(len(y_cal)), y_cal]
    qhat = conformal_quantile(scores, alpha)
    return (1 - proba_test) <= qhat


def conformal_interval(residuals_cal, pred_test, alpha):
    """Split conformal regression with the score |y - y_hat|: return (lower, upper) = pred -+ qhat."""
    q = conformal_quantile(np.abs(residuals_cal), alpha)
    return pred_test - q, pred_test + q


# ---------- Part C ----------
def psi(expected, actual, n_bins=10, eps=1e-6):
    """Population stability index of `actual` against the reference sample `expected`.
    Bin edges: the quantiles of `expected` at 0, 1/n_bins, ..., 1, with the outer edges replaced by -inf and +inf
    (so out-of-range values still land in a bin). Proportions are clipped below at eps.
        PSI = sum_b (a_b - e_b) * ln(a_b / e_b)
    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift."""
    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), eps, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))
