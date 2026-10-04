# EXERCISE: generated from solution.py by scripts/make_lab_stubs.py.
# Replace each `raise NotImplementedError` with your code, then run the tests (see README.md).
"""Lab 04: logistic regression from scratch, and classification metrics by hand.

Labels are 0/1 integer arrays. Probabilities are P(y = 1). Scores can be any real numbers (higher = more positive).
"""
import numpy as np
from scipy.stats import rankdata


def sigmoid(z):
    """Numerically stable logistic function (no overflow or invalid-value warnings for z = +-1000; harmless underflow to 0 is fine).
    Hint: for z < 0 use exp(z) / (1 + exp(z)); np.where evaluates both branches, so clip or split by mask."""
    raise NotImplementedError("your code here")


def log_loss_and_grad(X, y, w, b, l2=0.0):
    """Mean binary cross-entropy plus (l2 / 2) * ||w||^2. Return (loss, dw, db).

    Derivation hint: with p = sigmoid(X w + b), the gradient of the mean BCE is X^T (p - y) / n and mean(p - y).
    Compute the loss stably from the logits z: BCE = logaddexp(0, z) - y * z.
    """
    raise NotImplementedError("your code here")


def fit_logistic(X, y, lr=0.5, n_steps=2000, l2=0.0):
    """Gradient descent from zeros. Return (w, b)."""
    raise NotImplementedError("your code here")


def confusion(y_true, y_pred):
    """Return (tn, fp, fn, tp) as ints."""
    raise NotImplementedError("your code here")


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1). Any 0/0 is defined as 0.0."""
    raise NotImplementedError("your code here")


def roc_auc(y_true, scores):
    """ROC-AUC = P(score of a random positive > score of a random negative), ties counting 1/2.

    Subgoals (the Mann-Whitney U route, no thresholds needed):
      1. Rank all scores, giving tied scores their average rank (scipy.stats.rankdata does this)
      2. Sum the ranks of the positives, subtract n_pos (n_pos + 1) / 2
      3. Divide by n_pos * n_neg
    """
    raise NotImplementedError("your code here")


def best_threshold(y_true, proba, cost_fp, cost_fn):
    """Choose t minimizing total cost when predicting positive iff proba >= t.
    Candidates: every distinct value in proba, plus +inf (predict nothing positive).
    Return (t, cost); on ties return the smallest t."""
    raise NotImplementedError("your code here")
