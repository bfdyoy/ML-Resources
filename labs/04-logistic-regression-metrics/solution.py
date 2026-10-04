"""Lab 04: logistic regression from scratch, and classification metrics by hand.

Labels are 0/1 integer arrays. Probabilities are P(y = 1). Scores can be any real numbers (higher = more positive).
"""
import numpy as np
from scipy.stats import rankdata


def sigmoid(z):
    """Numerically stable logistic function (no overflow or invalid-value warnings for z = +-1000; harmless underflow to 0 is fine).
    Hint: for z < 0 use exp(z) / (1 + exp(z)); np.where evaluates both branches, so clip or split by mask."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss_and_grad(X, y, w, b, l2=0.0):
    """Mean binary cross-entropy plus (l2 / 2) * ||w||^2. Return (loss, dw, db).

    Derivation hint: with p = sigmoid(X w + b), the gradient of the mean BCE is X^T (p - y) / n and mean(p - y).
    Compute the loss stably from the logits z: BCE = logaddexp(0, z) - y * z.
    """
    z = X @ w + b
    p = sigmoid(z)
    n = len(y)
    loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * w @ w
    return loss, X.T @ (p - y) / n + l2 * w, np.mean(p - y)


def fit_logistic(X, y, lr=0.5, n_steps=2000, l2=0.0):
    """Gradient descent from zeros. Return (w, b)."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(n_steps):
        _, dw, db = log_loss_and_grad(X, y, w, b, l2)
        w, b = w - lr * dw, b - lr * db
    return w, b


def confusion(y_true, y_pred):
    """Return (tn, fp, fn, tp) as ints."""
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    return (int(np.sum(~y_true & ~y_pred)), int(np.sum(~y_true & y_pred)),
            int(np.sum(y_true & ~y_pred)), int(np.sum(y_true & y_pred)))


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1). Any 0/0 is defined as 0.0."""
    _, fp, fn, tp = confusion(y_true, y_pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def roc_auc(y_true, scores):
    """ROC-AUC = P(score of a random positive > score of a random negative), ties counting 1/2.

    Subgoals (the Mann-Whitney U route, no thresholds needed):
      1. Rank all scores, giving tied scores their average rank (scipy.stats.rankdata does this)
      2. Sum the ranks of the positives, subtract n_pos (n_pos + 1) / 2
      3. Divide by n_pos * n_neg
    """
    y_true = np.asarray(y_true, bool)
    ranks = rankdata(scores)
    n_pos, n_neg = y_true.sum(), (~y_true).sum()
    return (ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_threshold(y_true, proba, cost_fp, cost_fn):
    """Choose t minimizing total cost when predicting positive iff proba >= t.
    Candidates: every distinct value in proba, plus +inf (predict nothing positive).
    Return (t, cost); on ties return the smallest t."""
    y_true = np.asarray(y_true, bool)
    best = (np.inf, np.inf)
    for t in np.r_[np.unique(proba), np.inf]:
        _, fp, fn, _ = confusion(y_true, proba >= t)
        cost = cost_fp * fp + cost_fn * fn
        if cost < best[1]:
            best = (t, cost)
    return best
