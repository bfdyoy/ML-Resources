import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    p = 1 / (1 + np.exp(-(2 * X[:, 0] - X[:, 1] - 0.5)))
    return X, (rng.random(400) < p).astype(int)


def test_sigmoid_is_stable(impl):
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        s = impl.sigmoid(z)
    np.testing.assert_allclose(s, [0, 1 / (1 + np.e), 0.5, 1 / (1 + np.exp(-1)), 1], atol=1e-12)


def test_loss_and_grad(impl, data):
    X, y = data
    w, b, eps = np.array([0.4, -0.3]), 0.1, 1e-6
    loss, dw, db = impl.log_loss_and_grad(X, y, w, b)
    p = 1 / (1 + np.exp(-(X @ w + b)))
    assert np.isclose(loss, log_loss(y, p))
    num = [(impl.log_loss_and_grad(X, y, w + eps * e, b)[0] - impl.log_loss_and_grad(X, y, w - eps * e, b)[0]) / (2 * eps)
           for e in np.eye(2)]
    np.testing.assert_allclose(dw, num, rtol=1e-5)
    l2loss, l2dw, _ = impl.log_loss_and_grad(X, y, w, b, l2=0.5)
    assert np.isclose(l2loss, loss + 0.25 * w @ w) and np.allclose(l2dw, dw + 0.5 * w)


def test_fit_matches_sklearn(impl, data):
    X, y = data
    w, b = impl.fit_logistic(X, y, lr=0.5, n_steps=3000, l2=0.01)
    ref = LogisticRegression(C=1 / (0.01 * len(y)), tol=1e-10, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(w, ref.coef_[0], atol=1e-3)
    np.testing.assert_allclose(b, ref.intercept_[0], atol=1e-3)


def test_confusion_and_prf(impl, data):
    X, y = data
    pred = (X[:, 0] > 0).astype(int)
    tn, fp, fn, tp = impl.confusion(y, pred)
    assert (tn, fp, fn, tp) == tuple(confusion_matrix(y, pred).ravel())
    p, r, f = impl.precision_recall_f1(y, pred)
    assert np.isclose(p, precision_score(y, pred)) and np.isclose(r, recall_score(y, pred)) and np.isclose(f, f1_score(y, pred))
    assert impl.precision_recall_f1([1, 0], [0, 0]) == (0.0, 0.0, 0.0)


def test_roc_auc_including_ties(impl, data):
    X, y = data
    s = X[:, 0] - 0.5 * X[:, 1]
    assert np.isclose(impl.roc_auc(y, s), roc_auc_score(y, s))
    coarse = np.round(s)                                          # many ties
    assert np.isclose(impl.roc_auc(y, coarse), roc_auc_score(y, coarse))
    assert np.isclose(impl.roc_auc([0, 1], [0.5, 0.5]), 0.5)


def test_best_threshold(impl, data):
    X, y = data
    proba = 1 / (1 + np.exp(-(2 * X[:, 0] - X[:, 1] - 0.5)))       # the true (calibrated) probabilities
    t, cost = impl.best_threshold(y, proba, cost_fp=1.0, cost_fn=4.0)
    grid = np.r_[np.unique(proba), np.inf]
    costs = [np.sum((proba >= g) & (y == 0)) + 4 * np.sum((proba < g) & (y == 1)) for g in grid]
    assert cost == min(costs) and t == grid[int(np.argmin(costs))]
    assert 0.1 < t < 0.35                                          # near the theory value c_fp / (c_fp + c_fn) = 0.2
