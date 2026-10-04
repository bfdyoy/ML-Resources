import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([1.5, -2.0, 0.5]) + 3.0 + rng.normal(scale=0.1, size=200)
    return X, y


def test_closed_form_matches_sklearn(impl, data):
    X, y = data
    w, b = impl.fit_closed_form(X, y)
    ref = LinearRegression().fit(X, y)
    np.testing.assert_allclose(w, ref.coef_, atol=1e-8)
    np.testing.assert_allclose(b, ref.intercept_, atol=1e-8)


def test_gradient_matches_finite_differences(impl, data):
    X, y = data
    w, b, eps = np.array([0.3, -0.1, 0.2]), 0.5, 1e-6
    loss, dw, db = impl.mse_and_grad(X, y, w, b)
    assert np.isclose(loss, np.mean((X @ w + b - y) ** 2))
    num = [(impl.mse_and_grad(X, y, w + eps * e, b)[0] - impl.mse_and_grad(X, y, w - eps * e, b)[0]) / (2 * eps)
           for e in np.eye(3)]
    np.testing.assert_allclose(dw, num, rtol=1e-6)
    num_b = (impl.mse_and_grad(X, y, w, b + eps)[0] - impl.mse_and_grad(X, y, w, b - eps)[0]) / (2 * eps)
    np.testing.assert_allclose(db, num_b, rtol=1e-6)


def test_gd_converges_to_closed_form(impl, data):
    X, y = data
    w, b, losses = impl.fit_gd(X, y, lr=0.1, n_steps=500)
    w0, b0 = impl.fit_closed_form(X, y)
    np.testing.assert_allclose(w, w0, atol=1e-6)
    np.testing.assert_allclose(b, b0, atol=1e-6)
    assert len(losses) == 500 and np.isclose(losses[0], np.mean(y ** 2))
    assert np.all(np.diff(losses) <= 1e-12)                       # monotone for a small enough step


def test_gd_diverges_with_too_large_a_step(impl, data):
    X, y = data
    _, _, losses = impl.fit_gd(X, y, lr=1.5, n_steps=30)
    assert losses[-1] > losses[0]


def test_ridge_matches_sklearn_and_shrinks(impl, data):
    X, y = data
    for alpha in [0.0, 1.0, 100.0]:
        w, b = impl.fit_ridge(X, y, alpha)
        ref = Ridge(alpha=alpha).fit(X, y)
        np.testing.assert_allclose(w, ref.coef_, atol=1e-8)
        np.testing.assert_allclose(b, ref.intercept_, atol=1e-8)
    norms = [np.linalg.norm(impl.fit_ridge(X, y, a)[0]) for a in [0.0, 10.0, 1000.0]]
    assert norms[0] > norms[1] > norms[2]
