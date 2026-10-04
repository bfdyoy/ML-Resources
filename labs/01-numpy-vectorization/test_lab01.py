import numpy as np
import pytest

rng = np.random.default_rng(0)


def test_pairwise_sq_dists(impl):
    X, Y = rng.normal(size=(7, 3)), rng.normal(size=(5, 3))
    slow = np.array([[((x - y) ** 2).sum() for y in Y] for x in X])
    D = impl.pairwise_sq_dists(X, Y)
    assert D.shape == (7, 5)
    np.testing.assert_allclose(D, slow, atol=1e-10)
    assert (impl.pairwise_sq_dists(X, X).diagonal() >= 0).all()


def test_softmax_is_stable_and_normalized(impl):
    z = np.array([[1000.0, 1001.0, 1002.0], [-5.0, 0.0, 5.0]])
    p = impl.softmax(z)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(p[0], impl.softmax(np.array([0.0, 1.0, 2.0])))
    np.testing.assert_allclose(impl.softmax(z, axis=0).sum(axis=0), 1.0)


def test_one_hot(impl):
    out = impl.one_hot(np.array([2, 0, 1, 2]), 4)
    assert out.shape == (4, 4)
    np.testing.assert_array_equal(out.argmax(1), [2, 0, 1, 2])
    np.testing.assert_array_equal(out.sum(1), 1)


def test_moving_average(impl):
    x = rng.normal(size=50)
    slow = np.array([x[i:i + 5].mean() for i in range(46)])
    np.testing.assert_allclose(impl.moving_average(x, 5), slow)
    np.testing.assert_allclose(impl.moving_average(np.arange(4), 4), [1.5])


def test_standardize(impl):
    X = np.c_[rng.normal(3, 2, size=100), np.full(100, 7.0)]
    Z = impl.standardize(X)
    np.testing.assert_allclose(Z[:, 0].mean(), 0, atol=1e-12)
    np.testing.assert_allclose(Z[:, 0].std(), 1)
    np.testing.assert_array_equal(Z[:, 1], 0)


def test_batched_quadratic_form(impl):
    X, A = rng.normal(size=(6, 4)), rng.normal(size=(4, 4))
    np.testing.assert_allclose(impl.batched_quadratic_form(X, A), [x @ A @ x for x in X])


def test_knn_predict(impl):
    X_train = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(impl.knn_predict(X_train, y_train, np.array([[0.05], [5.05], [2.4]]), 3), [0, 1, 0])
    # tie (one vote each) goes to the smallest label
    np.testing.assert_array_equal(impl.knn_predict(np.array([[0.0], [1.0]]), np.array([1, 0]), np.array([[0.5]]), 2), [0])
