import numpy as np
import pytest
import torch


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 5))
    y = rng.integers(0, 3, size=8)
    params = {"W1": rng.normal(scale=0.5, size=(5, 7)), "b1": rng.normal(scale=0.1, size=7),
              "W2": rng.normal(scale=0.5, size=(7, 3)), "b2": rng.normal(scale=0.1, size=3)}
    return X, y, params


def test_linear_and_relu_backward(impl):
    rng = np.random.default_rng(1)
    X, W, b, dout = rng.normal(size=(4, 3)), rng.normal(size=(3, 2)), rng.normal(size=2), rng.normal(size=(4, 2))
    dX, dW, db = impl.linear_backward(dout, X, W)
    assert dX.shape == X.shape and dW.shape == W.shape and db.shape == b.shape
    f = lambda: float((impl.linear_forward(X, W, b) * dout).sum())
    assert impl.rel_error(dW, impl.numeric_grad(f, W)) < 1e-7
    assert impl.rel_error(dX, impl.numeric_grad(f, X)) < 1e-7
    x = np.array([-1.0, 0.5, 2.0])
    np.testing.assert_array_equal(impl.relu_backward(np.ones(3), x), [0, 1, 1])


def test_softmax_cross_entropy(impl, setup):
    X, y, _ = setup
    logits = np.random.default_rng(2).normal(size=(8, 3)) * 10
    loss, d = impl.softmax_cross_entropy(logits, y)
    t = torch.tensor(logits, requires_grad=True)
    tl = torch.nn.functional.cross_entropy(t, torch.tensor(y))
    tl.backward()
    assert np.isclose(loss, tl.item())
    np.testing.assert_allclose(d, t.grad.numpy(), atol=1e-12)
    big, _ = impl.softmax_cross_entropy(np.array([[1000.0, 0.0]]), np.array([1]))
    assert np.isfinite(big) and np.isclose(big, 1000.0)


def test_gradient_check(impl, setup):
    X, y, params = setup
    loss, grads = impl.mlp_loss_and_grads(params, X, y)
    for k in params:
        num = impl.numeric_grad(lambda: impl.mlp_loss_and_grads(params, X, y)[0], params[k])
        assert impl.rel_error(grads[k], num) < 1e-6, k


def test_against_torch_autograd(impl, setup):
    X, y, params = setup
    loss, grads = impl.mlp_loss_and_grads(params, X, y)
    tp = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
    logits = torch.relu(torch.tensor(X) @ tp["W1"] + tp["b1"]) @ tp["W2"] + tp["b2"]
    tl = torch.nn.functional.cross_entropy(logits, torch.tensor(y))
    tl.backward()
    assert np.isclose(loss, tl.item())
    for k in params:
        np.testing.assert_allclose(grads[k], tp[k].grad.numpy(), atol=1e-12)


def test_numeric_grad_restores_input(impl):
    x = np.array([1.0, 2.0, 3.0])
    g = impl.numeric_grad(lambda: float((x ** 2).sum()), x)
    np.testing.assert_allclose(g, [2, 4, 6], rtol=1e-6)
    np.testing.assert_array_equal(x, [1.0, 2.0, 3.0])
