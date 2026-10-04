import math

import pytest
import torch


def test_add_mul_backward(impl):
    V = impl.Value
    a, b = V(2.0), V(-3.0)
    c = a * b + a                         # dc/da = b + 1 = -2, dc/db = a = 2
    c.backward()
    assert c.data == -4.0 and a.grad == -2.0 and b.grad == 2.0


def test_gradients_accumulate_when_reused(impl):
    a = impl.Value(3.0)
    b = a + a                              # db/da = 2, not 1
    b.backward()
    assert a.grad == 2.0
    a = impl.Value(3.0)
    (a * a * a).backward()                 # 3 a^2 = 27
    assert math.isclose(a.grad, 27.0)


def test_against_torch(impl):
    V = impl.Value
    x1, x2, w1, w2, b = V(0.7), V(-1.3), V(-2.0), V(0.4), V(0.9)
    n = x1 * w1 + x2 * w2 + b
    out = (n.tanh() * 2 + n.exp() / 3 - (n ** 2).relu() + (x1 * x1 + 1).log() - 1 / (x2 - 2)) * 1.5
    out.backward()

    t = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
         for k, v in dict(x1=0.7, x2=-1.3, w1=-2.0, w2=0.4, b=0.9).items()}
    tn = t["x1"] * t["w1"] + t["x2"] * t["w2"] + t["b"]
    tout = (tn.tanh() * 2 + tn.exp() / 3 - (tn ** 2).relu() + (t["x1"] * t["x1"] + 1).log() - 1 / (t["x2"] - 2)) * 1.5
    tout.backward()
    assert math.isclose(out.data, tout.item(), rel_tol=1e-12)
    for name, v in dict(x1=x1, x2=x2, w1=w1, w2=w2, b=b).items():
        assert math.isclose(v.grad, t[name].grad.item(), rel_tol=1e-9, abs_tol=1e-12), name


def test_relu_kink(impl):
    a = impl.Value(-1.0)
    a.relu().backward()
    assert a.grad == 0.0


def test_train_a_neuron(impl):
    """Tiny end-to-end check: gradient descent with your engine fits y = 2x - 1."""
    V = impl.Value
    w, b = V(0.0), V(0.0)
    xs, ys = [-1.0, 0.0, 1.0, 2.0], [-3.0, -1.0, 1.0, 3.0]
    for _ in range(200):
        w.grad = b.grad = 0.0
        loss = sum(((w * x + b) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)
        loss.backward()
        w.data -= 0.1 * w.grad
        b.data -= 0.1 * b.grad
    assert abs(w.data - 2) < 1e-3 and abs(b.data + 1) < 1e-3
