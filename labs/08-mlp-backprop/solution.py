"""Lab 08: vectorized backprop for a 2-layer MLP, verified with a numeric gradient check.

Shapes: X (n, d), W1 (d, h), b1 (h,), W2 (h, C), b2 (C,), integer labels y (n,).
Forward: z1 = X W1 + b1 -> a1 = relu(z1) -> logits = a1 W2 + b2 -> mean softmax cross-entropy.
"""
import numpy as np


def linear_forward(X, W, b):
    """X @ W + b."""
    return X @ W + b


def linear_backward(dout, X, W):
    """Given dL/dout (n, k) for out = X W + b, return (dX, dW, db).
    Shape check: dX like X, dW like W, db like b. Hint: each is a matrix product or a sum over the batch."""
    return dout @ W.T, X.T @ dout, dout.sum(axis=0)


def relu_forward(x):
    """max(0, x), elementwise."""
    return np.maximum(0, x)


def relu_backward(dout, x):
    """dL/dx for out = relu(x): the upstream gradient passes where x > 0, zero elsewhere."""
    return dout * (x > 0)


def softmax_cross_entropy(logits, y):
    """Mean cross-entropy of softmax(logits) against integer labels y. Return (loss, dlogits).

    Subgoals: 1. shift logits by the row max  2. log-softmax = shifted - log(sum(exp(shifted)))
              3. loss = -mean(log-softmax[range(n), y])  4. dlogits = (softmax - one_hot(y)) / n
    """
    n = len(y)
    s = logits - logits.max(axis=1, keepdims=True)
    log_p = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
    loss = -log_p[np.arange(n), y].mean()
    d = np.exp(log_p)
    d[np.arange(n), y] -= 1
    return loss, d / n


def mlp_loss_and_grads(params, X, y):
    """Forward and backward through the whole network. params is a dict with W1, b1, W2, b2.
    Return (loss, grads) where grads has the same keys and shapes as params."""
    z1 = linear_forward(X, params["W1"], params["b1"])
    a1 = relu_forward(z1)
    logits = linear_forward(a1, params["W2"], params["b2"])
    loss, dlogits = softmax_cross_entropy(logits, y)
    da1, dW2, db2 = linear_backward(dlogits, a1, params["W2"])
    dz1 = relu_backward(da1, z1)
    _, dW1, db1 = linear_backward(dz1, X, params["W1"])
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def numeric_grad(f, x, eps=1e-6):
    """Central-difference gradient of the scalar function f() with respect to the array x, which f reads.
    Perturb x IN PLACE one entry at a time ((f(x+eps) - f(x-eps)) / 2eps), and restore each entry afterwards."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f()
        x[i] = old - eps
        fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def rel_error(a, b):
    """max |a - b| / max(|a| + |b|, 1e-12), elementwise; the usual gradient-check statistic."""
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-12)))
