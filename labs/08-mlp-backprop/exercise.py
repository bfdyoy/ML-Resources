# EXERCISE: generated from solution.py by scripts/make_lab_stubs.py.
# Replace each `raise NotImplementedError` with your code, then run the tests (see README.md).
"""Lab 08: vectorized backprop for a 2-layer MLP, verified with a numeric gradient check.

Shapes: X (n, d), W1 (d, h), b1 (h,), W2 (h, C), b2 (C,), integer labels y (n,).
Forward: z1 = X W1 + b1 -> a1 = relu(z1) -> logits = a1 W2 + b2 -> mean softmax cross-entropy.
"""
import numpy as np


def linear_forward(X, W, b):
    """X @ W + b."""
    raise NotImplementedError("your code here")


def linear_backward(dout, X, W):
    """Given dL/dout (n, k) for out = X W + b, return (dX, dW, db).
    Shape check: dX like X, dW like W, db like b. Hint: each is a matrix product or a sum over the batch."""
    raise NotImplementedError("your code here")


def relu_forward(x):
    """max(0, x), elementwise."""
    raise NotImplementedError("your code here")


def relu_backward(dout, x):
    """dL/dx for out = relu(x): the upstream gradient passes where x > 0, zero elsewhere."""
    raise NotImplementedError("your code here")


def softmax_cross_entropy(logits, y):
    """Mean cross-entropy of softmax(logits) against integer labels y. Return (loss, dlogits).

    Subgoals: 1. shift logits by the row max  2. log-softmax = shifted - log(sum(exp(shifted)))
              3. loss = -mean(log-softmax[range(n), y])  4. dlogits = (softmax - one_hot(y)) / n
    """
    raise NotImplementedError("your code here")


def mlp_loss_and_grads(params, X, y):
    """Forward and backward through the whole network. params is a dict with W1, b1, W2, b2.
    Return (loss, grads) where grads has the same keys and shapes as params."""
    raise NotImplementedError("your code here")


def numeric_grad(f, x, eps=1e-6):
    """Central-difference gradient of the scalar function f() with respect to the array x, which f reads.
    Perturb x IN PLACE one entry at a time ((f(x+eps) - f(x-eps)) / 2eps), and restore each entry afterwards."""
    raise NotImplementedError("your code here")


def rel_error(a, b):
    """max |a - b| / max(|a| + |b|, 1e-12), elementwise; the usual gradient-check statistic."""
    raise NotImplementedError("your code here")
