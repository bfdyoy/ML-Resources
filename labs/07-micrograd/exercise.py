# EXERCISE: generated from solution.py by scripts/make_lab_stubs.py.
# Replace each `raise NotImplementedError` with your code, then run the tests (see README.md).
"""Lab 07: a scalar autograd engine (after Karpathy's micrograd).

Each Value stores `data`, `grad`, its parents `_prev`, and a `_backward` closure that pushes
out.grad into the parents' .grad using the LOCAL derivative (chain rule). `backward()` runs the
closures in reverse topological order. Gradients ACCUMULATE (+=), because a value can feed several nodes.
"""
import math


class Value:
    def __init__(self, data, _prev=(), _op=""):
        """(given)"""
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_prev)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        """(given)"""
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    def __add__(self, other):
        """(given) A worked example of the pattern every op follows: compute out, then define how out.grad flows back.
        out = self + other. Local derivatives: 1 and 1."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        """out = self * other. Local derivatives: other.data and self.data."""
        raise NotImplementedError("your code here")

    def __pow__(self, k):
        """out = self ** k for a constant int/float k. Local derivative: k * x^(k-1)."""
        raise NotImplementedError("your code here")

    def exp(self):
        """out = e^x. Local derivative: e^x (reuse out.data)."""
        raise NotImplementedError("your code here")

    def log(self):
        """out = ln x. Local derivative: 1 / x."""
        raise NotImplementedError("your code here")

    def tanh(self):
        """out = tanh x. Local derivative: 1 - tanh(x)^2."""
        raise NotImplementedError("your code here")

    def relu(self):
        """out = max(0, x). Local derivative: 1 if x > 0 else 0."""
        raise NotImplementedError("your code here")

    def backward(self):
        """Set self.grad = 1, then call every node's _backward in reverse topological order.

        Subgoals: 1. build a topological order with a DFS over _prev (visit parents before appending the node)
                  2. seed self.grad = 1.0   3. run _backward over reversed(order)
        """
        raise NotImplementedError("your code here")

    # Derived operations, built from the ones above (given)
    def __neg__(self):
        """(given)"""
        return self * -1

    def __sub__(self, other):
        """(given)"""
        return self + (-other)

    def __truediv__(self, other):
        """(given)"""
        return self * (other if isinstance(other, Value) else Value(other)) ** -1

    def __radd__(self, other):
        """(given)"""
        return self + other

    def __rsub__(self, other):
        """(given)"""
        return Value(other) - self

    def __rmul__(self, other):
        """(given)"""
        return self * other

    def __rtruediv__(self, other):
        """(given)"""
        return Value(other) / self
