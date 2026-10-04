# DL-01 notes: Neural Networks from Scratch & Backprop

[← Lesson DL-01](../../lessons/deep-learning/01-neural-networks-from-scratch.md) · [All notes](../README.md) · [← CORE-12 notes](../core-ml/12-hyperparameter-optimization.md) · Next: [DL-02 notes →](02-pytorch-fluency.md)

> **Reading time** ≈ 65 min. **You need:** [MATH-02 notes](../math/02-calculus-optimization.md) §A3–A6 (the chain rule, computational graphs, gradient checks), [CORE-03 notes](../core-ml/03-classification-and-metrics.md) §1 (logistic regression, softmax), and [MATH-03 notes](../math/03-probability-statistics.md) §B–C (likelihood, cross-entropy).

---

## Where we are

Logistic regression is a single layer: a linear score, then a squashing function, then cross-entropy. A neural network **stacks** such layers and learns the intermediate features itself, instead of you engineering them
(CORE-07). The only new mathematical ingredient is computing gradients through many layers. That's **backpropagation**, and by the end of this note you'll have written it twice:
once in matrix form and once as a tiny autograd engine.

---

## 1. The multilayer perceptron (MLP)

A 2-layer MLP for classification, on one input $x \in \mathbb{R}^d$:

```math
a_1 = W_1 x + b_1,\quad h = \phi(a_1),\quad z = W_2 h + b_2,\quad p = \operatorname{softmax}(z),\quad L = -\log p_y .
```

$\phi$ is a non-linearity applied element-wise, such as ReLU, $\phi(a) = \max(0, a)$. The hidden vector $h$ is a set of **learned features**, and the last layer is plain logistic (softmax) regression on those features.

### 1.1 Why the non-linearity is essential

Remove $\phi$ and the network collapses: $z = W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$. That's a single linear map. **Any stack of linear layers is just one linear layer**, so depth would buy nothing.

### 1.2 What ReLU networks compute

Each ReLU unit is a "hinge": zero on one side of a hyperplane, linear on the other. A sum of hinges is a **piecewise-linear function**. More units mean more pieces, so the network can approximate any continuous function on a bounded region
(the **universal approximation theorem**) given enough width.

**Why depth helps.** Composing layers *multiplies* the number of linear pieces: each layer can fold the regions produced by the previous one. The number of linear regions a deep network can create grows exponentially with depth,
but only polynomially with width. So deep, narrow networks represent some functions far more efficiently than shallow, wide ones. UDL Ch. 3–4 has beautiful figures of this folding.

---

## 2. Where the loss comes from

For $K$ classes, model the label as a **categorical** random variable with probabilities $p = \operatorname{softmax}(z)$. The likelihood of the observed label $y$ is $p_y$, so

```math
\text{NLL} = -\sum_{i}\log p_{y_i}(x_i) = \sum_i H\big(\text{onehot}(y_i),\ p(x_i)\big),
```

which is **cross-entropy** between the one-hot label and the prediction (MATH-03 §C2). **Minimizing cross-entropy is maximum likelihood.** For regression with Gaussian noise, the same argument gives MSE (MATH-03 §B2).
The output layer and the loss come as a matched pair: softmax with cross-entropy for classes, linear with MSE for real values, sigmoid with binary cross-entropy for independent yes/no labels.

### 2.1 The softmax + cross-entropy gradient

With $p_k = e^{z_k}/\sum_j e^{z_j}$, we have $\log p_y = z_y - \log\sum_j e^{z_j}$, so

```math
\frac{\partial L}{\partial z_k} = -\mathbb{1}[k = y] + \frac{e^{z_k}}{\sum_j e^{z_j}} = p_k - \mathbb{1}[k=y] \qquad\Longrightarrow\qquad \frac{\partial L}{\partial z} = p - \text{onehot}(y).
```

It's the same "prediction minus truth" as in linear and logistic regression. This clean form is why frameworks fuse the two (`nn.CrossEntropyLoss` takes raw logits).
It's also numerically safer: compute $\log\sum_j e^{z_j}$ as $m + \log\sum_j e^{z_j - m}$ with $m = \max_j z_j$ (the **log-sum-exp trick**).

---

## 3. Backpropagation in matrix form

Use the chain rule backwards (MATH-02 §A5), keeping the forward values. For one example, write $\delta_z = \partial L/\partial z$ and so on:

```math
\begin{aligned}
\delta_z &= p - \text{onehot}(y) &&\text{(softmax + CE)}\\
\frac{\partial L}{\partial W_2} &= \delta_z\, h^\top,\qquad \frac{\partial L}{\partial b_2} = \delta_z &&\text{(the output layer is linear)}\\
\delta_h &= W_2^\top \delta_z &&\text{(send the error back through } W_2\text{)}\\
\delta_{a_1} &= \delta_h \odot \phi'(a_1) &&\text{(ReLU: let the gradient through where } a_1 > 0\text{)}\\
\frac{\partial L}{\partial W_1} &= \delta_{a_1}\, x^\top,\qquad \frac{\partial L}{\partial b_1} = \delta_{a_1}
\end{aligned}
```

**The pattern repeats for every linear layer $a = Wx + b$:**

- the weight gradient is **(error at the output) × (input)ᵀ**, an outer product;
- the error passed to the layer below is $W^\top$ **× (error at the output)**.

**Shape check:** $\partial L/\partial W_2$ must have the shape of $W_2$, $(K\times H)$, and $\delta_z h^\top$ is $(K\times1)(1\times H)$. ✔ For a mini-batch, stack the examples as rows and the outer products become a matrix product,
$\partial L/\partial W = \Delta^\top X / B$ when the loss is averaged over the batch.

### 3.1 Why backprop is cheap

Each layer's backward step is two matrix multiplies of the same size as the forward one (one for $\partial W$, one for $\delta$ to pass down). So **a backward pass costs about 2× the forward pass**, and the
full gradient costs about 3× a forward pass, *whatever the number of parameters*. The price is **memory**: every forward activation must be stored until the backward pass uses it. That's why activation memory dominates training (DL-07).

---

## 4. Autograd: backprop as a general algorithm

Instead of deriving each network's gradients by hand, record the forward computation as a **graph of primitive operations**, each of which knows its local derivative. Then:

1. **Topologically sort** the graph (every node after its inputs).
2. Set $\partial L/\partial L = 1$.
3. Visit the nodes in **reverse** order. Each node adds (local derivative × its own gradient) into each of its inputs' gradients.

**Why `+=` instead of `=`.** If a value feeds into *two* later operations (as $x$ does in $x\cdot x$, or as a weight shared across time steps), its total gradient is the **sum** over the paths (MATH-02 §A3, "add across paths").
Assigning would overwrite one path's contribution with another's. This is the same reason PyTorch accumulates into `.grad`, and why you must call `zero_grad()` between steps (DL-02).

Below is a complete micrograd-style engine. It's about 40 lines, and it trains a small network.

```python
import math, random

class Value:
    """A scalar that remembers how it was computed, so it can backprop."""
    def __init__(self, data, parents=(), op=""):
        self.data, self.grad = data, 0.0
        self._parents, self._op, self._backward = parents, op, lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad                  # d(a+b)/da = 1
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad     # d(ab)/da = b
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    __radd__ = __add__; __rmul__ = __mul__

    def backward(self):
        order, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents: build(p)
                order.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

# Why += matters: x is used twice in x*x, so dL/dx = 2x
x = Value(3.0); y = x * x; y.backward()
print("d(x*x)/dx at 3 =", x.grad)

# A tiny MLP: 2 -> 8 (tanh) -> 1, trained with squared error on XOR-like data
random.seed(0)
W1 = [[Value(random.uniform(-1, 1)) for _ in range(2)] for _ in range(8)]
b1 = [Value(0.0) for _ in range(8)]
W2 = [Value(random.uniform(-1, 1)) for _ in range(8)]
params = [w for row in W1 for w in row] + b1 + W2
def forward(x1, x2):
    h = [(W1[j][0] * x1 + W1[j][1] * x2 + b1[j]).tanh() for j in range(8)]
    return sum((W2[j] * h[j] for j in range(8)), Value(0.0)).tanh()
data = [((0, 0), -1), ((0, 1), 1), ((1, 0), 1), ((1, 1), -1)]
for step in range(300):
    errors = [forward(*x) - t for x, t in data]
    loss = sum((e * e for e in errors), Value(0.0))
    for p in params: p.grad = 0.0           # zero the grads, or they accumulate across steps
    loss.backward()
    for p in params: p.data -= 0.05 * p.grad
print("final loss:", round(loss.data, 4), " predictions:", [round(forward(*x).data, 2) for x, _ in data])
```

---

## 5. Checking your gradients

If a hand-written gradient disagrees with a finite-difference check by 30%, here's the drill:

1. **Check one layer at a time.** Gradient-check the last layer alone, then add layers back. The first layer that disagrees contains the bug.
2. **Use float64 and a tiny input** (2 examples, 3 features). Turn off randomness (dropout) and anything stateful (BatchNorm in training mode).
3. **Compare the relative error** $\frac{\lvert g_a - g_n\rvert}{\max(\lvert g_a\rvert, \lvert g_n\rvert)}$. Below $10^{-7}$ is good. A value around 0.3 is a real bug, not rounding.
4. **The usual suspects:** a missing transpose, a forgotten $1/B$ when averaging, `=` instead of `+=` for a reused variable, the ReLU derivative evaluated on the *output* instead of the pre-activation (harmless for ReLU, wrong for tanh or sigmoid), and kinks
   (ReLU at exactly 0) right where the finite difference straddles them.

```python
# Manual matrix backprop for a 2-layer MLP, checked against PyTorch autograd
import numpy as np, torch
rng = np.random.default_rng(0)
B, d, H, K = 4, 3, 5, 3
X = rng.normal(size=(B, d)); y = rng.integers(0, K, B)
W1, b1 = rng.normal(size=(H, d)), np.zeros(H)
W2, b2 = rng.normal(size=(K, H)), np.zeros(K)

a1 = X @ W1.T + b1; h = np.maximum(a1, 0); z = h @ W2.T + b2
z_shift = z - z.max(1, keepdims=True)                      # log-sum-exp trick
p = np.exp(z_shift) / np.exp(z_shift).sum(1, keepdims=True)
L = -np.log(p[np.arange(B), y]).mean()

dz = p.copy(); dz[np.arange(B), y] -= 1; dz /= B           # (p - onehot) / B
dW2 = dz.T @ h; db2 = dz.sum(0)
dh = dz @ W2
da1 = dh * (a1 > 0)
dW1 = da1.T @ X; db1 = da1.sum(0)

t = lambda a: torch.tensor(a, dtype=torch.float64, requires_grad=True)
tW1, tb1, tW2, tb2 = t(W1), t(b1), t(W2), t(b2)
logits = torch.relu(torch.tensor(X) @ tW1.T + tb1) @ tW2.T + tb2
tL = torch.nn.functional.cross_entropy(logits, torch.tensor(y))
tL.backward()
print(f"loss numpy {L:.6f}  torch {tL.item():.6f}")
for name, mine, ref in [("W1", dW1, tW1.grad), ("b1", db1, tb1.grad), ("W2", dW2, tW2.grad), ("b2", db2, tb2.grad)]:
    print(f"d{name}: max abs diff = {np.abs(mine - ref.numpy()).max():.2e}")

# Without the non-linearity, two layers equal one linear map
print("linear collapse holds:", np.allclose((X @ W1.T + b1) @ W2.T + b2, X @ (W2 @ W1).T + (W2 @ b1 + b2)))
```

---

## Pitfalls & misconceptions

- **Softmax, then `log`, then NLL, computed separately.** It's numerically unstable. Use the fused cross-entropy on logits.
- **"More layers = more power" without non-linearities.** Without them, the stack is still linear.
- **Forgetting to zero the gradients.** They accumulate across steps, by design.
- **Hand-derived gradients without a numeric check.**
- **Universal approximation ≠ learnability.** The theorem says a network *exists*. It says nothing about whether GD will find it, or how much data that takes.

## Cheat sheet

| Item | Formula |
|---|---|
| Layer | $h = \phi(Wx + b)$ |
| Softmax + CE gradient | $\partial L/\partial z = p - \text{onehot}(y)$ |
| Linear-layer backward | $\partial L/\partial W = \delta\, x^\top$; pass down $W^\top\delta$ |
| ReLU backward | $\delta \odot \mathbb{1}[a > 0]$ |
| Cost | backward ≈ 2× forward; memory = every stored activation |
| Log-sum-exp | $\log\sum e^{z_j} = m + \log\sum e^{z_j - m}$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why does an MLP with no activations collapse to a linear model?</summary>

A composition of affine maps is affine: $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1+b_2)$. The demo checks this numerically.
</details>

<details>
<summary>2. The backward pass for L = (w·x + b − y)².</summary>

Forward: $u = w^\top x$, $v = u + b$, $e = v - y$, $L = e^2$. Backward: $\partial L/\partial e = 2e$; $\partial L/\partial v = 2e$; $\partial L/\partial b = 2e$; $\partial L/\partial u = 2e$; $\partial L/\partial w = 2e\,x$. This is the CORE-02 gradient for a single example.
</details>

<details>
<summary>3. Why += in micrograd?</summary>

A value used in several places receives gradient from each path, and the multivariate chain rule *sums* them. Assigning would keep only the last path. Example: $x\cdot x$ needs $2x$, and assignment would give $x$.
</details>

<details>
<summary>4. Cross-entropy = maximum likelihood under a categorical model.</summary>

The likelihood of the labels is $\prod_i p_{y_i}(x_i)$, so the NLL is $-\sum_i\log p_{y_i}(x_i)$, which is exactly the sum of cross-entropies between the one-hot labels and the predicted distributions. Same objective, same minimizer.
</details>

<details>
<summary>5. The cost of backprop vs the forward pass.</summary>

About 2× the forward pass (two matmuls per layer: one for the weight gradient, one to propagate the error), so about 3× in total, independent of the number of parameters. It's cheap because reverse mode reuses the stored intermediate values and computes all the parameter gradients in one sweep. The cost is memory for the activations.
</details>

<details>
<summary>6. Hand-written backprop differs from finite differences by 30%.</summary>

Gradient-check layer by layer, in float64, on a tiny input, with randomness off. Look for transposes, a missing $1/B$, `=` vs `+=` for reused values, an activation derivative computed on the wrong tensor, or a check that sits exactly on the ReLU kink (§5).
</details>

## Where this leads

Next: [DL-02 notes](02-pytorch-fluency.md). You've built autograd by hand. PyTorch is that same engine, industrial-strength, running on tensors and GPUs. DL-02 makes you fluent in it,
so the remaining lessons can focus on ideas rather than plumbing.
