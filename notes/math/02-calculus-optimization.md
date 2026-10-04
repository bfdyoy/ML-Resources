# MATH-02 notes: Calculus & Optimization for ML

[← Lesson MATH-02](../../lessons/math/02-calculus-optimization.md) · [All notes](../README.md) · [Notation](../notation.md) · [← MATH-01 notes](01-linear-algebra.md) · Next: [MATH-03 notes →](03-probability-statistics.md)

> **Reading time** ≈ 60 min. **You need:** [MATH-01 notes](01-linear-algebra.md) Blocks A–B (vectors, dot products, the normal equations).

---

## Where we are

Training a model means **choosing parameters that make a loss small**. Calculus tells you which way is "downhill" (the gradient),
the chain rule tells you how to compute that direction through many layers (backpropagation), and optimization theory tells you
how big a step to take and why training is sometimes slow. That is the whole engine underneath CORE-02, DL-01, and DL-03.

---

## Block A: Derivatives, gradients, and the chain rule

### A1. A derivative is the best linear approximation

Forget "slope of a tangent" for a moment. The useful definition is this: near a point $x$, a smooth function behaves like a straight line,

```math
f(x + h) \approx f(x) + f'(x)\, h \qquad \text{(for small } h\text{)}.
```

$f'(x)$ is the **exchange rate**: nudge the input by $h$, and the output moves by about $f'(x)\,h$. Everything else on this page
generalizes this one sentence.

*Example.* $f(x) = x^2$ at $x=3$: $f'(3)=6$, so $f(3.01) \approx 9 + 6(0.01) = 9.06$. The truth is $9.0601$.

### A2. Many inputs: partial derivatives and the gradient

A loss depends on many parameters, $L(w_1, \dots, w_d)$. The **partial derivative** $\partial L/\partial w_j$ is the exchange rate for
nudging only $w_j$. Stack all of them and you get the **gradient**:

```math
\nabla L(w) = \Big(\tfrac{\partial L}{\partial w_1}, \dots, \tfrac{\partial L}{\partial w_d}\Big), \qquad
L(w + h) \approx L(w) + \nabla L(w)^\top h .
```

**Why the gradient points uphill.** Move a tiny distance $\epsilon$ in a unit direction $u$. The change is
$\epsilon\, \nabla L^\top u = \epsilon \lVert \nabla L\rVert \cos\theta$ (MATH-01 §B1). It is largest when $u$ points along $\nabla L$
and most negative when $u$ points along $-\nabla L$. So:

> **$-\nabla L$ is the direction of steepest descent**, and $\lVert\nabla L\rVert$ is how steep it is.

That is the entire justification for gradient descent.

### A3. The chain rule: exchange rates multiply

If $y = f(u)$ and $u = g(x)$, a nudge $h$ in $x$ moves $u$ by $g'(x)h$, which moves $y$ by $f'(u)\,g'(x)\,h$:

```math
\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}.
```

If $x$ influences $y$ through **several paths** (say through $u$ and through $v$), the effects **add up**:

```math
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial y}{\partial v}\frac{\partial v}{\partial x}.
```

"Multiply along a path, add across paths." Backpropagation is nothing more than applying this rule efficiently.

### A4. The matrix-calculus identities you actually need

You can derive almost every gradient in classical ML from four identities. Here $a$, $w$ are vectors and $A$ is a matrix:

| Function of $w$ | Gradient w.r.t. $w$ |
|---|---|
| $a^\top w$ | $a$ |
| $w^\top A w$ | $(A + A^\top)w$, which is $2Aw$ if $A$ is symmetric |
| $\lVert w \rVert^2 = w^\top w$ | $2w$ |
| $\lVert Xw - y\rVert^2$ | $2X^\top (Xw - y)$ |

**Shape check trick.** The gradient of a scalar with respect to $w$ always has the **same shape as $w$**. If your formula gives a different
shape, a transpose is missing. ([The Matrix Calculus You Need for Deep Learning](https://explained.ai/matrix-calculus/) has the full set.)

**Derivation of the last row** (do this once by hand, it's the most important gradient in the curriculum):

```math
\lVert Xw-y\rVert^2 = (Xw-y)^\top(Xw-y) = w^\top X^\top X w - 2\,y^\top X w + y^\top y .
```

Apply the first two identities. The first term gives $2X^\top X w$ ($X^\top X$ is symmetric), the second gives $-2X^\top y$, and the third is constant. So

```math
\nabla_w \lVert Xw-y\rVert^2 = 2X^\top X w - 2X^\top y = 2X^\top(Xw - y).
```

Set it to zero and you get $X^\top X w = X^\top y$, **the same normal equations** you found by geometry in MATH-01 §B4.

**Reading the formula:** $Xw - y$ is the vector of errors. $X^\top(\text{errors})$ sums, for each feature, "error × feature value" over all
examples. A weight's gradient is large when its feature correlates with the current mistakes. That intuition carries over unchanged to logistic regression (CORE-03) and to the last layer of every neural net.

### A5. Computational graphs and backpropagation

Write a computation as a graph of simple operations. Take the lesson's self-check function, $L = (\sigma(wx) - y)^2$, with the
sigmoid $\sigma(z) = 1/(1+e^{-z})$, whose derivative is $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:

```
w ─┐
   ├─(×)── z ──(σ)── s ──(−y)── e ──(square)── L
x ─┘
```

**Forward pass:** compute and *store* every intermediate value. **Backward pass:** start from $\partial L/\partial L = 1$ and walk the
graph in reverse. At each node, multiply the incoming gradient by that node's local derivative:

```math
\frac{\partial L}{\partial w} = \underbrace{2e}_{\partial L/\partial e}\cdot\underbrace{1}_{\partial e/\partial s}\cdot\underbrace{s(1-s)}_{\partial s/\partial z}\cdot\underbrace{x}_{\partial z/\partial w}.
```

*Worked numbers.* With $w = 0.5$, $x = 2$, $y = 1$:

| Step | Forward value | Local derivative | Running gradient $\partial L/\partial(\cdot)$ |
|---|---|---|---|
| $z = wx$ | $1$ | $\partial z/\partial w = x = 2$ | $-0.1058 \times 2 = -0.2115$ ← $\partial L/\partial w$ |
| $s = \sigma(z)$ | $0.7311$ | $s(1-s) = 0.1966$ | $-0.5379 \times 0.1966 = -0.1058$ |
| $e = s - y$ | $-0.2689$ | $1$ | $-0.5379$ |
| $L = e^2$ | $0.0723$ | $2e = -0.5379$ | $1$ (start here) |

Read the table bottom-up for the backward pass. The negative gradient says that **increasing $w$ lowers the loss**, which makes sense:
$s$ is below the target 1, and a bigger $w$ pushes $s$ up.

**Why reverse mode?** A network has one scalar loss and millions of parameters. Going backwards from the loss gives **all** the
parameter gradients in one sweep, at a cost of about 2–3 forward passes. Going forwards would need one sweep *per parameter*. That
asymmetry is why backprop made deep learning practical. You'll build exactly this engine in [DL-01](../deep-learning/01-neural-networks-from-scratch.md).

### A6. Always check gradients numerically

The **central difference** $\frac{L(w+h) - L(w-h)}{2h}$ has error $O(h^2)$, so with $h \approx 10^{-5}$ it matches an analytic gradient to
around 8 digits. Whenever you hand-write a gradient, check it this way. This is called **gradient checking**.

```python
import numpy as np

sig = lambda z: 1 / (1 + np.exp(-z))
def loss(w, x=2.0, y=1.0):
    return (sig(w * x) - y) ** 2

w, x, y = 0.5, 2.0, 1.0
z = w * x; s = sig(z); e = s - y; L = e ** 2
dL_dw = 2 * e * 1 * s * (1 - s) * x            # chain rule, node by node
h = 1e-5
numeric = (loss(w + h) - loss(w - h)) / (2 * h)
print(f"z={z:.4f} s={s:.4f} e={e:.4f} L={L:.4f}")
print(f"analytic dL/dw = {dL_dw:.6f}, numeric = {numeric:.6f}")

# Matrix gradient check: grad of ||Xw - y||^2 is 2 X^T (Xw - y)
rng = np.random.default_rng(0)
X, yv, wv = rng.normal(size=(6, 3)), rng.normal(size=6), rng.normal(size=3)
f = lambda w: np.sum((X @ w - yv) ** 2)
g = 2 * X.T @ (X @ wv - yv)
g_num = np.array([(f(wv + h * np.eye(3)[j]) - f(wv - h * np.eye(3)[j])) / (2 * h) for j in range(3)])
print("max |analytic - numeric| =", np.abs(g - g_num).max())
```

```python
# The same thing with PyTorch autograd: it builds the graph and runs the backward pass for you.
import torch
w_t = torch.tensor(0.5, requires_grad=True)
L_t = (torch.sigmoid(w_t * 2.0) - 1.0) ** 2
L_t.backward()
print("autograd dL/dw =", round(w_t.grad.item(), 6))
```

---

## Block B: Optimization

### B1. Gradient descent

Repeat this update:

```math
w_{t+1} = w_t - \eta\, \nabla L(w_t)
```

Here $\eta$ is the **learning rate**. Why it works: from §A2, $L(w - \eta g) \approx L(w) - \eta \lVert g\rVert^2$. That decrease is
negative for small enough $\eta$. "Small enough" is the whole difficulty: the linear approximation only holds locally.

### B2. Convexity: when "downhill" always leads home

A function is **convex** if the straight line between any two points on its graph lies above the graph:

```math
L(\theta a + (1-\theta) b) \le \theta L(a) + (1-\theta) L(b) \quad \text{for } \theta \in [0,1].
```

- **Twice-differentiable test:** $L$ is convex if and only if its Hessian (the matrix of second derivatives, $H_{ij} = \partial^2 L / \partial w_i \partial w_j$) is PSD everywhere.
- **The payoff:** for a convex $L$, any point where $\nabla L = 0$ is a **global** minimum. There are no bad local minima to get stuck in.
  So gradient descent with a suitable step size finds the global optimum.
- **Convex in ML:** least squares (its Hessian is $2X^\top X$, which is PSD), logistic regression, ridge, lasso, and SVMs.
- **Not convex:** neural networks. Their training works for different reasons (overparameterization and SGD noise; see DL-03).

### B3. Why the condition number controls speed

Near a minimum, every smooth loss looks like a quadratic bowl, $L(w) \approx \tfrac12 (w - w^*)^\top H (w - w^*)$. Rotate into the
Hessian's eigenbasis (MATH-01 §C2) and the bowl splits into independent 1-D problems, one per eigenvalue $\lambda_i$. GD on direction $i$ gives

```math
\delta_{t+1} = (1 - \eta\lambda_i)\,\delta_t .
```

- To converge in **every** direction you need $\lvert 1 - \eta\lambda_i\rvert < 1$ for all $i$, which means $\eta < 2/\lambda_{\max}$.
  Above that, the steepest direction **diverges**. That's your loss going to `nan`.
- The flattest direction then shrinks by a factor of only $1 - \eta\lambda_{\min} \approx 1 - 2/\kappa$ per step, where $\kappa = \lambda_{\max}/\lambda_{\min}$.
  The best fixed step size, $\eta = 2/(\lambda_{\max} + \lambda_{\min})$, gives a rate of $\frac{\kappa - 1}{\kappa + 1}$ per step.

**So a long, narrow valley (large $\kappa$) forces a small step and makes progress slow.** Feature scaling (CORE-02) makes the bowl
rounder (smaller $\kappa$). Normalization layers (DL-03) do the same for networks. Momentum and Adam are ways to cope with large $\kappa$.

### B4. Momentum

Keep a running "velocity" of past gradients:

```math
v_{t+1} = \beta v_t + \nabla L(w_t), \qquad w_{t+1} = w_t - \eta\, v_{t+1}.
```

In the narrow directions the gradient keeps flipping sign, so it cancels in $v$. Along the valley floor it points the same way every time, so it
**accumulates**. The result: oscillation is damped, and progress along the floor speeds up. With well-tuned $\beta$, the rate improves from
about $1 - 1/\kappa$ to about $1 - 1/\sqrt{\kappa}$, which is huge when $\kappa = 10^4$. [Why Momentum Really Works](https://distill.pub/2017/momentum/) animates this.

### B5. Stochastic gradients

The full gradient is an average over $N$ examples, $\nabla L = \frac1N \sum_i \nabla \ell_i$. A **mini-batch** of $B$ random examples gives an
**unbiased** estimate whose noise shrinks like $1/\sqrt{B}$. So with SGD:

- each step is $N/B$ times cheaper;
- steps are noisy, so you need a decaying learning rate, or else you accept "bouncing around" near the minimum;
- the noise can help with non-convex losses, by escaping sharp minima and saddle points.

### B6. Adam in one paragraph

Adam keeps a momentum estimate $m_t$ (the mean of the gradients) **and** a running average $v_t$ of squared gradients. It steps each
parameter by $m_t / (\sqrt{v_t} + \epsilon)$. Dividing by the gradient's typical size gives every parameter its own step size, a cheap
diagonal fix for bad conditioning. The full update, with bias correction, is in [DL-03 notes](../deep-learning/03-training-deep-networks.md).

### B7. Constraints: Lagrange multipliers (just the idea)

To minimize $f(w)$ subject to $g(w) = 0$: at the optimum you can't move along the constraint surface and still decrease $f$. So $\nabla f$
must be perpendicular to that surface, which means parallel to $\nabla g$: $\nabla f = \lambda \nabla g$. Bundle this into the
**Lagrangian** $\mathcal{L}(w, \lambda) = f(w) - \lambda g(w)$ and set its gradient to zero.

*Example.* PCA maximizes $u^\top S u$ subject to $u^\top u = 1$. The condition $\nabla_u = 2Su - 2\lambda u = 0$ gives $Su = \lambda u$: **the optimal direction
is an eigenvector**, which is the result MATH-01 §C3 reached by a different argument. SVMs (CORE-09) use the inequality version, the KKT conditions.

```python
# GD on an ill-conditioned bowl: L(w) = 0.5 * w^T H w with H = diag(1, 10)  (kappa = 10)
import numpy as np
H = np.diag([1.0, 10.0])
def run(eta, steps=300, beta=0.0):
    w, v = np.array([1.0, 1.0]), np.zeros(2)
    for t in range(steps):
        v = beta * v + H @ w
        w = w - eta * v
        if np.linalg.norm(w) < 1e-6:
            return t + 1
        if np.linalg.norm(w) > 1e6:
            return "diverged"
    return f">{steps}"
print("eta=0.05            steps:", run(0.05))
print("eta=2/(1+10)=0.182  steps:", run(2 / 11))
print("eta=0.21 (>2/10)    steps:", run(0.21))
print("momentum beta=0.5, eta=0.18 steps:", run(0.18, beta=0.5))
print("theory: best plain-GD rate (k-1)/(k+1) =", 9 / 11)
```

---

## Pitfalls & misconceptions

- **"The gradient is the direction to move."** It's the direction to move *away from*: subtract it.
- **"A zero gradient means a minimum."** It could be a maximum or a saddle point. In high dimensions, saddles are far more common than local minima.
- **Learning rate too big → `nan`.** That's $\eta > 2/\lambda_{\max}$ in action. Unscaled features with huge values make $\lambda_{\max}$ huge.
- **Hand-derived gradients without a numeric check.** Always gradient-check new code on a tiny input.
- **Mixing up $\nabla_w$ (shape of $w$) and the Jacobian** (shape output × input). Gradients are for scalar losses. Jacobians are for vector-valued functions.

## Cheat sheet

| Idea | Formula |
|---|---|
| Linear approximation | $f(x+h) \approx f(x) + \nabla f(x)^\top h$ |
| Chain rule | multiply along paths, add across paths |
| Least-squares gradient | $\nabla_w \lVert Xw-y\rVert^2 = 2X^\top(Xw-y)$ |
| Sigmoid derivative | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ |
| Gradient descent | $w \leftarrow w - \eta\nabla L$ |
| Stability limit | $\eta < 2/\lambda_{\max}(H)$ |
| Convergence rate (quadratic) | about $(\kappa-1)/(\kappa+1)$ per step at the best $\eta$ |
| Momentum | $v \leftarrow \beta v + \nabla L$, then $w \leftarrow w - \eta v$ |
| Numeric gradient | $\frac{L(w+h)-L(w-h)}{2h}$ |
| Lagrange condition | $\nabla f = \lambda \nabla g$ |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Compute ∂L/∂w for L = (σ(w·x) − y)² using the chain rule.</summary>

With $z = w^\top x$ and $s = \sigma(z)$: $\nabla_w L = 2(s - y)\, s(1-s)\, x$. The scalar case and its numbers are worked in §A5.
Notice the factor $s(1-s)$: when $s$ saturates near 0 or 1, the gradient almost vanishes, even if the prediction is badly wrong. That is why
classification uses cross-entropy instead of squared error (CORE-03 notes): with cross-entropy, the $s(1-s)$ factor cancels.
</details>

<details>
<summary>2. Why does a convex loss guarantee that GD finds the global minimum (with a suitable step size)?</summary>

For a convex $L$, the tangent plane at any point lies below the graph: $L(v) \ge L(w) + \nabla L(w)^\top (v - w)$. If $\nabla L(w) = 0$, this says
$L(v) \ge L(w)$ for every $v$, so any stationary point is a global minimum. With $\eta$ below $2/L$, where $L$ is the gradient's Lipschitz constant, each GD step
strictly decreases the loss until the gradient vanishes, so GD converges to that global minimum.
</details>

<details>
<summary>3. What does the Hessian's condition number say about how fast GD converges?</summary>

The step size is capped by the steepest direction ($\eta < 2/\lambda_{\max}$), while progress in the flattest direction is about $\eta\lambda_{\min}$ per step.
So the number of iterations grows roughly linearly with $\kappa = \lambda_{\max}/\lambda_{\min}$ (and with $\sqrt{\kappa}$ under momentum). The code cell above shows it.
</details>

## Where this leads

- **Next in math:** [MATH-03 Probability & Statistics](03-probability-statistics.md). Where do loss functions *come from*? From maximum likelihood.
- **Used directly in:** [CORE-02 notes](../core-ml/02-linear-models-gradient-descent.md) (GD for regression), [DL-01 notes](../deep-learning/01-neural-networks-from-scratch.md) (backprop for a whole network), [DL-03 notes](../deep-learning/03-training-deep-networks.md) (momentum, Adam, schedules).
