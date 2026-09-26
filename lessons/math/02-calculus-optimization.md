# MATH-02: Calculus & Optimization for ML (just-in-time)

| Track | Time | Level | Used by |
|---|---|---|---|
| Math | ~5 h (in pieces) | L1→L2 | CORE-02, DL-01, DL-03 |

## Block A: Derivatives and the chain rule (for CORE-02, DL-01)
| Step | Resource | Scope | Time |
|---|---|---|---|
| **Intuition** | [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | Ch. 1–4 (essence, derivative paradox, formulas, **chain rule and product rule**) | 1 h |
| **Read** | [MML book](https://mml-book.github.io/) ([PDF](https://mml-book.github.io/book/mml-book.pdf)) | Ch. 5 "Vector Calculus": partial derivatives and gradients (§5.1–5.2), gradients of vectors and matrices (§5.3–5.4), **backpropagation and automatic differentiation (§5.6)** | 2 h |

## Block B: Optimization (for CORE-02, DL-03)
| Step | Resource | Scope | Time |
|---|---|---|---|
| **Read** | [MML book](https://mml-book.github.io/) | Ch. 7 "Continuous Optimization": gradient descent, step size, momentum, SGD (§7.1). Convexity (§7.3). | 1 h |
| **Intuition** | [Why Momentum Really Works](https://distill.pub/2017/momentum/) (Distill) | The interactive figures | 30 min |

## Self-check
1. Compute ∂L/∂w for `L = (σ(w·x) − y)²` using the chain rule.
2. Why does a convex loss guarantee that gradient descent finds the global minimum (with a suitable step size)?
3. What does the Hessian's condition number say about how fast gradient descent converges?
