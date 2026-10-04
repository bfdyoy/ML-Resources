# Lab 03: Linear regression three ways

[← Labs](../README.md) · Lesson: [CORE-02 Linear models & gradient descent](../../lessons/core-ml/02-linear-models-gradient-descent.md) · Notes: [CORE-02 notes](../../notes/core-ml/02-linear-models-gradient-descent.md)

**Time** ≈ 1.5 h · **You'll practise:** the normal equations as a least-squares solve, deriving and *checking* a gradient, the effect of the step size, and ridge as a modified linear system.

| Function | Checked against |
|---|---|
| `fit_closed_form` | `sklearn.linear_model.LinearRegression` |
| `mse_and_grad` | central finite differences |
| `fit_gd` | your closed form; monotone loss; divergence when `lr` is too large |
| `fit_ridge` | `sklearn.linear_model.Ridge`; coefficients shrink as `alpha` grows |

```bash
pytest labs/03-linear-regression
```

**Bonus:**
1. For this data, the largest stable step size is `2 / λ_max`, where `λ_max` is the largest eigenvalue of the Hessian `(2/n) [1 X]^T [1 X]`.
   Compute it, and confirm that `fit_gd` diverges just above it and converges just below it.
2. Rescale one column of X by 100. How many more GD steps does convergence take, and why? (Condition number, CORE-02 notes.)
