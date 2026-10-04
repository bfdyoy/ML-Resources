# CORE-02: Linear Models & Gradient Descent

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Core ML | ~7 h | L2 | CORE-01 · math: [MATH-01](../math/01-linear-algebra.md), [MATH-02](../math/02-calculus-optimization.md) |

## Why this matters
Linear regression is the "hello world" of ML, and it is also what a neural network's last layer does. Almost
every idea you'll meet later first shows up here: loss functions, closed-form versus iterative solutions,
learning rates, feature scaling, and interpreting coefficients.

## Learning goals
By the end you can:
- Explain least squares geometrically and statistically, and interpret coefficients, standard errors, and R².
- Derive the gradient of MSE and implement batch, stochastic, and mini-batch gradient descent from scratch.
- Explain why feature scaling changes how fast gradient descent converges.
- Use polynomial features and recognise when a linear model is under-fitting.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Linear Models & Gradient Descent](../../notes/core-ml/02-linear-models-gradient-descent.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Intuition** | [MLU-Explain: Linear Regression](https://mlu-explain.github.io/linear-regression/) | The whole essay. Play with the interactive fits. | 20 min |
| 2 | **Read** | [ISLP](https://www.statlearning.com/) Ch. 3 "Linear Regression" | §3.1–3.3 (simple, multiple, qualitative predictors, interactions, potential problems). Skim §3.4 and read §3.5 (vs KNN). | 2 h |
| 3 | **Read + Build** | [Géron, *Hands-On ML*, Ch. 4 "Training Models"](https://github.com/ageron/handson-mlp) | Normal equation, the three GD variants, polynomial regression, learning curves. Run `04_training_linear_models.ipynb`. | 2 h |
| 4 | **Build** | From scratch, in NumPy | Write `fit_gd(X, y, lr, epochs, batch_size)` and check it matches `sklearn.linear_model.LinearRegression` to 4 decimals. | 1 h |
| 5 | *Watch (optional)* | [3Blue1Brown: Gradient descent](https://www.3blue1brown.com/lessons/gradient-descent/) | The visual picture of descending a loss surface | 20 min |

**Notes for the learner:** ISLP is about *understanding* the model (inference). Géron is about *training* it
(optimization). You need both views. Don't skip ISLP §3.3.3 "Potential Problems" (non-linearity, collinearity,
outliers, and high leverage). It's the practical diagnostic checklist.

## Check your understanding
1. What does a coefficient mean in multiple regression, and how does that differ from simple regression?
2. Why can the normal equation be slow or unstable, and what does gradient descent trade for that?
3. Sketch the loss path of GD on unscaled versus scaled features. Why do they differ?
4. How do you read a learning curve that shows training and validation error both high and close together?
5. What does collinearity do to coefficient estimates and to predictions?
6. *(debug)* Your GD loss goes to `nan` after 3 epochs. What are the two most likely causes?

## Mini-project
**Task:** Predict fuel efficiency. Fit OLS, then GD you wrote yourself, then polynomial degree 2 and degree 5. Plot learning
curves for each and explain what you see.
**Dataset:** Auto MPG (`seaborn.load_dataset("mpg")`) or ISLP's `Auto` dataset (`from ISLP import load_data`).
**Deliverable:** A notebook plus a one-paragraph diagnosis of under- versus over-fitting.

## Go deeper
- [MML book](https://mml-book.github.io/) Ch. 9 "Linear Regression": the full probabilistic (MLE/MAP/Bayesian) treatment.
- [Why Momentum Really Works](https://distill.pub/2017/momentum/) (Distill): when you're ready to see why vanilla GD is slow.

## Math refresher
- [MATH-01 Linear Algebra](../math/01-linear-algebra.md): matrix multiplication, projections
- [MATH-02 Calculus & Optimization](../math/02-calculus-optimization.md): gradients, the chain rule

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 02: Classical ML](../../toolbox/02-classical-ml.md), for every concept in this lesson, with alternatives.
- **Papers:** [Optimization, training & generalization](../../papers/01-optimization-training-generalization.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 1–3.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
- **Lab:** [Lab 03: Linear regression three ways](../../labs/03-linear-regression/README.md) (stubs + `pytest`).
