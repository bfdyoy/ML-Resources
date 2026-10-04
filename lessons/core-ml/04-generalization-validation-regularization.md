# CORE-04: Generalization: Bias-Variance, Validation & Regularization

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Core ML | ~7 h | L2 | CORE-02, CORE-03 |

## Why this matters
Every model you'll ever train is a trade-off between fitting the data and generalizing to new data. This lesson
gives you the tools to *measure* generalization honestly (cross-validation), to *control* it (regularization),
and to understand where the classical picture breaks down (double descent), which matters for deep learning later.

## Learning goals
By the end you can:
- Explain the bias-variance decomposition and connect it to model complexity.
- Choose between hold-out, k-fold, stratified, grouped, and time-series cross-validation.
- Explain how ridge and lasso shrink coefficients, and why the lasso produces sparse solutions.
- Tune hyperparameters with nested CV, or with a clean validation protocol, without leaking information.
- Describe double descent, and why "bigger model = more overfitting" isn't always true.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Generalization, Validation & Regularization](../../notes/core-ml/04-generalization-validation-regularization.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Intuition** | [MLU-Explain: Bias-Variance Tradeoff](https://mlu-explain.github.io/bias-variance/) then [Train, Test & Validation](https://mlu-explain.github.io/train-test-validation/) | Both essays | 40 min |
| 2 | **Read** | [ISLP](https://www.statlearning.com/) Ch. 2 §2.2.2 "The Bias-Variance Trade-Off" + Ch. 5 "Resampling Methods" | §2.2.2, §5.1 (all CV variants), §5.2 (bootstrap) | 1.5 h |
| 3 | **Read** | [ISLP](https://www.statlearning.com/) Ch. 6 "Linear Model Selection and Regularization" | §6.1 (subset selection, briefly), **§6.2 (ridge and lasso, in depth)**. Skim §6.4 (high dimensions). | 1.5 h |
| 4 | **Build** | [Inria scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/) | Modules "Selecting the best model" and "Hyperparameter tuning". Do the exercises, including nested CV. | 1.5 h |
| 5 | **Intuition** | [MLU-Explain: Double Descent](https://mlu-explain.github.io/double-descent/) | The whole essay. It bridges to deep learning. | 20 min |

**Notes for the learner:** The one sentence to remember: *any decision you make by looking at data is part of the
model, so it has to happen inside the cross-validation loop.*

## Check your understanding
1. Draw the typical training-error and test-error curves against model complexity, and label the high-bias and high-variance regions.
2. Why does lasso set some coefficients exactly to zero, while ridge doesn't? (Think about the constraint shapes.)
3. You have 5 measurements per patient. Why is plain k-fold CV wrong here, and what do you use instead?
4. Why must you use `TimeSeriesSplit` (or similar) for forecasting problems?
5. What problem does nested cross-validation solve?
6. *(debug)* You tried 400 hyperparameter configs, and the best CV score is much better than the score on fresh data. What happened?

## Mini-project
**Task:** On a wide dataset (p close to n), compare OLS, ridge, lasso, and elastic net, with regularization strength
chosen by CV. Plot coefficient paths, and report test performance using a correct nested protocol.
**Dataset:** ISLP `Hitters`, or `sklearn.datasets.load_diabetes` with polynomial features.
**Deliverable:** Coefficient-path plots, plus a table of CV score and test score per model.

## Go deeper
- [UDL](https://udlbook.github.io/udlbook/) Ch. 8 "Measuring Performance": the modern DL view of generalization, including double descent.
- [MLU-Explain: Double Descent 2](https://mlu-explain.github.io/double-descent2/): the mathematical follow-up.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 03: Data, features & evaluation](../../toolbox/03-data-features-evaluation.md), for every concept in this lesson, with alternatives.
- **Papers:** [Optimization, training & generalization](../../papers/01-optimization-training-generalization.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 3, 7, 17.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
- **Lab:** [Lab 05, part A: CV splitters](../../labs/05-trees-and-cross-validation/README.md) (stubs + `pytest`).
- **Playbook:** [shuffled-label test](../../playbook/05-outside-the-box.md#9-shuffle-the-labels-a-lie-detector-for-your-pipeline) · [adversarial validation](../../playbook/02-tabular-tricks.md#1-adversarial-validation-can-a-model-tell-train-from-test).
