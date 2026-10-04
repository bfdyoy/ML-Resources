# CORE-11: Uncertainty: Calibration & Conformal Prediction

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Core ML | ~6 h | L2 | CORE-03, CORE-04 · math: [MATH-03](../math/03-probability-statistics.md) block A |

## Why this matters
A model that says "90% sure" should be right 90% of the time. Most models, especially deep nets, aren't. And point predictions
hide how wrong a model *might* be. Calibration fixes the first problem. Conformal prediction gives prediction sets and intervals
with **guaranteed coverage** on top of any model, with no retraining. Both are cheap and high-value in production.

## Learning goals
By the end you can:
- Read a reliability diagram and compute expected calibration error (ECE).
- Recalibrate a classifier with Platt scaling, isotonic regression, or temperature scaling, using held-out data.
- Explain split conformal prediction and its coverage guarantee, and what "exchangeability" means.
- Build conformal prediction sets for classification and intervals for regression (including conformalized quantile regression).
- Know the limits: coverage is marginal, not per-group, and breaks under distribution shift.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Uncertainty: Calibration & Conformal Prediction](../../notes/core-ml/11-uncertainty-calibration-conformal.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [sklearn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | The whole page: calibration curves, `CalibratedClassifierCV`, sigmoid vs isotonic | 45 min |
| 2 | **Read** | [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (Guo et al.) | §1–3 and §4.2 (temperature scaling). Look at Figure 1. | 45 min |
| 3 | **Read** | [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511) (Angelopoulos & Bates) | §1 (the core recipe), §2 (examples), §3 (evaluating conformal procedures) | 1.5 h |
| 4 | **Build** | [conformal-prediction notebooks](https://github.com/aangelopoulos/conformal-prediction) (the paper's companion) + [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | Run the classification and regression examples. Then apply MAPIE to your own model. | 1.5 h |

**Notes for the learner:** This is one of the rare topics where a paper *is* the best first read. The Angelopoulos & Bates
tutorial was written for practitioners and comes with runnable code.

## Check your understanding
1. Why are modern deep nets usually *over*confident?
2. Why must you calibrate on data the model wasn't trained on?
3. What exactly does "90% coverage" guarantee in split conformal prediction? Over what is the probability taken?
4. Why are conformal sets bigger for hard examples? Is that a feature or a bug?
5. Your conformal intervals have 90% coverage overall, but 70% for one demographic group. How is that possible, and what can you do?
6. *(debug)* After deploying, empirical coverage drops from 90% to 75%. Name the most likely cause and a fix.

## Mini-project
**Task:** Take a GBM classifier and a small neural net. Plot reliability diagrams before and after calibration, report ECE,
then wrap both with conformal prediction sets at 90%. Report coverage and average set size overall and per class.
**Dataset:** A multiclass tabular dataset (e.g. `fetch_covtype` subset), or CIFAR-10 logits from your DL-04 model.
**Deliverable:** Reliability diagrams, and a coverage/set-size table.

## Go deeper
- [EL-02 Bayesian & Probabilistic ML](../electives/02-bayesian-probabilistic-ml.md): the Bayesian route to uncertainty.
- [Probabilistic Machine Learning](https://probml.github.io/pml-book/): the chapters on Bayesian inference and uncertainty.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 03: Data, features & evaluation](../../toolbox/03-data-features-evaluation.md) (metrics & calibrated predictions) · [Toolbox 12: Specialized topics](../../toolbox/12-specialized-topics.md) (uncertainty).
- **Papers:** [Interpretability, uncertainty & responsible ML](../../papers/10-interpretability-uncertainty-responsible.md). Start with the ⭐ ones.
- **Implement it yourself:** write ECE and split-conformal from scratch (≈30 lines each). Check them against MAPIE.
- **Drills:** more in [exercises/](../../exercises/README.md).
