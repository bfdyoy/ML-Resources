# EL-05: Causal Inference & Uplift Modeling

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Elective | ~11 h | L2 | CORE-02, CORE-05, [MATH-03](../math/03-probability-statistics.md) |

## Why this matters
Predictive models answer "what will happen?". Businesses and scientists usually need "what happens **if we act**?": does the
discount cause purchases, does the feature cause retention? Correlational models give confidently wrong answers here. Causal
inference gives you the tools (potential outcomes, DAGs, and identification strategies) and ML-powered estimators for
heterogeneous effects, which is uplift modeling.

## Learning goals
By the end you can:
- Frame questions with potential outcomes (ATE, ATT, CATE), and explain the fundamental problem of causal inference.
- Draw causal DAGs, and spot confounders, colliders, and mediators (what to control for, and what *not* to).
- Apply the main identification strategies: randomization, adjustment/IPW/matching, difference-in-differences, instrumental variables, and regression discontinuity.
- Estimate heterogeneous treatment effects with meta-learners (S/T/X) and double/debiased ML.
- Evaluate uplift models (uplift/Qini curves), and know their limits.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Causal Inference & Uplift Modeling](../../notes/electives/05-causal-inference-uplift.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read + Build** | [Causal Inference for the Brave and True](https://matheusfacure.github.io/python-causality-handbook/) (Part I) | From "Introduction To Causality" through randomized experiments, regression, graphical causal models, propensity score, and difference-in-differences. Run the notebooks. | 4 h |
| 2 | **Read** | [Brady Neal: Introduction to Causal Inference](https://www.bradyneal.com/causal-inference-course) | The course-notes chapters on causal graphs, d-separation, and the backdoor criterion | 2 h |
| 3 | **Read + Build** | [Brave and True](https://matheusfacure.github.io/python-causality-handbook/) (Part II) | The heterogeneous-effects chapters, including [Meta Learners](https://matheusfacure.github.io/python-causality-handbook/21-Meta-Learners.html) | 2 h |
| 4 | **Build** | [EconML](https://github.com/py-why/EconML) | T-/X-learner and DML on a marketing-style dataset; plot uplift curves | 1.5 h |
| 5 | *Read (optional)* | [Causal Inference: The Mixtape](https://mixtape.scunning.com/) | The chapters on IV and regression discontinuity (the econometric view) | 1.5 h |

## Check your understanding
1. Why can't you ever observe an individual treatment effect directly?
2. Draw a DAG where controlling for a variable *introduces* bias (a collider). Why does it?
3. What assumptions does propensity-score weighting need (unconfoundedness, overlap)?
4. What is the parallel-trends assumption in diff-in-diff, and how would you sanity-check it?
5. S-learner vs T-learner vs X-learner: when does each struggle?
6. Why can't you evaluate uplift models with ordinary accuracy?
7. *(debug)* Your observational estimate says a feature *increases* churn, but an A/B test says it decreases it. What are the likely explanations?

## Mini-project
**Task:** Using a randomized marketing dataset (or one you simulate with known effects), estimate the ATE naively and properly, then fit uplift models
and choose whom to target under a budget. If you simulated the data, compare the estimates with the true effects.
**Deliverable:** A DAG, an estimates table, an uplift/Qini curve, and a targeting recommendation.

## Go deeper
- Papers: [Double/Debiased ML](https://arxiv.org/abs/1608.00060) · [Meta-learners for heterogeneous treatment effects](https://arxiv.org/abs/1706.03461)
- [EL-02 Bayesian & Probabilistic ML](02-bayesian-probabilistic-ml.md): [Statistical Rethinking 2026](https://github.com/rmcelreath/stat_rethinking_2026) has an excellent causal-DAG treatment.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 12: Specialized topics](../../toolbox/12-specialized-topics.md) (causal inference & uplift).
- **Papers:** [Tabular, time series, recsys & causal](../../papers/09-tabular-timeseries-recsys-causal.md) (causal ML section).
- **Implement it yourself:** IPW and T-learner estimators from scratch. Check them against EconML on simulated data with a known effect.
- **Drills:** more in [exercises/](../../exercises/README.md).
