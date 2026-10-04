# CORE-12: Hyperparameter Optimization & Experiment Discipline

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Core ML | ~6 h | L2 | CORE-04, CORE-05 (CORE-09 helps for the GP part) |

## Why this matters
Tuning is where many people either waste weeks or quietly overfit their validation set. A good search strategy (random search,
Bayesian optimization, early stopping of bad trials) and good experiment hygiene get you most of the gains at a fraction of the cost.
They also keep you honest about what actually improved.

## Learning goals
By the end you can:
- Explain why random search beats grid search when only a few hyperparameters matter.
- Explain Bayesian optimization (surrogate model + acquisition function), and when it's worth the overhead.
- Use successive halving / Hyperband to stop bad configurations early.
- Run an Optuna study with pruning, and read its importance and history plots.
- Avoid "overfitting the validation set" with nested CV, a held-out test set, and logging every trial.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Hyperparameter Optimization](../../notes/core-ml/12-hyperparameter-optimization.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read + Build** | [Inria scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/) | Module "Hyperparameter tuning" (grid and random search, nested CV) and its exercises | 1.5 h |
| 2 | **Read** | [sklearn: Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html) | Randomized search, successive halving (`HalvingRandomSearchCV`), and the tips section | 45 min |
| 3 | **Read** | [Practical Bayesian Optimization of ML Algorithms](https://arxiv.org/abs/1206.2944) | §1–3: the GP surrogate and expected improvement. Skim the experiments. | 1 h |
| 4 | **Build** | [Optuna](https://github.com/optuna/optuna) | Tune a LightGBM/XGBoost model with a TPE sampler and median pruning. Plot optimization history and parameter importances. | 1.5 h |
| 5 | *Read (DL-specific)* | [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) | "Choosing the initial configuration" and "A scientific approach to improving model performance" | 45 min |

## Check your understanding
1. Draw why random search covers "important" dimensions better than a grid with the same budget.
2. What does the acquisition function trade off in Bayesian optimization?
3. How does successive halving decide what to kill, and when can it kill a slow starter that would have won?
4. You ran 500 trials and picked the best CV score. Why is that score optimistically biased, and how do you get an honest estimate?
5. Which hyperparameters would you tune first for a GBM? And for a neural net?
6. *(debug)* Optuna's "best" trial is much worse when rerun with another seed. What went wrong, and how do you design the objective better?

## Mini-project
**Task:** Tune one GBM three ways under the same budget (50 trials): grid, random, and Optuna TPE with pruning. Compare the best CV
score, the test score, and the wall-clock time. Log every trial (CSV or MLflow).
**Dataset:** UCI Adult, or your CORE-05 dataset.
**Deliverable:** A comparison table and an optimization-history plot.

## Go deeper
- [Hyperband](https://arxiv.org/abs/1603.06560): early stopping framed as a bandit problem.
- [Optuna paper](https://arxiv.org/abs/1907.10902): the design behind define-by-run search spaces.
- [A disciplined approach to NN hyper-parameters](https://arxiv.org/abs/1803.09820): the LR range test and 1-cycle schedule for deep nets.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 03: Data, features & evaluation](../../toolbox/03-data-features-evaluation.md) (validation & model selection).
- **Papers:** [Tabular, time series, recsys & causal](../../papers/09-tabular-timeseries-recsys-causal.md) (hyperparameter optimization section).
- **Implement it yourself:** a 40-line successive-halving loop around any sklearn estimator.
- **Drills:** more in [exercises/](../../exercises/README.md).
- **Playbook:** [choosing a search strategy](../../playbook/01-choosing-algorithms.md#12-hyperparameter-search-grid-vs-random-vs-bayesian-vs-successive-halving) · [hill-climbing blends](../../playbook/02-tabular-tricks.md#8-blend-by-hill-climbing-on-out-of-fold-predictions).
