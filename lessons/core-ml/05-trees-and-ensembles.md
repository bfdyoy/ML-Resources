# CORE-05: Trees, Random Forests & Gradient Boosting

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Core ML | ~7 h | L2 | CORE-04 |

## Why this matters
On tabular data, which is most business data, gradient-boosted trees (XGBoost, LightGBM, CatBoost) are still
the model to beat in 2026. Knowing *why* bagging reduces variance and boosting reduces bias lets you tune them
on purpose instead of by trial and error.

## Learning goals
By the end you can:
- Explain how a decision tree chooses splits (impurity / RSS), and why deep trees overfit.
- Explain bagging and random forests as variance reduction, and out-of-bag error.
- Explain gradient boosting as gradient descent in function space: fitting residuals step by step.
- Tune the key GBM hyperparameters (learning rate, number of trees, depth, subsampling, early stopping).
- Say when a linear model or a neural net would beat trees on a given problem.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Intuition** | [MLU-Explain: Decision Trees](https://mlu-explain.github.io/decision-tree/) then [Random Forest](https://mlu-explain.github.io/random-forest/) | Both essays | 40 min |
| 2 | **Read** | [ISLP](https://www.statlearning.com/) Ch. 8 "Tree-Based Methods" | §8.1 (regression and classification trees, pruning), §8.2 (bagging, random forests, boosting, BART) | 2 h |
| 3 | **Read + Build** | [Géron, *Hands-On ML*, Ch. 5 "Decision Trees" + Ch. 6 "Ensemble Learning and Random Forests"](https://github.com/ageron/handson-mlp) | Run `05_decision_trees.ipynb` and `06_ensemble_learning_and_random_forests.ipynb`. Focus on the gradient boosting and stacking sections. | 2.5 h |
| 4 | **Watch** | [StatQuest: Gradient Boost & XGBoost](https://www.youtube.com/playlist?list=PLZ5DHV9_5h9vQwAImmNi1RfoTtSuOUjwM) | Gradient Boost parts 1–2 (regression). Add XGBoost part 1 if you'll use XGBoost. | 1 h |
| 5 | **Build** | [Kaggle Learn: Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) | The "XGBoost" lesson and its exercise (early stopping, `n_estimators`, `learning_rate`) | 45 min |

**Notes for the learner:** StatQuest is the one place in this track where video is the best *first* explanation.
Watching the residual-fitting steps drawn out makes boosting click. Still, read ISLP §8.2.3 afterwards for the
algorithm in writing.

## Check your understanding
1. Why is a single deep tree high-variance, and how does averaging many trees fix that?
2. What does the random forest's feature subsampling (`max_features`) add on top of bagging?
3. In gradient boosting, what exactly does each new tree fit? Why is a small learning rate plus many trees usually better?
4. Why are tree models insensitive to monotonic feature transformations (e.g. log-scaling)?
5. Why can't a tree-based model extrapolate beyond the range of its training targets?
6. *(debug)* Your LightGBM model's validation loss improves for 300 rounds, then gets worse, while training loss keeps falling. What do you change?

## Mini-project
**Task:** On one tabular dataset, compare logistic regression, random forest, and a GBM (`HistGradientBoostingClassifier`,
XGBoost, or LightGBM) with early stopping. Report CV scores and training time, plus a short note on which you'd ship and why.
**Dataset:** UCI Adult (`sklearn.datasets.fetch_openml("adult", version=2)`) or the Kaggle Titanic competition data.
**Deliverable:** A comparison table plus a paragraph of reasoning.

## Go deeper
- [Inria scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/): module "Ensemble of models", which covers bagging vs boosting with clean experiments.
- [The Hundred-Page ML Book](https://themlbook.com/): the ensemble-learning chapter, for a compact recap.
