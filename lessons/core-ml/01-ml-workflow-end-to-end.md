# CORE-01: The ML Workflow, End to End

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Core ML | ~7 h | L1→L2 | Python, pandas basics |

## Why this matters
Most failed ML projects don't fail because of the algorithm. They fail because the problem was framed badly, the
metric was wrong, or the data leaked. Before going deep on any model, you want a reliable **workflow**,
from framing the problem through evaluation to iteration. That workflow is the frame every later lesson fits into.

## Learning goals
By the end you can:
- Decide whether a problem should be solved with ML at all, and frame it (task type, label, metric, baseline).
- Run a full project: get data, explore it, split it, preprocess it, train, evaluate, and iterate.
- Explain why the test set is touched *once*, and what a validation set is for.
- Build a scikit-learn `Pipeline` so preprocessing is fitted on training data only.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: The ML Workflow, End to End](../../notes/core-ml/01-ml-workflow-end-to-end.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Intuition** | [Introduction to ML Problem Framing](https://developers.google.com/machine-learning/problem-framing) (Google) | The whole mini-course. Do the framing exercise. | 45 min |
| 2 | **Read + Build** | [Géron, *Hands-On ML*, Ch. 2 "End-to-End Machine Learning Project"](https://github.com/ageron/handson-mlp) | Read the chapter and run `02_end_to_end_machine_learning_project.ipynb` side by side. If you don't have the book, the notebook's comments are detailed enough to follow on their own. | 3 h |
| 3 | **Read** | [ISLP](https://www.statlearning.com/) Ch. 2 "Statistical Learning" ([free PDF](https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html)) | §2.1 (what we're estimating, and prediction vs inference) and §2.2.1 (measuring fit, training vs test MSE) | 1 h |
| 4 | **Build** | [Inria scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/) | Module "The predictive modeling pipeline". Do the notebooks on tabular data, preprocessing, and pipelines. | 1.5 h |

**Notes for the learner:** You're intermediate, so skim the parts you know (pandas plotting, for example). But don't
skip Géron's sections on stratified splitting and on `ColumnTransformer` + `Pipeline`. Those two habits prevent most
beginner bugs.

## Check your understanding
1. Give an example of a problem where ML is the *wrong* tool, and say what you would use instead.
2. Why do we stratify the train/test split, and when does it matter most?
3. What is a baseline model, and why should it be the first thing you build?
4. Why is `StandardScaler().fit(X)` on the full dataset a bug, even though the model code looks fine?
5. What's the difference between a validation set and a test set? What goes wrong if you tune on the test set?
6. *(debug)* Your model scores 0.97 in cross-validation but 0.71 once deployed. List three possible causes.

## Mini-project
**Task:** Build an end-to-end regression pipeline on a dataset you haven't used before. Frame it, get a baseline,
try two models, and write a 5-line "model card" (data, metric, result, known limitations).
**Dataset:** `sklearn.datasets.fetch_california_housing` (for a like-for-like with Géron) or the [Kaggle
House Prices](https://www.kaggle.com/learn/intermediate-machine-learning) data used in Kaggle's Intermediate ML course.
**Deliverable:** A notebook with a single `Pipeline` object that goes from raw DataFrame to prediction.

## Go deeper
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course): the "Datasets, generalization and overfitting" module, for a quick second pass.
- [The Hundred-Page ML Book](https://themlbook.com/), Ch. 1–2: a very fast overview if you want the big picture in one sitting.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 03: Data, features & evaluation](../../toolbox/03-data-features-evaluation.md), for every concept in this lesson, with alternatives.
- **Papers:** [ML in production](../../papers/11-ml-in-production.md). Start with the ⭐ ones.
- **Implement it yourself:** [from-scratch ladder](../../exercises/from-scratch-ladder.md), rungs 1, 7.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
- **Playbook:** [no-model baselines](../../playbook/05-outside-the-box.md#1-the-no-model-baselines) · [debugging tabular ML](../../playbook/06-debugging-playbook.md#a-tabular-and-classical-ml).
