# Course 1: Core ML, a 14-week syllabus

[← Courses](README.md) · Path: [Path 1: Core ML Practitioner](../paths/01-core-ml-practitioner.md) · Labs: [index](../labs/README.md) · Cards: [core-ml deck](flashcards/README.md)

> **Pace** ≈ 8 h/week for 14 weeks: 12 teaching weeks and 2 review weeks. **Before you start:** [Course 0](00-python-for-ml.md),
> or pass its self-checks. **Running project:** one tabular dataset of *your* choosing. It starts as a baseline in week 1 and becomes the capstone.

Every week uses the [weekly loop](README.md#32-the-weekly-loop-about-8-hours): **warm-up → notes → lesson → lab → self-check from memory → explain it back → project milestone**.
"Warm-up" means drawing cards from the [flashcard deck](flashcards/README.md) for the lessons named, plus answering the interleaving question.

## How this course is shaped

- **Whole game first** (fast.ai, Kaggle Learn). In week 1 you make a complete, honest pipeline and a submission *before* any theory.
  Every later week makes one part of that pipeline better and explains why it works.
- **Build each algorithm once** (CS231n, CS336). The labs implement linear and logistic regression, metrics, CV splits, trees, k-means/PCA and conformal prediction.
  Each comes with tests, and then you go back to `scikit-learn` knowing what it does.
- **Deploy early** (ML Zoomcamp). The midterm in week 6 puts your model behind an HTTP endpoint, long before the production course.
- **Strategy alongside algorithms** (*Machine Learning Yearning*). From week 4, each week has a [playbook](../playbook/README.md) reading on *which* method to pick and *what to try next*.

---

## Week 1: The whole game
- **Do:** [CORE-01](../lessons/core-ml/01-ml-workflow-end-to-end.md) with its [notes](../notes/core-ml/01-ml-workflow-end-to-end.md).
- **Project:** choose your dataset (open data from your domain, or a closed Kaggle competition). Write a one-paragraph framing (target, unit of prediction, metric, what a 10% improvement is worth). Submit a **dummy baseline** and **one real model** in a leak-free `Pipeline`.
- **Playbook:** ["No-model" baselines](../playbook/05-outside-the-box.md#1-the-no-model-baselines) (15 min).
- **Explain it back:** why the test set is touched exactly once.

## Week 2: Linear models & gradient descent
- **Warm-up:** CORE-01 cards. *Interleave:* "Your validation score beats the training score. Name two reasons."
- **Math, just in time:** [MATH-01](../lessons/math/01-linear-algebra.md) blocks A–B and [MATH-02](../lessons/math/02-calculus-optimization.md) blocks A–B.
- **Do:** [CORE-02](../lessons/core-ml/02-linear-models-gradient-descent.md).
- **Lab:** [Lab 03: Linear regression three ways](../labs/03-linear-regression/README.md) (closed form, gradient descent, ridge).
- **Project:** add a linear model with standardized features, and read its coefficients. Does any sign surprise you?

## Week 3: Classification & metrics
- **Warm-up:** CORE-02 cards (1 week ago) + CORE-01 cards. *Interleave:* "Why does gradient descent need standardized features but the normal equation doesn't?"
- **Math:** [MATH-03](../lessons/math/03-probability-statistics.md) blocks A–B.
- **Do:** [CORE-03](../lessons/core-ml/03-classification-and-metrics.md).
- **Lab:** [Lab 04: Logistic regression & metrics](../labs/04-logistic-regression-metrics/README.md) (loss, gradient, confusion matrix, ROC-AUC by ranking, cost-optimal threshold).
- **Project:** pick the metric your problem *really* needs, and write down the cost of each error type. Choose the threshold from those costs.

## Week 4: Generalization, validation & regularization
- **Warm-up:** CORE-03 + CORE-01 cards. *Interleave:* "Accuracy is 0.97 on a dataset with 3% positives. What do you report instead?"
- **Do:** [CORE-04](../lessons/core-ml/04-generalization-validation-regularization.md).
- **Lab:** [Lab 05](../labs/05-trees-and-cross-validation/README.md), part A (k-fold and group k-fold splitters written by hand).
- **Playbook:** [Debugging: "my CV score and my leaderboard disagree"](../playbook/06-debugging-playbook.md#a-tabular-and-classical-ml).
- **Project:** switch to the right CV scheme for your data (grouped? time-ordered?). Note how much the score changes.

## Week 5: Trees & ensembles
- **Warm-up:** CORE-04 + CORE-02 cards. *Interleave:* "Ridge shrinks coefficients and trees ignore feature scale. Which preprocessing would you remove for a GBM?"
- **Do:** [CORE-05](../lessons/core-ml/05-trees-and-ensembles.md).
- **Lab:** [Lab 05](../labs/05-trees-and-cross-validation/README.md), part B (Gini, best split).
- **Playbook:** [Choosing algorithms: linear vs trees vs kNN](../playbook/01-choosing-algorithms.md).
- **Project:** a GBM with early stopping. You now have three model families under the same CV.

## Week 6: Review week + **midterm**
- **Interleaved quiz:** 20 cards drawn from CORE-01…05, shuffled. Re-answer any you miss tomorrow.
- **Redo from a blank file:** Lab 04's `roc_auc` and Lab 05's `best_split`, without looking.
- **Midterm (see the rubric below):** serve your best model with FastAPI or Gradio, with a one-page write-up.

## Week 7: Unsupervised learning
- **Warm-up:** CORE-05 + CORE-03 cards. *Interleave:* "A random forest's out-of-bag score is close to which kind of validation?"
- **Math:** MATH-01 block C (eigen/SVD).
- **Do:** [CORE-06](../lessons/core-ml/06-unsupervised-learning.md).
- **Lab:** [Lab 06: k-means & PCA via SVD](../labs/06-kmeans-pca/README.md).
- **Project:** cluster your *errors*, not your data. Do the misclassified rows form a segment? ([playbook: an error model for slice discovery](../playbook/05-outside-the-box.md#5-train-a-model-on-your-models-errors)).

## Week 8: Feature engineering, pipelines & leakage
- **Warm-up:** CORE-06 + CORE-04 cards. *Interleave:* "PCA before a GBM: when would it help, and when would it hurt?"
- **Do:** [CORE-07](../lessons/core-ml/07-feature-engineering-pipelines-leakage.md).
- **Lab:** [Lab 05](../labs/05-trees-and-cross-validation/README.md), part C (out-of-fold target encoding).
- **Playbook:** [Tabular tricks](../playbook/02-tabular-tricks.md) §1–§4 (adversarial validation, OOF encodings, the noise-feature null, target transforms).
- **Project:** a feature ablation table. Add an **adversarial validation** check between train and test.

## Week 9: Interpretability & responsible ML
- **Warm-up:** CORE-07 + CORE-05 cards. *Interleave:* "Your target-encoded feature is the top SHAP feature, and the test score fell. Hypothesis?"
- **Do:** [CORE-08](../lessons/core-ml/08-interpretability-and-responsible-ml.md).
- **Project:** a SHAP summary, three local explanations, and one fairness slice. Write down which explanation surprised you.

## Week 10: Kernels, SVMs, nearest neighbours & GPs
- **Warm-up:** CORE-08 + CORE-06 cards. *Interleave:* "Permutation importance with two highly correlated features: what goes wrong?"
- **Do:** [CORE-09](../lessons/core-ml/09-kernels-svms-nearest-neighbours.md).
- **Playbook:** [Random projections & the JL lemma](../playbook/05-outside-the-box.md#7-random-projections-are-almost-free-dimensionality-reduction), and [kNN features](../playbook/02-tabular-tricks.md#6-nearest-neighbour-features).
- **Project:** try kNN-based features in your GBM. Did they help under CV?

## Week 11: Data-centric ML
- **Warm-up:** CORE-09 + CORE-07 cards. *Interleave:* "An RBF-SVM and kNN both fail on raw, unscaled features. Why the same failure?"
- **Do:** [CORE-10](../lessons/core-ml/10-data-centric-ml.md).
- **Project:** a label audit. Find your 20 most suspicious labels with out-of-fold probabilities and look at them yourself.

## Week 12: Uncertainty: calibration & conformal
- **Warm-up:** CORE-10 + CORE-08 cards. *Interleave:* "Oversampling the minority class changes predicted probabilities. In which direction, and how do you undo it?"
- **Math:** MATH-03 block D.
- **Do:** [CORE-11](../lessons/core-ml/11-uncertainty-calibration-conformal.md).
- **Lab:** [Lab 12](../labs/12-calibration-conformal-drift/README.md), parts A–B (ECE, split conformal).
- **Project:** calibrated probabilities *or* conformal intervals with measured coverage.

## Week 13: Hyperparameter optimization
- **Warm-up:** CORE-11 + CORE-09 cards. *Interleave:* "Your conformal sets cover 90% overall but 70% on one group. What would you change?"
- **Do:** [CORE-12](../lessons/core-ml/12-hyperparameter-optimization.md).
- **Playbook:** [Ensembling: hill climbing and rank averaging](../playbook/02-tabular-tricks.md#8-blend-by-hill-climbing-on-out-of-fold-predictions).
- **Project:** one tuning run with a fixed budget, logged. A blend of your best two models, built from OOF predictions.

## Week 14: Review week + **capstone**
- **Interleaved quiz:** 30 cards from the whole deck.
- **Capstone:** finish, write up, and (optionally) swap it with a peer for review using the rubric.

---

## Midterm rubric (week 6)

| Criterion | 0: missing | 1: partial | 2: solid |
|---|---|---|---|
| Framing | No metric rationale | Metric named | Metric tied to error costs, with a baseline number |
| Honest validation | Random split on grouped/time data | Correct CV, test set reused | Correct CV scheme, test set touched once |
| Model comparison | One model | Several, different protocols | ≥3 families under identical CV, with spread (± std) |
| Served model | None | Runs locally by hand | `curl` or a UI returns a prediction; input schema validated |
| Write-up | None | Results only | Results, what didn't work, and the next experiment |

Pass: ≥ 7/10.

## Capstone rubric (week 14)

The capstone is the [Path 1 capstone](../paths/01-core-ml-practitioner.md#capstone). Score each 0–2.

| Criterion | What "2" looks like |
|---|---|
| Problem & data | A framing doc, data audit, and label-quality check with examples |
| Validation | A CV scheme matching the deployment split; adversarial validation run; nested or held-out tuning |
| Modeling | Baseline → ≥3 families → tuned search → a feature ablation, with a justified final choice |
| Decision layer | Threshold from costs; calibrated probabilities or conformal sets with measured coverage |
| Understanding | SHAP global + local explanations, plus a fairness or risk slice |
| Error analysis | Errors grouped into slices, with a next step for each |
| Reproducibility | One command reruns everything; seeds fixed; environment pinned |
| Communication | A README a stranger can follow, and a model card |

Pass: ≥ 12/16. **Distinction:** ≥ 14/16, with a short "what I'd do with two more weeks" section.

## Assessment summary

| Component | Weight | Why it's there |
|---|---|---|
| Labs passing (`pytest`) | 20% | Proves the from-scratch understanding |
| Weekly self-checks from memory (honest self-grading) | 15% | Retrieval practice |
| Midterm | 25% | An early whole-game checkpoint |
| Capstone | 40% | Transfer to a problem of your own |
