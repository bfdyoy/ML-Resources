# Path 1: Core ML Practitioner

**Goal:** Go from "I've trained a few models" to "I can take a tabular problem from framing to a trustworthy, explained model."
**Duration:** ~72 h (≈ 9–10 weeks at 7–8 h/week) · **Level:** L1→L2 · **Primary books:** ISLP (free) + Géron's notebooks (free)

> **Fast-track for intermediate learners:** Before each lesson, try its *Check your understanding* questions.
> If you can answer ≥ 80% confidently, do only the mini-project and move on.

| # | Lesson | What you'll be able to do | Time |
|---|---|---|---|
| 1 | [CORE-01 The ML Workflow, End to End](../lessons/core-ml/01-ml-workflow-end-to-end.md) | Frame a problem and run a clean, leak-free pipeline | 6 h |
| 2 | [CORE-02 Linear Models & Gradient Descent](../lessons/core-ml/02-linear-models-gradient-descent.md) | Fit, interpret, and optimize linear models; write GD from scratch | 6 h |
| 3 | [CORE-03 Classification & Evaluation Metrics](../lessons/core-ml/03-classification-and-metrics.md) | Pick the right metric and threshold for the business cost | 6 h |
| 4 | [CORE-04 Generalization: Bias-Variance, Validation & Regularization](../lessons/core-ml/04-generalization-validation-regularization.md) | Measure generalization honestly; regularize on purpose | 6 h |
| 5 | [CORE-05 Trees, Random Forests & Gradient Boosting](../lessons/core-ml/05-trees-and-ensembles.md) | Train and tune the strongest tabular models | 7 h |
| 6 | [CORE-06 Unsupervised Learning](../lessons/core-ml/06-unsupervised-learning.md) | Explore, compress, and segment unlabeled data | 6 h |
| 7 | [CORE-07 Feature Engineering, Pipelines & Leakage](../lessons/core-ml/07-feature-engineering-pipelines-leakage.md) | Engineer features that help, and catch leaks | 6 h |
| 8 | [CORE-08 Interpreting Models & Responsible ML](../lessons/core-ml/08-interpretability-and-responsible-ml.md) | Explain predictions and audit for fairness | 5 h |
| 9 | [CORE-09 Kernel Methods, SVMs, Nearest Neighbours & GPs](../lessons/core-ml/09-kernels-svms-nearest-neighbours.md) | Use margin- and similarity-based models, and GP uncertainty | 7 h |
| 10 | [CORE-10 Data-Centric ML](../lessons/core-ml/10-data-centric-ml.md) | Fix labels, handle imbalance, and learn with few labels | 7 h |
| 11 | [CORE-11 Uncertainty: Calibration & Conformal Prediction](../lessons/core-ml/11-uncertainty-calibration-conformal.md) | Make probabilities trustworthy, with guaranteed prediction sets | 5 h |
| 12 | [CORE-12 Hyperparameter Optimization](../lessons/core-ml/12-hyperparameter-optimization.md) | Tune efficiently without fooling yourself | 5 h |

**Math, just in time:** [MATH-01](../lessons/math/01-linear-algebra.md) block A–B (before CORE-02), block C (before CORE-06) ·
[MATH-02](../lessons/math/02-calculus-optimization.md) block A–B (before CORE-02) ·
[MATH-03](../lessons/math/03-probability-statistics.md) block A–B (before CORE-03), block D (before CORE-11)

## Capstone
**Build an end-to-end tabular ML project on a dataset from your own domain or interests.** Required: problem framing doc
→ label/data-quality audit → baseline → at least 3 model families compared with proper CV and a tuned search → feature engineering ablation →
threshold/metric justification → calibrated probabilities or conformal intervals → SHAP-based explanation → fairness/risk section → model card.
Publish it as a GitHub repo with a clean README. That's your portfolio piece for this path.

> **Core vs extended:** CORE-01…08 are the essentials. CORE-09…12 round out a professional toolkit, and you can take them in any order after CORE-08.

**Next:** [Path 2: Deep Learning](02-deep-learning.md) or [Path 4: ML in Production](04-ml-in-production.md).
