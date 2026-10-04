# EL-06: Anomaly & Outlier Detection

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Elective | ~6 h | L2 | CORE-06, CORE-10 |

## Why this matters
Fraud, intrusions, equipment failures, and data-pipeline bugs are all rare events with few or no labels. Anomaly detection is the toolkit
for finding them: unsupervised detectors, novelty detection, and careful evaluation when positives are scarce. It's also how you catch bad
data before it reaches your models.

## Learning goals
By the end you can:
- Tell outlier detection (contaminated training data) apart from novelty detection (clean training data).
- Explain and apply isolation forest, LOF, one-class SVM, and robust covariance (elliptic envelope).
- Use reconstruction-based detection (PCA/autoencoders), and know when it fails.
- Evaluate detectors with few labels (PR-AUC, precision@k), and set thresholds from an alert budget.
- Combine anomaly scores with rules and human review in a real workflow.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Anomaly Detection](../../notes/electives/06-anomaly-detection.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read** | [sklearn: Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html) | The whole page, including the comparison figure across detectors | 1 h |
| 2 | **Read** | [MIT DCAI](https://dcai.csail.mit.edu/) | The lecture "Class Imbalance, Outliers, and Distribution Shift" (outlier part) | 45 min |
| 3 | **Build** | [PyOD](https://github.com/yzhao062/pyod) | Benchmark 5+ detectors (IForest, LOF, OCSVM, ECOD, autoencoder) on one dataset with a common evaluation | 2 h |
| 4 | **Build** | Your choice | PCA reconstruction error as an anomaly score. Show one case where it works and one where it fails. | 1 h |

## Check your understanding
1. Why does isolation forest isolate anomalies in fewer splits?
2. LOF is "local": what failure of global methods does it fix?
3. Why is ROC-AUC misleading when anomalies are 0.1% of the data, and what do you report instead?
4. How do you choose a threshold when analysts can review only 50 alerts a day?
5. Why can an autoencoder trained on contaminated data learn to reconstruct the anomalies too?
6. *(debug)* Your detector flags mostly weekend transactions. What's going on, and how do you fix the features?

## Mini-project
**Task:** On a credit-card-fraud-style dataset, compare 4 detectors using precision@k for your alert budget, and compare them with a supervised GBM trained on
the few labels you have. Write up when you'd use each.
**Deliverable:** A comparison table and a precision@k curve.

## Go deeper
- [EL-01 Time Series](01-time-series-forecasting.md): forecast-residual methods for detecting anomalies in time series.
- [PROD-03 Testing, Monitoring & Drift](../production/03-testing-monitoring-drift.md): the "is my data weird?" version of this problem.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 12: Specialized topics](../../toolbox/12-specialized-topics.md) (anomaly & outlier detection).
- **Papers:** [Interpretability, uncertainty & responsible ML](../../papers/10-interpretability-uncertainty-responsible.md) (conformal prediction gives principled anomaly thresholds).
- **Implement it yourself:** an isolation forest (random splits + path length) in NumPy. Compare its scores with sklearn's.
- **Drills:** more in [exercises/](../../exercises/README.md).
