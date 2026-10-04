# Lab 04: Logistic regression & metrics by hand

[← Labs](../README.md) · Lesson: [CORE-03 Classification & metrics](../../lessons/core-ml/03-classification-and-metrics.md) · Notes: [CORE-03 notes](../../notes/core-ml/03-classification-and-metrics.md)

**Time** ≈ 2 h · **You'll practise:** a stable sigmoid and loss, the `X^T (p - y)` gradient, metrics from a confusion matrix, AUC as a ranking statistic, and choosing a threshold from costs.

| Function | Checked against |
|---|---|
| `sigmoid` | no floating-point warnings at ±1000 |
| `log_loss_and_grad` | `sklearn.metrics.log_loss` and finite differences |
| `fit_logistic` | `LogisticRegression` with the matching `C = 1 / (l2 · n)` |
| `confusion`, `precision_recall_f1` | `sklearn.metrics` |
| `roc_auc` | `roc_auc_score`, including tied scores |
| `best_threshold` | brute force; close to the Bayes threshold `c_FP / (c_FP + c_FN)` |

```bash
pytest labs/04-logistic-regression-metrics
```

**Bonus:** why does `best_threshold` land *near* 0.2 but not exactly on it? (Finite sample. Which way would more data move it?)
Then rescale all scores with any increasing function. Which of your metrics change, and which don't?
