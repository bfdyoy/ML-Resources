# CORE-03: Classification & Evaluation Metrics

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Core ML | ~6 h | L2 | CORE-02 · math: [MATH-03](../math/03-probability-statistics.md) |

## Why this matters
"Accuracy 95%" means nothing on a dataset that is 95% negatives. Choosing and reading the **right metric**
(precision, recall, F1, ROC-AUC, PR-AUC, calibration) is where practitioners most often go wrong, and where
you add the most value.

## Learning goals
By the end you can:
- Explain logistic regression as a linear model for log-odds, trained with cross-entropy.
- Build and read a confusion matrix. Compute precision, recall, F1, and specificity by hand.
- Choose between ROC-AUC and PR-AUC for a given class balance and business cost.
- Move the decision threshold to trade precision for recall, and justify the choice.
- Compare discriminative (logistic regression) and generative (LDA, Naive Bayes) classifiers.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Intuition** | [MLU-Explain: Logistic Regression](https://mlu-explain.github.io/logistic-regression/) | The whole essay | 20 min |
| 2 | **Read** | [ISLP](https://www.statlearning.com/) Ch. 4 "Classification" | §4.1–4.3 (logistic regression), §4.4 (LDA, QDA, Naive Bayes), §4.5 (comparison). Skim §4.6 (GLMs). | 2 h |
| 3 | **Intuition** | [MLU-Explain: Precision & Recall](https://mlu-explain.github.io/precision-recall/) then [ROC & AUC](https://mlu-explain.github.io/roc-auc/) | Both essays, back to back | 40 min |
| 4 | **Read + Build** | [Géron, *Hands-On ML*, Ch. 3 "Classification"](https://github.com/ageron/handson-mlp) | MNIST: confusion matrix, precision/recall trade-off, ROC, multiclass and multilabel. Run `03_classification.ipynb`. | 2 h |
| 5 | **Build** | [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) | Read "Metrics and scoring" (the classification metrics section) and "Probability calibration". Plot a reliability diagram. | 1 h |

**Notes for the learner:** Keep one question in mind throughout: *what does a false positive cost compared with a false negative?*
Every metric choice follows from the answer.

## Check your understanding
1. Why do we model log-odds instead of the probability directly?
2. With 1% positives, why can ROC-AUC look great while the model is useless? What metric would you report instead?
3. How does moving the threshold from 0.5 to 0.2 change precision and recall, and why?
4. When would LDA beat logistic regression?
5. What does it mean for a classifier to be *calibrated*, and when do you care?
6. *(debug)* Your fraud model has 99.2% accuracy and catches almost no fraud. Walk through how you'd diagnose and fix it.

## Mini-project
**Task:** Build a credit-default or churn classifier. Report PR-AUC, pick a threshold from a cost matrix you define,
and show a calibration plot before and after `CalibratedClassifierCV`.
**Dataset:** ISLP `Default` (`from ISLP import load_data`), or any imbalanced Kaggle churn dataset.
**Deliverable:** A notebook that ends with a table of threshold, precision, recall, and expected cost.

## Go deeper
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course): the "Logistic regression" and "Classification" modules, which have good quick exercises.
- [Probabilistic ML (Murphy)](https://probml.github.io/pml-book/) Ch. 10 "Logistic Regression": the full derivation.
