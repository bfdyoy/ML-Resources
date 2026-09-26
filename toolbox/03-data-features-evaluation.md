# Toolbox 03: Data, Features & Evaluation

[← Toolbox](README.md) · Guided version: [CORE-01](../lessons/core-ml/01-ml-workflow-end-to-end.md), [CORE-03](../lessons/core-ml/03-classification-and-metrics.md), [CORE-04](../lessons/core-ml/04-generalization-validation-regularization.md), [CORE-07](../lessons/core-ml/07-feature-engineering-pipelines-leakage.md)

## Problem framing & data work
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| Framing a problem as ML (or not) | [Google: Problem Framing](https://developers.google.com/machine-learning/problem-framing) | [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml) (rules 1–3) | Write a one-page framing doc for a problem of your own |
| EDA & data wrangling | [Jay Alammar: Visual intro to pandas](https://jalammar.github.io/gentle-visual-intro-to-data-analysis-python-pandas/) | [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) (pandas chapter) | [pandas exercises](https://github.com/guipsamora/pandas_exercises) |
| Missing values & imputation | [Kaggle Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) (Missing Values) | [sklearn: Imputation](https://scikit-learn.org/stable/modules/impute.html) · [FES](https://bookdown.org/max/FES) (missing data chapter) | Compare mean, iterative, and indicator-based imputation with CV |
| Categorical encoding (one-hot, ordinal, target) | [Kaggle Feature Engineering](https://www.kaggle.com/learn/feature-engineering) (Target Encoding) | [FES](https://bookdown.org/max/FES) (encoding categorical predictors) · [CatBoost paper](https://arxiv.org/abs/1706.09516) | [sklearn: Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html) incl. `TargetEncoder` |
| Scaling, transforms, pipelines | [sklearn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | [sklearn: Pipelines & composite estimators](https://scikit-learn.org/stable/modules/compose.html) | Build one `ColumnTransformer` pipeline end to end |
| Feature creation & selection | [Kaggle Feature Engineering](https://www.kaggle.com/learn/feature-engineering) | [FES](https://bookdown.org/max/FES) · [sklearn: Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) | Ablation table of engineered features |
| Data leakage | [Kaggle Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) (Data Leakage) | [sklearn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Plant a leak, then catch it |
| Label errors & data-centric AI | [MIT Data-Centric AI course](https://dcai.csail.mit.edu/) (Lecture 1) | DCAI lectures on label errors, class imbalance, outliers, and distribution shift | [dcai-lab exercises](https://github.com/dcai-course/dcai-lab) · [cleanlab](https://github.com/cleanlab/cleanlab) |
| Human-labeled data quality | — | [Lilian Weng: Thinking about High-Quality Human Data](https://lilianweng.github.io/posts/2024-02-05-human-data-quality/) | Measure inter-annotator agreement on a small labeling task |
| Imbalanced data | [MLU-Explain: Precision & Recall](https://mlu-explain.github.io/precision-recall/) | [imbalanced-learn: Common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html) · [SMOTE](https://arxiv.org/abs/1106.1813) | Compare class weights, resampling, and threshold tuning |

## Validation & model selection
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| Train/validation/test | [MLU-Explain: Train, Test & Validation](https://mlu-explain.github.io/train-test-validation/) | [ISLP](https://www.statlearning.com/) §5.1 | — |
| Cross-validation variants (k-fold, stratified, group, time series) | — | [sklearn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) (study the visualisations of each splitter) | Use `GroupKFold` on repeated-measures data |
| Bias–variance & learning curves | [MLU-Explain: Bias-Variance](https://mlu-explain.github.io/bias-variance/) | [sklearn: Validation & learning curves](https://scikit-learn.org/stable/modules/learning_curve.html) | Diagnose 3 models with learning curves |
| Double descent | [MLU-Explain: Double Descent](https://mlu-explain.github.io/double-descent/) | [Deep Double Descent](https://arxiv.org/abs/1912.02292) | Reproduce double descent with random features + ridge |
| Hyperparameter search (grid, random, Bayesian, successive halving) | [Inria MOOC: Hyperparameter tuning](https://inria.github.io/scikit-learn-mooc/) | [sklearn: Tuning hyperparameters](https://scikit-learn.org/stable/modules/grid_search.html) · [Hyperband](https://arxiv.org/abs/1603.06560) · [Bayesian optimization](https://arxiv.org/abs/1206.2944) | [Optuna](https://github.com/optuna/optuna) study with pruning |

## Metrics & calibrated predictions
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| Confusion matrix, precision/recall/F1 | [MLU-Explain: Precision & Recall](https://mlu-explain.github.io/precision-recall/) | [sklearn: Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) | Compute every metric by hand for a 2×2 matrix |
| ROC-AUC vs PR-AUC | [MLU-Explain: ROC & AUC](https://mlu-explain.github.io/roc-auc/) | [sklearn: Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) | Show ROC-AUC staying high while PR-AUC collapses under imbalance |
| Decision thresholds & cost-sensitive decisions | — | [sklearn: Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html) | `TunedThresholdClassifierCV` with a custom cost |
| Regression metrics (MAE, RMSE, MAPE, pinball) | — | [sklearn: Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) · [FPP](https://otexts.com/fpppy/) (evaluating accuracy) | Quantile regression with the pinball loss |
| Ranking metrics (NDCG, MRR, MAP@k) | — | [IR book](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) Ch. 8 | Implement NDCG@k from scratch |
| Probability calibration | — | [sklearn: Calibration](https://scikit-learn.org/stable/modules/calibration.html) · [Guo et al. 2017](https://arxiv.org/abs/1706.04599) | Reliability diagram + temperature scaling |
| Conformal prediction (uncertainty with guarantees) | — | [Angelopoulos & Bates: A Gentle Introduction](https://arxiv.org/abs/2107.07511) | [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) · [conformal-prediction notebooks](https://github.com/aangelopoulos/conformal-prediction) |

## Interpretability & fairness
| Concept | Start here | Go deeper | Practice |
|---|---|---|---|
| Permutation importance vs impurity importance | [explained.ai: Beware Default RF Importances](https://explained.ai/rf-importance/) | [sklearn: Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html) | Add a random column and see how each method ranks it |
| PDP / ICE plots | [Kaggle: ML Explainability](https://www.kaggle.com/learn/machine-learning-explainability) | [sklearn: Partial dependence](https://scikit-learn.org/stable/modules/partial_dependence.html) · [Molnar](https://christophm.github.io/interpretable-ml-book/) | PDP vs ICE on correlated features |
| SHAP, LIME | [Kaggle: ML Explainability](https://www.kaggle.com/learn/machine-learning-explainability) (SHAP lessons) | [Molnar: SHAP](https://christophm.github.io/interpretable-ml-book/shap.html) · [SHAP paper](https://arxiv.org/abs/1705.07874) · [LIME paper](https://arxiv.org/abs/1602.04938) | [shap](https://github.com/shap/shap) on a GBM |
| Fairness metrics & mitigation | [Google MLCC: Fairness](https://developers.google.com/machine-learning/crash-course/fairness) | [UDL](https://udlbook.github.io/udlbook/) Ch. 21 | Audit group metrics on UCI Adult |
| Documentation | — | [Model Cards](https://arxiv.org/abs/1810.03993) · [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) | Write a model card for every capstone |
