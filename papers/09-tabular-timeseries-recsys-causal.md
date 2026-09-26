# Papers: Tabular ML, Time Series, Recommenders & Causal ML

[← Papers library](README.md) · Background: [CORE-05](../lessons/core-ml/05-trees-and-ensembles.md), [EL-01](../lessons/electives/01-time-series-forecasting.md), [EL-04](../lessons/electives/04-recommender-systems.md), [EL-05](../lessons/electives/05-causal-inference-uplift.md) · [Toolbox: Classical ML](../toolbox/02-classical-ml.md), [Toolbox: Specialized](../toolbox/12-specialized-topics.md)

## Tabular & boosting
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) | 2016 | ⭐ Regularized objective, split finding, and systems engineering. | L2 | CORE-05 |
| [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516) | 2017 | Ordered boosting and target statistics without leakage. | L2 | CORE-07 |
| [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) | 2022 | ⭐ A benchmark plus an analysis of inductive biases. | L1 | CORE-05 |
| [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848) | 2022 | In-context learning for small tabular datasets. | L2 | EL-10 |
| [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813) | 2002 (arXiv 2011) | The classic oversampling method. Also know when *not* to use it (it often doesn't beat reweighting). | L1 | CORE-10 |

## Hyperparameter optimization
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Practical Bayesian Optimization of ML Algorithms](https://arxiv.org/abs/1206.2944) | 2012 | ⭐ GP-based Bayesian optimization for tuning. | L2 | CORE-12 |
| [Hyperband](https://arxiv.org/abs/1603.06560) | 2016 | Early stopping as a bandit problem. | L2 | CORE-12 |
| [Optuna](https://arxiv.org/abs/1907.10902) | 2019 | Define-by-run search spaces and pruning. The paper behind the tool. | L1 | CORE-12 |

## Time-series forecasting
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [DeepAR: Probabilistic Forecasting with Autoregressive RNNs](https://arxiv.org/abs/1704.04110) | 2017 | ⭐ Global models across many related series. | L2 | EL-01 |
| [N-BEATS](https://arxiv.org/abs/1905.10437) | 2019 | A pure-MLP forecaster that won on M4. | L2 | EL-01 |
| [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363) | 2019 | Multi-horizon, interpretable, handles known-future inputs. | L3 | EL-01 |
| [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) | 2022 | ⭐ A simple linear model beats many transformers. A healthy dose of skepticism. | L1 | EL-01 |
| [A decoder-only foundation model for time-series forecasting (TimesFM)](https://arxiv.org/abs/2310.10688) | 2023 | Zero-shot forecasting with a pretrained model. | L2 | EL-01 |
| [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815) | 2024 | Tokenize values and reuse LM architectures. | L2 | EL-01 |

## Recommender systems
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) | 2016 | ⭐ Memorization + generalization, deployed on Google Play. | L2 | EL-04 |
| [DeepFM](https://arxiv.org/abs/1703.04247) | 2017 | Factorization machines + DNN for CTR prediction. | L2 | EL-04 |
| [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) | 2017 | An MLP replaces the dot product in matrix factorization. (Later work questions how much this helps.) | L2 | EL-04 |
| [Self-Attentive Sequential Recommendation (SASRec)](https://arxiv.org/abs/1808.09781) | 2018 | ⭐ Transformers for next-item prediction. | L2 | EL-04 |
| [BERT4Rec](https://arxiv.org/abs/1904.06690) | 2019 | Bidirectional masked-item modeling. | L2 | EL-04 |
| [Deep Learning Recommendation Model (DLRM)](https://arxiv.org/abs/1906.00091) | 2019 | Meta's industrial recsys architecture and its systems challenges. | L3 | EL-04 |

## Causal ML
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Double/Debiased Machine Learning for Treatment and Causal Parameters](https://arxiv.org/abs/1608.00060) | 2016 | ⭐ Using ML models to estimate causal effects without bias. | L3 | EL-05 |
| [Meta-learners for Estimating Heterogeneous Treatment Effects](https://arxiv.org/abs/1706.03461) | 2017 | S-, T-, and X-learners for uplift and CATE. | L2 | EL-05 |
