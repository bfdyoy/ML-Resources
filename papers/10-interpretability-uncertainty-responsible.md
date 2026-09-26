# Papers: Interpretability, Uncertainty & Responsible ML

[← Papers library](README.md) · Background: [CORE-08](../lessons/core-ml/08-interpretability-and-responsible-ml.md), [CORE-11](../lessons/core-ml/11-uncertainty-calibration-conformal.md), [EL-07](../lessons/electives/07-mechanistic-interpretability.md) · [Toolbox: Data, features & evaluation](../toolbox/03-data-features-evaluation.md), [Toolbox: Specialized](../toolbox/12-specialized-topics.md)

## Explaining predictions
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| ["Why Should I Trust You?" (LIME)](https://arxiv.org/abs/1602.04938) | 2016 | Local surrogate explanations. | L1 | CORE-08 |
| [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874) | 2017 | ⭐ Shapley values unify many attribution methods. | L2 | CORE-08 |
| [Axiomatic Attribution for Deep Networks (Integrated Gradients)](https://arxiv.org/abs/1703.01365) | 2017 | Attribution for neural nets, derived from axioms. | L2 | DL-04 |
| [Grad-CAM](https://arxiv.org/abs/1610.02391) | 2016 | Class activation heatmaps for CNNs. | L2 | DL-04 |
| [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292) | 2018 | ⭐ Many saliency maps don't depend on the model at all. Be skeptical. | L2 | DL-04 |

## Mechanistic interpretability
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | 2021 | ⭐ The residual stream, QK/OV circuits, induction heads. | L3 | EL-07 |
| [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) | 2022 | Why individual neurons are polysemantic. | L3 | EL-07 |

## Uncertainty & calibration
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | 2017 | ⭐ Deep nets are overconfident. Temperature scaling fixes a lot of it. | L2 | CORE-11 |
| [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511) | 2021 | ⭐ Distribution-free prediction intervals with guarantees, and working code. | L2 | CORE-11 |

## Documentation & responsible ML
| Paper | Year | Why read it | Level | After |
|---|---|---|---|---|
| [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) | 2018 | ⭐ The standard way to document a model. Use it in every capstone. | L1 | CORE-08 |
| [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) | 2018 | The dataset counterpart of model cards. | L1 | CORE-08 |
