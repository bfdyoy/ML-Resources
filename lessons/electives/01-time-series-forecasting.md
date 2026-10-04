# EL-01: Time Series Forecasting

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Elective | ~11 h | L1→L2 | CORE-04, CORE-05 |

## Why this matters
Demand, traffic, energy, finance: forecasting is everywhere. It also breaks the i.i.d. assumption the rest of
the curriculum relies on, so validation, features, and baselines all work differently.

## Learning goals
By the end you can:
- Decompose a series into trend, seasonality, and remainder, and read ACF plots.
- Build and beat simple baselines (naive, seasonal naive, drift). These are surprisingly hard to beat!
- Fit exponential smoothing (ETS) and ARIMA models, and evaluate them with time-series cross-validation.
- Frame forecasting as supervised learning with lag features and GBMs, and know when that works better.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: Time-Series Forecasting](../../notes/electives/01-time-series-forecasting.md) | The ideas and the math, step by step, with worked examples, runnable code, and answer sketches for the questions below. Read it first. | ~1 h |
| 1 | **Read + Build** | [Forecasting: Principles and Practice, the Pythonic Way](https://otexts.com/fpppy/) | Chapters on time-series graphics, decomposition, the forecaster's toolbox (baselines, residual diagnostics, evaluating accuracy, time-series CV) | 3 h |
| 2 | **Read + Build** | [FPP, Pythonic Way](https://otexts.com/fpppy/) | Chapters on exponential smoothing and ARIMA | 3 h |
| 3 | **Read + Build** | [FPP, Pythonic Way](https://otexts.com/fpppy/) | The chapter on neural networks / ML-based forecasting (this is new to the Python edition) | 1.5 h |
| 4 | **Build** | [Kaggle Learn: Time Series](https://www.kaggle.com/learn/time-series) | The lessons on lag features, hybrid models, and forecasting with ML (GBMs over many series) | 1.5 h |
| 5 | *Build (optional)* | [Chronos](https://github.com/amazon-science/chronos-forecasting) or [TimesFM](https://github.com/google-research/timesfm) | Zero-shot forecasts from a pretrained foundation model on your series. Compare them with your tuned models. | 1 h |

**Notes for the learner:** If the Python edition is missing a section you need, the [R original (fpp3)](https://otexts.com/fpp3/)
has the same explanations. The prose is the valuable part.

## Check your understanding
1. Why is random k-fold CV invalid for time series? Describe rolling-origin evaluation.
2. When is a seasonal naive forecast a strong baseline?
3. What does differencing do, and how do you decide how many times to difference?
4. What are the risks of using lag features with a GBM (leakage, recursive forecasting error)?
5. *(debug)* Your forecast residuals show strong autocorrelation at lag 7. What does that tell you?

## Mini-project
**Task:** Forecast daily demand 14 days ahead. Compare seasonal naive, ETS, ARIMA, and a LightGBM model with lag features,
using rolling-origin CV.
**Dataset:** Any public daily series, e.g. bike-sharing demand (UCI) or electricity load.
**Deliverable:** A notebook with an accuracy table (MASE/RMSE) per model.

## Go deeper
- [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504): a healthy dose of skepticism about deep forecasters.
- [DeepAR](https://arxiv.org/abs/1704.04110) · [N-BEATS](https://arxiv.org/abs/1905.10437) · [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363)
- [GluonTS](https://github.com/awslabs/gluonts): probabilistic deep forecasting models.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 12: Specialized topics](../../toolbox/12-specialized-topics.md), for every concept in this lesson, with alternatives.
- **Papers:** [Tabular, time series, recsys & causal](../../papers/09-tabular-timeseries-recsys-causal.md). Start with the ⭐ ones.
- **Drills:** [Deep-ML](https://www.deep-ml.com/problems) problems on this topic · more in [exercises/](../../exercises/README.md).
