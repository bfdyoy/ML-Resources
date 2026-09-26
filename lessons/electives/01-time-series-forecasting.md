# EL-01: Time Series Forecasting

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Elective | ~8 h | L1→L2 | CORE-04, CORE-05 |

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
| 1 | **Read + Build** | [Forecasting: Principles and Practice, the Pythonic Way](https://otexts.com/fpppy/) | Chapters on time-series graphics, decomposition, the forecaster's toolbox (baselines, residual diagnostics, evaluating accuracy, time-series CV) | 3 h |
| 2 | **Read + Build** | [FPP, Pythonic Way](https://otexts.com/fpppy/) | Chapters on exponential smoothing and ARIMA | 3 h |
| 3 | **Read + Build** | [FPP, Pythonic Way](https://otexts.com/fpppy/) | The chapter on neural networks / ML-based forecasting (this is new to the Python edition) | 1.5 h |

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
