# EL-01 notes: Time-Series Forecasting

[← Lesson EL-01](../../lessons/electives/01-time-series-forecasting.md) · [All notes](../README.md) · [← CV-04 notes](../vision/04-3d-vision-neural-rendering.md) · Next: [EL-02 notes →](02-bayesian-probabilistic-ml.md)

> **Reading time** ≈ 55 min. **You need:** [CORE-02 notes](../core-ml/02-linear-models-gradient-descent.md) (regression), [CORE-04 notes](../core-ml/04-generalization-validation-regularization.md) §2.3 (time-series CV), and [CORE-05 notes](../core-ml/05-trees-and-ensembles.md) (GBMs and their inability to extrapolate).

---

## Where we are

Time series break the i.i.d. assumption behind most of Path 1: today depends on yesterday, and the future may not look like the past. That changes how you **validate** (no shuffling), what you **compare against** (naive baselines that are surprisingly strong), and how you **build features** (lags, without peeking at the future).

---

## 1. Structure: trend, seasonality, remainder

An additive decomposition writes $y_t = T_t + S_t + R_t$: a slowly varying **trend**, a repeating **seasonal** pattern with period $m$ (7 for daily data with a weekly cycle, 12 for monthly data with a yearly one), and a **remainder**.
Use a multiplicative form, $y_t = T_t\times S_t\times R_t$, when the seasonal swings grow with the level, or take logs to make it additive.

**Autocorrelation (ACF):** the correlation between the series and itself shifted by $k$ steps,

```math
r_k = \frac{\sum_{t=k+1}^{n}(y_t - \bar y)(y_{t-k} - \bar y)}{\sum_{t=1}^{n}(y_t - \bar y)^2}.
```

How to read an ACF plot:

- slowly decaying spikes → a trend (non-stationary);
- spikes at $m, 2m, \dots$ → seasonality with period $m$;
- a cut-off after lag $q$ → MA-like behaviour.

**Stationarity** means the statistical properties (mean, variance, autocorrelation) don't change over time. ARIMA needs it. **Differencing** creates it:

- $y'_t = y_t - y_{t-1}$ removes a (local) linear trend;
- a seasonal difference, $y_t - y_{t-m}$, removes stable seasonality.

How many times? Usually 0–1 of each. Use unit-root tests (KPSS, ADF) or look at the ACF of the differenced series. **Over-differencing** adds noise and a negative lag-1 autocorrelation.

---

## 2. Baselines that are hard to beat

| Baseline | Forecast | Wins when |
|---|---|---|
| Naive | $\hat y_{t+h} = y_t$ | Random-walk-like series (prices) |
| Seasonal naive | $\hat y_{t+h} = y_{t+h-m}$ (the same period last cycle) | **Strong, stable seasonality**, little trend (retail daily sales, electricity load) |
| Drift | the last value + the average historical slope × $h$ | A steady trend |
| Mean | $\bar y$ | Stationary noise |

**Scale-free accuracy (MASE):** divide your MAE by the in-sample MAE of the one-step seasonal naive forecast. MASE < 1 means you beat that baseline. **If a fancy model doesn't beat seasonal naive under proper evaluation, it isn't helping.**

---

## 3. Classical models

### 3.1 Exponential smoothing (ETS)

**Simple exponential smoothing** forecasts with a weighted average whose weights decay geometrically into the past:

```math
\hat y_{t+1} = \alpha y_t + (1-\alpha)\hat y_t = \alpha\sum_{j\ge0}(1-\alpha)^j y_{t-j}.
```

A large $\alpha$ reacts fast, and a small $\alpha$ smooths more. **Holt** adds a smoothed trend, and **Holt–Winters** a smoothed seasonal component. The ETS framework adds error, trend, and seasonal types (additive or multiplicative), chosen by AIC, with proper prediction intervals.

### 3.2 ARIMA(p, d, q)

Difference $d$ times, then model the stationary series as **autoregressive** on its own past values (order $p$) plus a **moving average** of past forecast errors (order $q$):

```math
y'_t = c + \sum_{i=1}^{p}\phi_i y'_{t-i} + \sum_{j=1}^{q}\theta_j\varepsilon_{t-j} + \varepsilon_t .
```

Seasonal ARIMA adds the same structure at lag $m$. In practice, use an automatic order search (`auto_arima`, or `AutoARIMA` in statsforecast).

---

## 4. Evaluation: rolling-origin, never shuffled

Random k-fold CV puts future observations in the training set, so the model "predicts" the past from the future, which deployment never allows. That's **temporal leakage** (CORE-07 §4). Use **rolling-origin evaluation** instead:

1. Train on data up to time $t_0$, and forecast $t_0+1, \dots, t_0+H$.
2. Move the origin forward (expanding or sliding window), refit (or update), and forecast again.
3. Average the errors **by horizon $h$**. Errors grow with $h$, so report them per horizon.

Use the horizon you actually need in production. A model that's great at $h = 1$ can be poor at $h = 14$.

---

## 5. Forecasting as supervised learning (GBMs with lags)

Turn each time point into a row of features: **lags** ($y_{t-1}, y_{t-7}, \dots$), rolling statistics (the 7-day mean *of past values*), calendar features (day of week, holidays), and known-future covariates (promotions, prices).
Then fit a GBM. **Global models** trained across many related series (all the stores × products) often beat per-series ARIMA, because they share patterns.

**Risks:**

- **Leakage:** a rolling mean that includes the current day, features computed over the whole series, target-derived features without a shift. Every feature for time $t$ must use data available at $t - h$, for horizon $h$.
- **Recursive vs direct forecasting:** to forecast $h$ steps ahead, a *recursive* model feeds its own predictions back in as lags, and **errors compound**. A *direct* strategy trains a separate model (or uses horizon-specific features) for each $h$, which avoids that, at the cost of more models.
- **Trends:** GBMs can't extrapolate (CORE-05 §1.4). Model differences or ratios, or detrend first.

**Foundation models** (Chronos, TimesFM) forecast zero-shot from the history alone. They're strong baselines worth trying, but still evaluate them against seasonal naive and a tuned GBM with rolling origins.

**Debug: residual autocorrelation at lag 7.** The residuals still contain a **weekly pattern** the model didn't capture. That's information you're leaving on the table. Add weekly seasonality (a seasonal term, day-of-week features, or a lag-7 feature), and recheck the residual ACF. Good residuals look like white noise.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
rng = np.random.default_rng(0)

# Daily series: trend + weekly seasonality + noise
n = 730
t = np.arange(n)
y = 50 + 0.05 * t + 8 * np.sin(2 * np.pi * t / 7) + 4 * np.cos(2 * np.pi * t / 7 * 2) + rng.normal(0, 2, n)

def acf(x, k):
    x = x - x.mean(); return np.sum(x[k:] * x[:-k]) / np.sum(x * x)
print("ACF of y at lags 1, 7, 14:", [round(float(acf(y, k)), 2) for k in (1, 7, 14)])

def ses(series, alpha):
    level = series[0]
    for v in series[1:]: level = alpha * v + (1 - alpha) * level
    return level

H, m = 14, 7
def rolling_origin(forecaster, starts=range(500, 700, 14)):
    errs = []
    for s in starts:
        hist, fut = y[:s], y[s:s + H]
        errs.append(np.abs(forecaster(hist, H) - fut))
    return np.mean(errs, axis=0)                     # MAE per horizon

forecasters = {
    "naive":           lambda h, H: np.repeat(h[-1], H),
    "seasonal naive":  lambda h, H: np.array([h[-m + (i % m)] for i in range(H)]),
    "drift":           lambda h, H: h[-1] + (h[-1] - h[0]) / (len(h) - 1) * np.arange(1, H + 1),
    "SES (alpha=0.3)": lambda h, H: np.repeat(ses(h, 0.3), H),
}
scale = np.mean(np.abs(y[m:500] - y[:500 - m]))      # in-sample seasonal-naive MAE, for MASE
for name, f in forecasters.items():
    mae = rolling_origin(f)
    print(f"{name:16s} MAE h=1 {mae[0]:.2f}  h=14 {mae[-1]:.2f}  mean MASE {mae.mean() / scale:.2f}")
```

```python
# --- GBM with lag features (direct strategy: features available at the forecast origin) --------------------------
def make_rows(series, h):
    X, Y = [], []
    for s in range(28, len(series) - h):
        hist = series[:s]
        X.append([hist[-1], hist[-7], hist[-14], hist[-28:].mean(), (s + h) % 7, s + h])
        Y.append(series[s + h - 1] - hist[-1])      # predict the CHANGE: trees can't extrapolate levels
    return np.array(X), np.array(Y)

maes = []
for h in [1, 7, 14]:
    X, Y = make_rows(y[:500], h)
    model = HistGradientBoostingRegressor(random_state=0).fit(X, Y)
    errs = []
    for s in range(500, 700, 14):
        hist = y[:s]
        x = [[hist[-1], hist[-7], hist[-14], hist[-28:].mean(), (s + h) % 7, s + h]]
        errs.append(abs(hist[-1] + model.predict(x)[0] - y[s + h - 1]))
    maes.append(np.mean(errs))
print("GBM (direct, lag features) MAE at h=1, 7, 14:", np.round(maes, 2))
# It LOSES to seasonal naive (MAE about 2.05) here: one short series whose signal y[t-7] already captures.
# This is the usual first result, and a healthy one: GBMs earn their keep as global models across many
# related series, with covariates (promotions, prices, holidays) that naive baselines can't use.

# --- Residual ACF: a model that ignores the weekly cycle leaves lag-7 structure ---------------------------------------
trend_only = np.polyval(np.polyfit(t, y, 1), t)
resid = y - trend_only
print("residual ACF at lag 7, trend-only model:", round(float(acf(resid, 7)), 2))
X_season = np.c_[t, np.sin(2*np.pi*t/7), np.cos(2*np.pi*t/7), np.sin(4*np.pi*t/7), np.cos(4*np.pi*t/7), np.ones(n)]
resid2 = y - X_season @ np.linalg.lstsq(X_season, y, rcond=None)[0]
print("residual ACF at lag 7, with weekly terms:", round(float(acf(resid2, 7)), 2))
```

---

## Pitfalls & misconceptions

- **Shuffled CV, or features computed over the whole series.**
- **No seasonal-naive baseline.**
- **Reporting one averaged error across all horizons** when production needs a specific one.
- **GBMs on raw levels of a trending series.** They predict flat at the training maximum.
- **Ignoring prediction intervals.** Forecasts drive decisions (stock, staffing) that depend on the uncertainty.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Decomposition | $y = T + S + R$ (or multiplicative; logs) |
| ACF | $r_k = \sum(y_t-\bar y)(y_{t-k}-\bar y)/\sum(y_t-\bar y)^2$ |
| Seasonal naive | $\hat y_{t+h} = y_{t+h-m}$ |
| SES | $\hat y_{t+1} = \alpha y_t + (1-\alpha)\hat y_t$ |
| ARIMA | difference $d$×, then AR($p$) + MA($q$) |
| MASE | MAE / in-sample seasonal-naive MAE (< 1 beats the baseline) |
| Evaluation | rolling origin, errors per horizon |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why is random k-fold invalid; what is rolling-origin evaluation?</summary>

Shuffling puts future observations in training (temporal leakage) and ignores autocorrelation, so the scores are optimistic. Rolling origin trains on data up to $t_0$, forecasts the next $H$ points, moves $t_0$ forward, and repeats, reporting the errors per horizon.
</details>

<details>
<summary>2. When is seasonal naive a strong baseline?</summary>

When seasonality is strong and stable and the trend is weak: daily retail sales, energy load, web traffic. The same day last week (or year) already captures most of the signal.
</details>

<details>
<summary>3. What does differencing do, and how many times?</summary>

It removes a trend (first difference) or a stable seasonality (seasonal difference), making the series closer to stationary. Usually 0–1 of each, chosen with unit-root tests (KPSS/ADF) and by checking the ACF. Avoid over-differencing.
</details>

<details>
<summary>4. Risks of lag features with a GBM.</summary>

Leakage (features that use data after the forecast origin, or rolling windows that include the target day), compounding errors in recursive multi-step forecasting, and GBMs' inability to extrapolate trends (model changes, or detrend).
</details>

<details>
<summary>5. Residual autocorrelation at lag 7.</summary>

The residuals contain a weekly pattern the model missed. It's under-specified: add weekly seasonality, day-of-week features, or a lag-7 term (the demo shows the lag-7 residual ACF dropping once weekly terms are added).
</details>

## Where this leads

Next: [EL-02 notes](02-bayesian-probabilistic-ml.md). Forecasts need honest uncertainty. Bayesian modeling makes uncertainty a first-class output for any model, not just time series.
