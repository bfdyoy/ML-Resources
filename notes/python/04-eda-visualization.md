# PY-04 notes: Exploratory Data Analysis & Visualization

[← Lesson PY-04](../../lessons/python/04-eda-visualization.md) · [All notes](../README.md) · [← PY-03 notes](03-pandas-data-wrangling.md) · Next: [PY-05 notes →](05-engineering-ml-code.md)

> **Reading time** ≈ 60 min. **You need:** [PY-03 notes §3](03-pandas-data-wrangling.md#3-split-apply-combine-agg-vs-transform-vs-filter) (groupby) and [§7](03-pandas-data-wrangling.md#7-types-and-missing-values) (types and missing values).

---

## Where we are

PY-03 turned raw files into a tidy table. Before modeling, we need to **understand** that table: what's in it, what's broken, and what relates to the target.
This note gives a checklist, the statistics that don't lie, the two classic traps (identical statistics, reversed associations), and how to pick a chart that answers a question.

---

## 1. The first five checks, before any plot

Most data problems show up in a one-screen audit: **shape and types, duplicates, missingness, suspicious values, and the target**.

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

rng = np.random.default_rng(0)
n = 1000
df = pd.DataFrame({
    "id": np.arange(n),
    "age": rng.integers(18, 80, n).astype(float),
    "income": rng.lognormal(10, 0.6, n),
    "country": rng.choice(["RO", "DE", "FR"], n),
    "version": "v2",                                   # a constant column
})
df.loc[rng.choice(n, 40, replace=False), "age"] = -999  # a sentinel value pretending to be data
df.loc[rng.choice(n, 80, replace=False), "income"] = np.nan
df = pd.concat([df, df.iloc[:5]], ignore_index=True)   # 5 duplicated rows

def audit(d):
    print("shape", d.shape, "| duplicated rows:", d.duplicated().sum(), "| id unique:", d["id"].is_unique)
    print("missing:", d.isna().sum()[d.isna().sum() > 0].to_dict())
    print("constant columns:", [c for c in d if d[c].nunique(dropna=False) == 1])
    num = d.select_dtypes("number")
    print("min per numeric column:", num.min().round(1).to_dict())
    print("most common value share:", {c: float(round(d[c].value_counts(normalize=True).iloc[0], 3)) for c in ["age", "country"]})

audit(df)
```

In a few lines, the audit finds every planted problem: **5 duplicated rows** (and non-unique ids), **80 missing incomes**, a **constant** `version` column, and a minimum age of **−999**, a sentinel for "unknown" that would wreck any mean or model.
None of these needs a plot. *After* the audit, plot the distributions, and every spike, gap or impossible value becomes a question for whoever produced the data.

## 2. Robust statistics: median and IQR before mean and std

The mean and standard deviation are pulled by outliers and long tails. The **median** and the **interquartile range** (IQR = Q3 − Q1) aren't.
For skewed data (incomes, prices, durations), report the median and IQR, or work on a log scale.

```python
clean_age = df.loc[df.age >= 0, "age"]
with_sentinel = df["age"]
print(f"age mean {with_sentinel.mean():.1f} vs {clean_age.mean():.1f}; median {with_sentinel.median():.1f} vs {clean_age.median():.1f}")
inc = df["income"].dropna()
print(f"income mean {inc.mean():,.0f}  median {inc.median():,.0f}  skew {stats.skew(inc):.2f}  skew of log {stats.skew(np.log(inc)):.2f}")
```

The 40 sentinel values drag the mean age from **49.4** down to **7.7**, while the median barely moves (from **50.0** to **49.0**).
The log-normal income has a mean (**25,858**) well above its median (**21,576**) and a skewness of **1.87**. After a log transform, the skewness is **0.02**, nearly symmetric.

## 3. Same statistics, different data: Anscombe's quartet

Anscombe's four small datasets (1973) have practically identical means, variances, correlation, and regression line, and look nothing alike when plotted:
one is linear with noise, one is a clean curve, one is a perfect line with a single outlier, and one is a vertical stack with a single high-leverage point.

```python
x123 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
anscombe = {
    "I":   (x123, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II":  (x123, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (x123, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV":  ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}
for name, (x, y) in anscombe.items():
    x, y = np.array(x, float), np.array(y)
    slope, intercept = np.polyfit(x, y, 1)
    print(f"{name:3s} mean_y {y.mean():.2f}  var_y {y.var(ddof=1):.2f}  corr {np.corrcoef(x, y)[0, 1]:.3f}  fit y = {intercept:.2f} + {slope:.3f} x")
```

All four print a mean of y of **7.50**, a correlation of **0.816** (0.817 for IV, a rounding difference), and the fit `y = 3.00 + 0.500 x`. The variances agree to two decimals: 4.13 vs 4.12.
The lesson is the reason EDA exists: **summary statistics compress away exactly the structure that matters**. Plot before you summarize, and plot your *residuals* after you fit.

## 4. Associations that reverse: Simpson's paradox

An association across the whole dataset can **reverse** within every subgroup when a third variable drives both. It's not a curiosity: it's the default when groups differ in size and baseline.

```python
parts = []
for g, (x0, y0) in enumerate([(2, 10), (5, 6), (8, 2)]):         # three regions with different baselines
    x = x0 + rng.normal(size=200)
    parts.append(pd.DataFrame({"region": g, "x": x, "y": y0 + 0.8 * (x - x0) + rng.normal(scale=0.5, size=200)}))
sp = pd.concat(parts, ignore_index=True)
print(f"overall corr {sp.x.corr(sp.y):+.2f}")
print("within regions:", sp.groupby("region")[["x", "y"]].apply(lambda d: round(d.x.corr(d.y), 2)).tolist())
```

Overall, x and y are strongly *negatively* correlated (**−0.80**). Within each of the three regions, the correlation is strongly *positive* (**0.86, 0.84, 0.86**).
Which answer is "right" depends on the question. Within a region, raising x goes with higher y. Across regions, region membership drives both.
In EDA, **always break key relationships down by the main segments** (region, time period, product, cohort).
Causal questions need more than plots ([EL-05](../../lessons/electives/05-causal-inference-uplift.md)).

## 5. Correlation measures linear association, not dependence

Pearson correlation only detects **linear** relationships. A perfect but U-shaped relationship can have a correlation near zero.
Spearman correlation catches any monotone relationship. Mutual information catches any dependence at all, at the cost of being noisier to estimate.

```python
x = rng.uniform(-3, 3, 2000)
y = x ** 2 + rng.normal(scale=0.5, size=2000)
mi = mutual_info_regression(x[:, None], y, random_state=0)[0]
print(f"pearson {stats.pearsonr(x, y)[0]:+.3f}  spearman {stats.spearmanr(x, y)[0]:+.3f}  mutual information {mi:.2f} nats")
```

y is almost entirely determined by x, yet the Pearson correlation is **−0.013** and the Spearman correlation **+0.008** (the relationship isn't monotone). Mutual information is clearly positive (**1.53** nats).
A correlation heatmap is a fine first screen, but "uncorrelated" doesn't mean "unrelated". Scatter plots, and models such as trees, see what correlation can't.

## 6. Choose the chart from the question

| Question | Chart | Watch out for |
|---|---|---|
| How is one variable distributed? | Histogram (try several bin widths), density, ECDF | Bins hiding spikes; log scale for skewed data |
| How do distributions compare across groups? | Boxplots, violins, overlaid ECDFs, small multiples | Boxplots hide bimodality; show the n per group |
| How do two numeric variables relate? | Scatter (+ a smoother); hexbin for many points | Overplotting; outliers driving the trend |
| How does something change over time? | Line chart, with time on x | Uneven sampling; seasonality; missing periods |
| How do parts make up a whole? | Stacked bars (few parts) | Pie charts with many slices; 3-D anything |
| Where is data missing? | A missingness heatmap or bar chart; missingness vs time | Missing-not-at-random patterns |

Use matplotlib's **object-oriented interface**: create the figure and axes explicitly, then draw on the axes. It scales from one plot to grids of small multiples:

```py
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
axes[0].hist(np.log10(df["income"].dropna()), bins=40)
axes[0].set(title="log10 income", xlabel="log10(EUR)", ylabel="count")
axes[1].hexbin(sp.x, sp.y, gridsize=30, cmap="viridis")            # overplotting fix for many points
axes[1].set(title="x vs y (all regions)")
for g, d in sp.groupby("region"):
    axes[2].scatter(d.x, d.y, s=4, alpha=0.4, label=f"region {g}")  # Simpson's paradox, made visible
axes[2].legend(); axes[2].set(title="x vs y by region")
fig.savefig("eda.png", dpi=150)
```

## 7. Plots that mislead, and the fixes

- **Truncated bar axes:** bars encode a quantity by their *length*, so a y-axis that starts at 95 exaggerates differences. Start bars at 0. A line or dot plot may use a tight range, because position, not length, carries the value (Wilke's "principle of proportional ink").
- **Overplotting:** 10⁶ points become a solid blob. Use transparency, hexbin/2-D histograms, a random sample, or contour densities.
- **Bad colour scales:** rainbow colour maps create false boundaries. Use perceptually uniform maps (viridis) for ordered data and a diverging map centred on a meaningful midpoint for signed data.
- **Dual y-axes:** two scales chosen arbitrarily can make any two series look related. Use two panels instead.
- **No n:** a group mean from 3 rows and one from 3,000 look equally solid. Show counts or intervals.

---

## Pitfalls & misconceptions

- **Plotting before auditing.** Duplicates, sentinels (−999, 0, 9999), constant columns and impossible values show up in a 10-line audit. Run it first.
- **Trusting summary statistics.** Anscombe's quartet: identical numbers, different data. Plot the data, and plot the residuals after fitting.
- **Ignoring segments.** Aggregate trends can reverse within groups (Simpson's paradox). Break key relationships down by the main segments.
- **"No correlation, so no relationship."** Pearson only sees straight lines. Look at scatter plots, Spearman, or mutual information.
- **Using the test set for EDA** leaks information into your decisions. Explore the training split only.
- **Decorative charts.** A chart without a question and a one-sentence answer doesn't belong in a report.

## Cheat sheet

| Step | What to do |
|---|---|
| Audit | Shape, dtypes, duplicates, key uniqueness, missingness, constant columns, min/max, most-frequent-value share |
| Univariate | Histogram/ECDF; median + IQR; log scale for skewed data |
| Target | Its distribution, its trend over time, its rate by segment |
| Bivariate | Scatter/hexbin with a smoother; Spearman or MI beside Pearson |
| Segments | Repeat key plots per region, period, and cohort (Simpson) |
| Plot hygiene | OO matplotlib; bars start at 0; viridis; small multiples instead of dual axes; show n |
| Output | Every plot = question + one-sentence answer + a follow-up |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Datasets with identical statistics that look different</summary>

Anscombe's quartet (§3): four datasets with the same means, variances, correlation (0.816) and regression line, but different shapes (linear, curved, an outlier, a single leverage point).
Summary statistics compress away structure. Plot the data, and the residuals.
</details>

<details>
<summary>2. Statistics and plots for a right-skewed target</summary>

Median and IQR (or quantiles) rather than mean ± std. A histogram on a log scale, or an ECDF. Consider modeling `log1p(y)` ([tabular tricks §4](../../playbook/02-tabular-tricks.md#4-transform-the-target-not-just-the-features)) (§2).
</details>

<details>
<summary>3. Negative overall, positive within every region</summary>

Simpson's paradox: region affects both variables, so the pooled association mixes the between-region differences with the within-region relationship (§4).
Report the stratified relationships, include region in the model, and treat any causal claim with the tools of EL-05.
</details>

<details>
<summary>4. Scatter plots of 1,000,000 points</summary>

The points overplot into a blob, so density and outliers become invisible and the eye is drawn to the extremes. Fixes: transparency (alpha), hexbin or a 2-D histogram, plotting a random sample, or density contours (§7).
</details>

<details>
<summary>5. A bar chart's y-axis starting at 95</summary>

It's misleading for **bars**, because their length encodes the value, so a truncated axis multiplies the visual differences. It's fine for line or dot plots, where position encodes the value and a zoomed range shows meaningful variation (with the axis clearly labeled) (§7).
</details>

<details>
<summary>6. The first five checks on a new dataset</summary>

Shape and dtypes; duplicated rows and key uniqueness; missingness per column (and over time); suspicious values (min/max, sentinels, constant columns, the most-frequent-value share); the target's distribution and definition (§1).
</details>

<details>
<summary>7. (debug) A huge spike at exactly 0, −1, or 999</summary>

(1) A **sentinel for missing or unknown** values (the system writes −1 or 999 instead of null). Confirm with the data producer, or check whether the spike co-occurs with other missing fields.
(2) A **default or fallback value** from a pipeline step (a failed join filled with 0, a sensor reset). Check whether the spike starts at a specific date or source. Either way, recode it as missing and add a "was missing" flag if it's informative (§1).
</details>

## Where this leads

Next: [PY-05 notes](05-engineering-ml-code.md). The audit functions and features from these notes deserve better than a notebook cell. The next note turns them into a tested, reproducible package.
