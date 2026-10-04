# PY-03: pandas & Data Wrangling

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Python | ~7 h | L1→L2 | PY-02 |

## Why this matters
Most of an ML project's code, and most of its bugs, live in data preparation: joins that silently duplicate rows, group statistics that include the row being predicted,
time features that peek into the future, and `SettingWithCopyWarning`s that hide a no-op. Fluent, *safe* pandas is the difference between a model you can trust and one that
only looks good in the notebook.

## Learning goals
By the end you can:
- **Explain** the Index and alignment: why operations between Series match on labels, not positions.
- **Select** data precisely with `.loc`/`.iloc`/boolean masks, and avoid chained assignment.
- **Use** split-apply-combine: `groupby` with `agg` (one row per group) vs `transform` (one row per input row) vs `apply`.
- **Combine** tables with `merge` safely (`validate=`, `indicator=`), and reshape with `pivot_table`/`melt`.
- **Build** time-aware features (per-entity `shift`, past-only rolling windows) that don't leak the target.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 0 | **Primer** | [Study notes: pandas & data wrangling](../../notes/python/03-pandas-data-wrangling.md) | Index alignment, selection, groupby agg vs transform, merges and their failure modes, reshaping, time features without leakage, with runnable examples. Read it first. | ~1 h |
| 1 | **Intuition** | [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) | The whole page, typing along: a map of the API before the details | 30 min |
| 2 | **Read** | [*Python for Data Analysis*, 3rd ed.](https://wesmckinney.com/book/) (McKinney, the creator of pandas; free online) | Ch. 5 "Getting Started with pandas", Ch. 8 "Data Wrangling: Join, Combine, and Reshape", and Ch. 10 "Data Aggregation and Group Operations" | 2.5 h |
| 3 | **Read** | [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) | "Group by: split-apply-combine" (the `transform` and `filter` sections) and "Merge, join, concatenate and compare" (the `validate` argument) | 45 min |
| 4 | **Build** | [Kaggle Learn: pandas](https://www.kaggle.com/learn/pandas) + [Lab 02: pandas wrangling](../../labs/02-pandas-wrangling/README.md) | The Kaggle exercises for grouping/sorting and renaming/combining, then make Lab 02's tests pass | 1.5 h |

**Notes for the learner:** after every `merge`, check the row count. After every feature built with `groupby`, ask "does this value use the current row's target, or any later row?"
Those two habits prevent most tabular leakage ([CORE-07](../core-ml/07-feature-engineering-pipelines-leakage.md) comes back to them).

## Check your understanding
1. `s1 = pd.Series([1, 2], index=["a", "b"])` and `s2 = pd.Series([10, 20], index=["b", "c"])`. What is `s1 + s2`, and why?
2. What is the difference between `df.groupby("g")["x"].agg("mean")` and `.transform("mean")`? When do you need each one?
3. A left join of 1,000 orders onto a customer table returns 1,180 rows. What happened, and which argument would have raised an error instead?
4. Why is `df[df.a > 0]["b"] = 1` a bug, and what do you write instead?
5. You build "the customer's average purchase" as a feature to predict their next purchase. How can that leak, and how do you compute it safely?
6. When would you `melt` a table, and when would you `pivot_table` it?
7. *(debug)* After a cleaning step, your model's validation score jumps from 0.71 to 0.93. Name two pandas-level causes you'd check first, and how.

## Mini-project
**Task:** turn your running dataset (Course 0) from raw files into one tidy modeling table: typed columns, documented missing-value handling, at least one safe merge
(with `validate=`), two group-aggregate features and two past-only time features per entity, and an `assert` for every assumption (unique keys, no future data, row counts).
**Dataset:** your Course 0 dataset, or a public bike-sharing or taxi-trip dataset with a separate stations/zones table to join.
**Deliverable:** a `clean.py` module plus a short notebook that prints the final table's schema and the result of every check.

## Go deeper
- [*Python Data Science Handbook*](https://jakevdp.github.io/PythonDataScienceHandbook/), Ch. 3 "Data Manipulation with Pandas": hierarchical indexing, pivot tables, and vectorized string and time-series operations.
- [*Python for Data Analysis*](https://wesmckinney.com/book/), Ch. 7 "Data Cleaning and Preparation" and Ch. 11 "Time Series".
- [pandas_exercises](https://github.com/guipsamora/pandas_exercises): more drills by topic.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 03: Data, features & evaluation](../../toolbox/03-data-features-evaluation.md) ("Python, NumPy & pandas foundations" and "Problem framing & data work").
- **Papers:** none.
- **Implement it yourself:** [Lab 02](../../labs/02-pandas-wrangling/README.md).
- **Drills:** [Kaggle Learn: pandas](https://www.kaggle.com/learn/pandas), [pandas_exercises](https://github.com/guipsamora/pandas_exercises) · more in [exercises/](../../exercises/README.md).
