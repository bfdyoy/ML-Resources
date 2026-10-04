# PY-03 notes: pandas & Data Wrangling

[← Lesson PY-03](../../lessons/python/03-pandas-data-wrangling.md) · [All notes](../README.md) · [← PY-02 notes](02-numpy-vectorized-thinking.md) · Next: [PY-04 notes →](04-eda-visualization.md)

> **Reading time** ≈ 70 min. **You need:** [PY-02 notes §2–§5](02-numpy-vectorized-thinking.md#2-views-copies-and-strides) (views vs copies, broadcasting, masks).

---

## Where we are

NumPy gave us fast, single-type arrays addressed by **position**. Real data comes as tables: named columns of different types, missing values, IDs, and timestamps.
pandas puts an **index** (labels) on top of NumPy arrays. Almost everything that's convenient about pandas, and most of its bugs, come from that index.

---

## 1. The index, and automatic alignment

Operations between Series (and DataFrames) **align on labels first**, then compute. Labels present on only one side produce `NaN`.

```python
import numpy as np
import pandas as pd

s1 = pd.Series([1, 2], index=["a", "b"])
s2 = pd.Series([10, 20], index=["b", "c"])
print((s1 + s2).to_dict())
print(s1.add(s2, fill_value=0).to_dict())

prices = pd.Series([100.0, 200.0, 300.0])
sorted_prices = prices.sort_values(ascending=False)
print((prices - sorted_prices).tolist())                          # aligned by label, NOT position
print((prices.to_numpy() - sorted_prices.to_numpy()).tolist())    # positional
```

`s1 + s2` is `{'a': nan, 'b': 12.0, 'c': nan}`: only `b` exists on both sides. `add(..., fill_value=0)` treats missing labels as 0 instead.
Subtracting a *re-sorted copy* of a Series from itself gives all zeros, because pandas realigns the labels. With `.to_numpy()` the subtraction is positional and gives `[-200.0, 0.0, 200.0]`.
Alignment is what you want most of the time. When you deliberately want positions, convert to NumPy, or call `reset_index(drop=True)` on both sides.

## 2. Selecting and assigning safely

- `df.loc[row_labels_or_mask, column_labels]`: by **label** (and boolean masks).
- `df.iloc[row_positions, column_positions]`: by **position**.
- `df[mask]` filters rows. `df["col"]` selects a column.

**Never chain selection with assignment.** `df[df.a > 0]["b"] = 1` first builds a filtered *copy*, then sets `b` on that temporary copy, and the original `df` is untouched.
Use one `.loc` call, which selects and assigns in one step on the original:

```python
import warnings

df = pd.DataFrame({"a": [-1, 2, 3], "b": [0, 0, 0]})
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    df[df.a > 0]["b"] = 1          # chained: modifies a temporary copy
print(df["b"].tolist())
df.loc[df.a > 0, "b"] = 1          # one step: modifies df
print(df["b"].tolist())
```

The chained version leaves `b` as `[0, 0, 0]`. The `.loc` version gives `[0, 1, 1]`. Older pandas warns with `SettingWithCopyWarning`.
pandas 3 uses **copy-on-write**, so chained assignment *never* works, and it warns with a `ChainedAssignmentError`. Either way, the fix is one `.loc`.

## 3. Split-apply-combine: `agg` vs `transform` vs `filter`

`groupby` splits the rows by key, applies a function per group, and combines the results. **What comes back** depends on the method:

| Method | Returns | Use for |
|---|---|---|
| `.agg("mean")` | **One row per group** | Summary tables; features to merge back by key |
| `.transform("mean")` | **One row per input row**, same index | Per-row features: group mean, deviation from it, rank within the group |
| `.filter(f)` | The rows of the groups where `f(group)` is True | "Keep stores with ≥ 30 days of data" |

```python
sales = pd.DataFrame({"store": ["a", "b", "a", "b", "a"], "units": [10, 4, 20, 6, 30]})
print(sales.groupby("store")["units"].agg("mean").to_dict())
sales["store_mean"] = sales.groupby("store")["units"].transform("mean")
sales["dev"] = sales["units"] - sales["store_mean"]
sales["rank_in_store"] = sales.groupby("store")["units"].rank(ascending=False)
print(sales.to_string(index=False))
print(sales.groupby("store").filter(lambda g: len(g) >= 3)["store"].unique().tolist())
```

`agg` gives `{'a': 20.0, 'b': 5.0}`, one value per store. `transform` broadcasts the store mean back onto each of the 5 rows, so `dev` and `rank_in_store` are ordinary columns. `filter` keeps only store `a`, which has 3 rows.

## 4. Merges: the silent row multiplier

`merge` joins tables on keys. If the right table has **duplicate keys**, each matching left row is repeated once per duplicate. No error, just more rows.
Two arguments make joins safe: `validate=` asserts the key relationship (`"one_to_one"`, `"many_to_one"`, …) and raises if it's violated,
and `indicator=True` adds a `_merge` column that tells you which rows matched.

```python
orders = pd.DataFrame({"order": [1, 2, 3, 4], "cust": ["x", "y", "x", "z"]})
customers = pd.DataFrame({"cust": ["x", "y", "y"], "tier": ["gold", "silver", "bronze"]})   # y is duplicated!

joined = orders.merge(customers, on="cust", how="left")
print(len(orders), "->", len(joined))
try:
    orders.merge(customers, on="cust", how="left", validate="many_to_one")
except pd.errors.MergeError as e:
    print("MergeError:", str(e)[:50])
dedup = customers.drop_duplicates("cust")
checked = orders.merge(dedup, on="cust", how="left", validate="many_to_one", indicator=True)
print(checked["_merge"].value_counts().to_dict())
```

The careless left join turns **4 orders into 5 rows** (order 2 now appears twice), and every sum or count computed afterwards is wrong.
With `validate="many_to_one"`, the same merge raises `MergeError` immediately. After deduplicating, `indicator=True` shows 3 matched rows and **1 order** (customer `z`) with no customer record: `{'both': 3, 'left_only': 1, 'right_only': 0}`.
**Habit:** after every merge, check `len()` and the `_merge` counts.

## 5. Reshaping: long ↔ wide

**Long** format has one row per observation (`id, variable, value`). **Wide** format has one column per variable. Plotting libraries and `groupby` like long data. Models like wide.

```python
wide = pd.DataFrame({"store": ["a", "b"], "jan": [10, 4], "feb": [12, 5]})
long = wide.melt(id_vars="store", var_name="month", value_name="units")
back = long.pivot_table(index="store", columns="month", values="units", aggfunc="sum")
print(long.shape, back.loc["a", "feb"])
```

`melt` turns the 2×3 wide table into a 4×3 long table. `pivot_table` reverses it (and aggregates duplicates with `aggfunc`, which plain `pivot` would reject).

## 6. Time features without leaking the future

To predict an entity's value at time t, a feature may only use information **available before t**. Two rules cover most cases:

1. **Shift within each entity**: `groupby(id)[col].shift(1)`. A plain `.shift(1)` would pass the last row of one entity into the first row of the next.
2. **Rolling windows over the past only**: shift by one, *then* roll, so the window excludes the current row (whose target you're predicting).

```python
ts = pd.DataFrame({"id": ["u", "u", "u", "v", "v"], "day": [1, 2, 3, 1, 2], "spend": [5.0, 7.0, 9.0, 100.0, 120.0]})
ts = ts.sort_values(["id", "day"])
ts["naive_lag"] = ts["spend"].shift(1)                                   # WRONG across ids
ts["lag1"] = ts.groupby("id")["spend"].shift(1)                         # right
ts["past_mean"] = ts.groupby("id")["spend"].transform(lambda s: s.shift(1).expanding().mean())
ts["leaky_mean"] = ts.groupby("id")["spend"].transform("mean")          # uses the future AND the current target
print(ts.to_string(index=False))
```

`naive_lag` gives user `v`'s first day the value **9.0**, which is user `u`'s last spend. `lag1` correctly gives `NaN`.
`past_mean` for `u` on day 3 is **6.0**, the mean of days 1–2. `leaky_mean` is **7.0** on every `u` row, which already includes days 2 and 3, the values being predicted.
Leaky group means like this are the classic way a validation score jumps from plausible to amazing ([CORE-07 notes](../core-ml/07-feature-engineering-pipelines-leakage.md)).
Then split by **time** as well: train on days before a cutoff and validate after it.

## 7. Types and missing values

```python
raw = pd.DataFrame({"n_items": [1, None, 3], "city": ["Cluj", "Iasi", "Cluj"], "when": ["2024-01-05", "2024-02-10", "bad"]})
print(raw["n_items"].dtype)                                          # NaN forces float64
raw["n_items"] = raw["n_items"].astype("Int64")                      # nullable integer
raw["city"] = raw["city"].astype("category")
raw["when"] = pd.to_datetime(raw["when"], errors="coerce")           # unparsable -> NaT
print(raw.dtypes.astype(str).to_dict(), raw.isna().sum().to_dict())
```

A missing value turns an integer column into `float64`. The nullable `Int64` dtype keeps integers and `<NA>`. `category` stores repeated strings once (less memory, faster `groupby`).
`to_datetime(errors="coerce")` turns unparsable dates into `NaT`, so `isna()` can count them, instead of crashing or silently keeping strings: here **1** bad date and **1** missing item count.

---

## Pitfalls & misconceptions

- **"Operations are positional."** They're aligned by label. A re-sorted or filtered Series aligns back to its original rows. Use `.to_numpy()` or `reset_index(drop=True)` when you mean positions.
- **Chained assignment** (`df[mask]["col"] = v`) modifies a temporary copy. Always use `df.loc[mask, "col"] = v`.
- **Merges that multiply rows.** Duplicate keys on the right side silently duplicate left rows. Use `validate=` and check `len()`.
- **`agg` when you needed `transform`** (or the reverse) leads to an index mismatch, or a feature full of NaNs after assignment.
- **Group statistics that include the current row or later rows** leak the target. Shift within the group, then roll or expand.
- **`apply(axis=1)` row loops** are Python loops in disguise: slow. Look for a vectorized column expression or a `groupby().transform` instead.

## Cheat sheet

| Need | pandas |
|---|---|
| Label vs position | `.loc[...]` vs `.iloc[...]` |
| Conditional assignment | `df.loc[mask, "col"] = v` |
| One row per group | `groupby(k)[c].agg(...)` |
| Group value on every row | `groupby(k)[c].transform(...)` |
| Keep whole groups | `groupby(k).filter(f)` |
| Safe join | `merge(..., validate="many_to_one", indicator=True)` + check `len()` |
| Long ↔ wide | `melt` / `pivot_table(aggfunc=...)` |
| Past-only features | `groupby(id)[c].shift(1)`, then `.rolling(w)` or `.expanding()` |
| Types | `astype("Int64")`, `astype("category")`, `pd.to_datetime(errors="coerce")` |

## Answer sketches for the lesson's self-check

<details>
<summary>1. <code>s1 + s2</code> with indexes <code>[a, b]</code> and <code>[b, c]</code></summary>

`{'a': NaN, 'b': 12, 'c': NaN}`: the indexes are aligned first, and only `b` exists in both (§1). Use `s1.add(s2, fill_value=0)` to treat missing labels as 0.
</details>

<details>
<summary>2. <code>agg("mean")</code> vs <code>transform("mean")</code></summary>

`agg` returns one value per group (indexed by the group key), for summary tables or features to merge back. `transform` returns a value for every input row, aligned to the original index, for per-row features such as the group mean or the deviation from it (§3).
</details>

<details>
<summary>3. 1,000 orders become 1,180 rows after a left join</summary>

The customer table has duplicate keys, so each order that matches a duplicated customer is repeated. `validate="many_to_one"` would raise a `MergeError` instead.
Fix the duplicates (deduplicate, or decide which record wins), then join again and check the row count (§4).
</details>

<details>
<summary>4. Why <code>df[df.a > 0]["b"] = 1</code> is a bug</summary>

It's chained assignment: `df[df.a > 0]` creates a new (copied) DataFrame, and `["b"] = 1` modifies that temporary object, so `df` doesn't change. Use `df.loc[df.a > 0, "b"] = 1` (§2).
</details>

<details>
<summary>5. "The customer's average purchase" as a feature</summary>

A mean over *all* of the customer's purchases includes the purchase being predicted (and later ones), which leaks the target. Compute it from strictly earlier rows:
sort by time, then `groupby(customer)[amount].transform(lambda s: s.shift(1).expanding().mean())`, and validate with a time-based split (§6).
</details>

<details>
<summary>6. <code>melt</code> vs <code>pivot_table</code></summary>

`melt` goes wide → long (columns become rows of `variable, value`), which suits `groupby`, plotting libraries, and tidy storage. `pivot_table` goes long → wide (one column per category, aggregating duplicates), which suits model features and human-readable summaries (§5).
</details>

<details>
<summary>7. (debug) The validation score jumps from 0.71 to 0.93 after cleaning</summary>

(1) **Target leakage through group features**: a group mean, count, or rolling window that includes the current row or future rows. Check that every group feature is computed with `shift(1)` and only from the training period (§6).
(2) **A merge that duplicated rows**, so copies of the same entity land in both train and validation. Check `len()` before and after each merge, and check for duplicate keys across the split (§4).
</details>

## Where this leads

Next: [PY-04 notes](04-eda-visualization.md). With a clean, tidy table in hand, the next step is to *look* at it: the distributions, relationships and anomalies that decide which features and which model make sense.
