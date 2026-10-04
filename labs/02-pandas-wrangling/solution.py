"""Lab 02: pandas wrangling without leakage or silent row explosions.

Every function returns a NEW DataFrame (never mutate the input). No `iterrows`/`apply(axis=1)`:
use groupby-transform, merge with `validate`, and groupby-shift.
"""
import pandas as pd


def add_group_stats(df, group, col):
    """Add `<col>_group_mean` (mean of col within each group, broadcast back to every row)
    and `<col>_dev` (col minus that mean). Row count and order must not change.

    Hint: `groupby(...)[col].transform("mean")` keeps the original index; `.agg` does not.
    """
    out = df.copy()
    out[f"{col}_group_mean"] = out.groupby(group)[col].transform("mean")
    out[f"{col}_dev"] = out[col] - out[f"{col}_group_mean"]
    return out


def safe_left_merge(left, right, on):
    """Left-join `right` onto `left`. Raise pandas.errors.MergeError if `right` has duplicate keys
    (that would silently duplicate rows of `left`). Hint: one argument of `pd.merge` does this."""
    return left.merge(right, on=on, how="left", validate="many_to_one")


def lag_features(df, id_col, time_col, value_col, lags):
    """For each lag L in `lags`, add `<value_col>_lag<L>`: the value L time steps earlier FOR THE SAME ID
    (NaN when there is none). Return the frame sorted by (id_col, time_col) with a fresh 0..n-1 index.

    Subgoals: 1. sort  2. groupby(id_col)[value_col].shift(L)  3. reset the index
    Pitfall: a plain `.shift(L)` without groupby leaks one id's values into the next id's rows.
    """
    out = df.sort_values([id_col, time_col]).reset_index(drop=True)
    g = out.groupby(id_col)[value_col]
    for L in lags:
        out[f"{value_col}_lag{L}"] = g.shift(L)
    return out


def past_rolling_mean(df, id_col, time_col, value_col, window):
    """Add `<value_col>_past_mean<window>`: the mean of the previous `window` values of the same id,
    EXCLUDING the current row (so it is safe to use as a feature to predict the current value).
    Use min_periods=1, so the first row of each id is NaN and the second row equals the first value.
    Return sorted by (id_col, time_col) with a fresh index.

    Hint: shift by one inside each group first, then roll; `transform` keeps the alignment.
    """
    out = df.sort_values([id_col, time_col]).reset_index(drop=True)
    out[f"{value_col}_past_mean{window}"] = out.groupby(id_col)[value_col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    return out


def top_n_per_group(df, group, col, n):
    """The n rows with the largest `col` within each group, sorted by (group ascending, col descending),
    with a fresh index. Hint: sort once, then `groupby(...).head(n)`."""
    out = df.sort_values([group, col], ascending=[True, False])
    return out.groupby(group).head(n).reset_index(drop=True)


def fill_by_group_median(df, group, col):
    """Fill NaNs in `col` with the median of its group; if a whole group is NaN, use the global median."""
    out = df.copy()
    out[col] = out[col].fillna(out.groupby(group)[col].transform("median")).fillna(out[col].median())
    return out
