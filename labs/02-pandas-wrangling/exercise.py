# EXERCISE: generated from solution.py by scripts/make_lab_stubs.py.
# Replace each `raise NotImplementedError` with your code, then run the tests (see README.md).
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
    raise NotImplementedError("your code here")


def safe_left_merge(left, right, on):
    """Left-join `right` onto `left`. Raise pandas.errors.MergeError if `right` has duplicate keys
    (that would silently duplicate rows of `left`). Hint: one argument of `pd.merge` does this."""
    raise NotImplementedError("your code here")


def lag_features(df, id_col, time_col, value_col, lags):
    """For each lag L in `lags`, add `<value_col>_lag<L>`: the value L time steps earlier FOR THE SAME ID
    (NaN when there is none). Return the frame sorted by (id_col, time_col) with a fresh 0..n-1 index.

    Subgoals: 1. sort  2. groupby(id_col)[value_col].shift(L)  3. reset the index
    Pitfall: a plain `.shift(L)` without groupby leaks one id's values into the next id's rows.
    """
    raise NotImplementedError("your code here")


def past_rolling_mean(df, id_col, time_col, value_col, window):
    """Add `<value_col>_past_mean<window>`: the mean of the previous `window` values of the same id,
    EXCLUDING the current row (so it is safe to use as a feature to predict the current value).
    Use min_periods=1, so the first row of each id is NaN and the second row equals the first value.
    Return sorted by (id_col, time_col) with a fresh index.

    Hint: shift by one inside each group first, then roll; `transform` keeps the alignment.
    """
    raise NotImplementedError("your code here")


def top_n_per_group(df, group, col, n):
    """The n rows with the largest `col` within each group, sorted by (group ascending, col descending),
    with a fresh index. Hint: sort once, then `groupby(...).head(n)`."""
    raise NotImplementedError("your code here")


def fill_by_group_median(df, group, col):
    """Fill NaNs in `col` with the median of its group; if a whole group is NaN, use the global median."""
    raise NotImplementedError("your code here")
