import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sales():
    return pd.DataFrame({
        "store": ["a", "b", "a", "b", "a", "b"],
        "day":   [2, 1, 1, 2, 3, 3],
        "units": [20.0, 5.0, 10.0, 7.0, 30.0, 9.0],
    })


def test_add_group_stats(impl, sales):
    before = sales.copy()
    out = impl.add_group_stats(sales, "store", "units")
    pd.testing.assert_frame_equal(sales, before)                       # input untouched
    assert len(out) == len(sales) and list(out.index) == list(sales.index)
    np.testing.assert_allclose(out["units_group_mean"], [20, 7, 20, 7, 20, 7])
    np.testing.assert_allclose(out["units_dev"], [0, -2, -10, 0, 10, 2])


def test_safe_left_merge(impl, sales):
    stores = pd.DataFrame({"store": ["a", "b"], "city": ["X", "Y"]})
    out = impl.safe_left_merge(sales, stores, "store")
    assert len(out) == len(sales) and out["city"].tolist() == ["X", "Y", "X", "Y", "X", "Y"]
    dup = pd.DataFrame({"store": ["a", "a", "b"], "city": ["X", "Z", "Y"]})
    with pytest.raises(pd.errors.MergeError):
        impl.safe_left_merge(sales, dup, "store")


def test_lag_features_do_not_cross_ids(impl, sales):
    out = impl.lag_features(sales, "store", "day", "units", [1, 2])
    assert out["store"].tolist() == ["a", "a", "a", "b", "b", "b"]
    assert list(out.index) == list(range(6))
    np.testing.assert_allclose(out["units_lag1"], [np.nan, 10, 20, np.nan, 5, 7])
    np.testing.assert_allclose(out["units_lag2"], [np.nan, np.nan, 10, np.nan, np.nan, 5])


def test_past_rolling_mean_excludes_current_row(impl, sales):
    out = impl.past_rolling_mean(sales, "store", "day", "units", 2)
    np.testing.assert_allclose(out["units_past_mean2"], [np.nan, 10, 15, np.nan, 5, 6])


def test_top_n_per_group(impl, sales):
    out = impl.top_n_per_group(sales, "store", "units", 2)
    assert out[["store", "units"]].values.tolist() == [["a", 30.0], ["a", 20.0], ["b", 9.0], ["b", 7.0]]


def test_fill_by_group_median(impl):
    df = pd.DataFrame({"g": ["x", "x", "x", "y", "z"], "v": [1.0, np.nan, 3.0, 10.0, np.nan]})
    out = impl.fill_by_group_median(df, "g", "v")
    np.testing.assert_allclose(out["v"], [1, 2, 3, 10, 3])            # z has no values -> global median of [1, 3, 10]
    assert df["v"].isna().sum() == 2
