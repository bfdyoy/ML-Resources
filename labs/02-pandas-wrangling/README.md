# Lab 02: pandas wrangling

[← Labs](../README.md) · Lesson: [PY-03 pandas & data wrangling](../../lessons/python/03-pandas-data-wrangling.md) · Notes: [PY-03 notes](../../notes/python/03-pandas-data-wrangling.md)

**Time** ≈ 1.5 h · **You'll practise:** `groupby().transform`, `merge(validate=...)`, per-group `shift`/`rolling`, and not mutating inputs.

| Function | The lesson it encodes |
|---|---|
| `add_group_stats` | `transform` broadcasts back to the rows; `agg` collapses them |
| `safe_left_merge` | duplicate keys silently multiply rows unless you `validate` |
| `lag_features` | shift *within* each id, or you leak one series into the next |
| `past_rolling_mean` | a feature that includes the current target is a leak |
| `top_n_per_group` | sort once, then `head(n)` per group |
| `fill_by_group_median` | hierarchical imputation with a fallback |

```bash
pytest labs/02-pandas-wrangling
```

**Bonus:** in a time-series *train/test split*, which of these functions must be fitted on the training period only, and which are safe to compute on the full table? (Hint: the [CORE-07 notes](../../notes/core-ml/07-feature-engineering-pipelines-leakage.md) on leakage.)
