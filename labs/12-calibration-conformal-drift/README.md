# Lab 12: Calibration error, conformal prediction & drift (PSI)

[← Labs](../README.md) · Lessons: [CORE-11](../../lessons/core-ml/11-uncertainty-calibration-conformal.md) (parts A–B) · [PROD-03](../../lessons/production/03-testing-monitoring-drift.md) (part C)

**Time** ≈ 2 h · **You'll practise:** binning-based calibration error, the `ceil((n+1)(1−α))` rank that makes conformal's guarantee exact, and the drift statistic most monitoring tools report.

| Part | Function | Checked against |
|---|---|---|
| A | `expected_calibration_error` | ≈0 for calibrated synthetic data; large for a distorted copy; a hand value |
| B | `conformal_quantile` | hand ranks, including the "infinite set" edge case |
| B | `conformal_prediction_sets`, `conformal_interval` | empirical coverage ≈ 1 − α on held-out data |
| C | `psi` | ≈0 with no shift; moderate for a 0.4σ shift; major for a 1σ shift |

```bash
pytest labs/12-calibration-conformal-drift
```

**Bonus:**
1. Coverage holds *on average*, not per class. Measure per-class coverage for your sets, then implement class-conditional conformal (one `qhat` per class).
2. PSI depends on the binning. Compute it with 5, 10 and 50 bins for the same shift. What happens with 50 bins on 500 samples?
