import numpy as np
import pytest

rng = np.random.default_rng(0)


def test_ece(impl):
    p = rng.random(20_000)
    y = rng.random(20_000) < p                                   # perfectly calibrated by construction
    assert impl.expected_calibration_error(y, p) < 0.02
    assert impl.expected_calibration_error(y, p ** 3) > 0.1     # overconfident-low distortion
    assert np.isclose(impl.expected_calibration_error([1, 0], [1.0, 0.0]), 0.0)   # 1.0 lands in the last bin
    assert np.isclose(impl.expected_calibration_error([0, 0, 1, 1], [0.15, 0.15, 0.15, 0.15], n_bins=10), 0.35)


def test_conformal_quantile(impl):
    scores = np.arange(1, 10)                                    # n = 9
    assert impl.conformal_quantile(scores, 0.1) == 9             # ceil(10 * 0.9) = 9th smallest
    assert impl.conformal_quantile(scores, 0.5) == 5
    assert impl.conformal_quantile(scores, 0.05) == np.inf       # ceil(10 * 0.95) = 10 > n


def test_conformal_sets_cover(impl):
    n, C = 4000, 4
    logits = rng.normal(size=(n, C)) * 2
    proba = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(C, p=p) for p in proba])
    sets = impl.conformal_prediction_sets(proba[:2000], y[:2000], proba[2000:], alpha=0.1)
    assert sets.shape == (2000, C) and sets.dtype == bool
    coverage = sets[np.arange(2000), y[2000:]].mean()
    assert 0.88 <= coverage <= 0.93


def test_conformal_interval(impl):
    x = rng.normal(size=3000)
    y = 2 * x + rng.standard_t(3, size=3000)                     # heavy-tailed noise
    pred = 2 * x
    lo, hi = impl.conformal_interval((y - pred)[:1500], pred[1500:], alpha=0.2)
    cover = np.mean((y[1500:] >= lo) & (y[1500:] <= hi))
    assert 0.77 <= cover <= 0.83
    assert np.allclose(hi - lo, (hi - lo)[0])                    # constant width with this score


def test_psi(impl):
    ref = rng.normal(size=5000)
    assert impl.psi(ref, rng.normal(size=5000)) < 0.02
    assert 0.1 < impl.psi(ref, rng.normal(0.4, 1, size=5000)) < 0.25
    assert impl.psi(ref, rng.normal(1.0, 1, size=5000)) > 0.25
    assert np.isfinite(impl.psi(ref, np.full(100, 50.0)))        # out-of-range values still binned
