import numpy as np
import pandas as pd
import pytest

from prodma.modeling.intervals import (
    CalibrationInsufficient,
    calibrated_interval,
    conditional_coverage,
    split_conformal_radius,
)


def test_radius_uses_finite_sample_correction_on_absolute_residuals():
    residuals = np.arange(1.0, 31.0)  # n = 30
    radius = split_conformal_radius(residuals, alpha=0.2)
    # rank = ceil((n+1)(1-alpha)) = ceil(31*0.8) = 25 -> 25th smallest absolute residual
    assert radius == 25.0


def test_calibration_needs_a_minimum_sample_and_fails_closed():
    with pytest.raises(CalibrationInsufficient):
        split_conformal_radius(np.array([0.1, 0.2]), alpha=0.1)


def test_empty_calibration_is_never_silently_filled():
    with pytest.raises(CalibrationInsufficient):
        split_conformal_radius(np.array([]), alpha=0.1)


def test_interval_is_symmetric_around_prediction_and_uses_only_given_residuals():
    rng = np.random.default_rng(0)
    calibration = rng.normal(scale=2.0, size=500)
    prediction = pd.Series([10.0, 20.0], index=pd.date_range("2026-01-01", periods=2, freq="h"))
    lower, upper = calibrated_interval(prediction, calibration, alpha=0.1)
    radius = split_conformal_radius(calibration, alpha=0.1)
    np.testing.assert_allclose(upper - prediction, radius)
    np.testing.assert_allclose(prediction - lower, radius)


def test_out_of_sample_coverage_meets_nominal_level_on_exchangeable_data():
    rng = np.random.default_rng(42)
    calibration = rng.normal(size=2000)
    test_residuals = rng.normal(size=4000)
    radius = split_conformal_radius(calibration, alpha=0.1)
    coverage = float(np.mean(np.abs(test_residuals) <= radius))
    assert coverage == pytest.approx(0.9, abs=0.02)


def test_conditional_coverage_is_published_per_horizon():
    index = pd.date_range("2026-01-01", periods=6, freq="15min")
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, 6.0], index=index)
    prediction = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=index)
    lower = prediction - 1.0
    upper = prediction + 1.0
    horizon = pd.Series([1, 1, 1, 2, 2, 2], index=index)
    table = conditional_coverage(y, lower, upper, horizon)
    assert list(table["horizonte"]) == [1, 2]
    assert list(table["n"]) == [3, 3]
    assert table.set_index("horizonte").loc[1, "cobertura"] == 1.0
    assert table.set_index("horizonte").loc[2, "cobertura"] == pytest.approx(2 / 3)


def test_conditional_coverage_rejects_misaligned_inputs():
    index = pd.date_range("2026-01-01", periods=3, freq="15min")
    y = pd.Series([1.0, 2.0, 3.0], index=index)
    with pytest.raises(ValueError):
        conditional_coverage(y, y - 1, y + 1, pd.Series([1, 1], index=index[:2]))
