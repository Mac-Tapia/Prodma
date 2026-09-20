import numpy as np
import pandas as pd

from prodma.limpieza.descriptive import describe_variable
from prodma.limpieza.electrical import ElectricalLimits, detect_frozen_sensor, electrical_flags
from prodma.limpieza.outliers import detect_causal_outliers


def test_outlier_decision_does_not_change_when_future_changes():
    idx = pd.date_range("2025-01-01", periods=800, freq="15min")
    base = pd.Series(100 + np.sin(np.arange(800) / 10), index=idx)
    base.iloc[700] = 500
    changed = base.copy()
    changed.iloc[701:] = 10_000
    a = detect_causal_outliers(base, window=672, threshold=6)
    b = detect_causal_outliers(changed, window=672, threshold=6)
    pd.testing.assert_series_equal(a.is_outlier.loc[:idx[700]], b.is_outlier.loc[:idx[700]])
    assert a.is_outlier.iloc[700]


def test_electrical_flags_detect_impossible_apparent_power():
    frame = pd.DataFrame({"P": [100.0], "Q": [80.0], "S_medida": [90.0]})
    flags = electrical_flags(frame, ElectricalLimits())
    assert bool(flags.loc[0, "S_incoherente"])


def test_frozen_sensor_requires_configured_run_length():
    s = pd.Series([4.0] * 20 + [5.0])
    result = detect_frozen_sensor(s, min_run=16, tolerance=1e-9)
    assert result.iloc[:20].all() and not result.iloc[20]


def test_descriptive_counts_observed_imputed_and_missing():
    frame = pd.DataFrame({"observado": [1.0, np.nan, np.nan], "contexto": [1.0, 1.0, np.nan],
                          "es_atipico": [False, False, False], "es_recortado": [False, False, False]})
    result = describe_variable(frame)
    assert result["observados"] == 1
    assert result["imputados"] == 1
    assert result["faltantes_contexto"] == 1
