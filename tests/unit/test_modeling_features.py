import numpy as np
import pandas as pd

from prodma.modeling.features import make_features


def test_future_context_change_does_not_change_past_features():
    idx = pd.date_range("2025-01-01", periods=800, freq="15min")
    clean = pd.DataFrame({"contexto": np.arange(800, dtype=float), "es_imputado": False}, index=idx)
    altered = clean.copy()
    altered.loc[idx[-100]:, "contexto"] *= 100
    cutoff = idx[-101]
    pd.testing.assert_frame_equal(make_features(clean).loc[:cutoff], make_features(altered).loc[:cutoff])


def test_dynamic_features_use_only_prior_values():
    idx = pd.date_range("2025-01-01", periods=700, freq="15min")
    clean = pd.DataFrame({"contexto": np.arange(700, dtype=float), "es_imputado": False}, index=idx)
    row = make_features(clean).loc[idx[672]]
    assert row["lag_1"] == 671
    assert row["lag_672"] == 0
    assert row["media_4"] == np.mean([668, 669, 670, 671])
