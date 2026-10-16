import numpy as np
import pandas as pd

from prodma.modeling.ablation import FeatureSetBundle, compare_feature_sets, feature_set_columns

EXPECTED_SETS = {"endogenous_only", "calendar", "weather", "calendar_weather", "socioeconomic", "all_eligible"}
CALENDAR = ["feriado", "dia_1"]
WEATHER = ["temp_climatologica_c", "temperatura_c"]
SOCIOECONOMIC = ["pbi_nacional", "poblacion_ica"]


def _exogenous(index):
    columns = CALENDAR + WEATHER + SOCIOECONOMIC
    return pd.DataFrame({name: np.zeros(len(index)) for name in columns}, index=index)


def _folds(n):
    return [
        (np.arange(0, 200), np.arange(200, 260)),
        (np.arange(0, 260), np.arange(260, 320)),
        (np.arange(0, 320), np.arange(320, n)),
    ]


def test_feature_sets_cover_every_declared_group():
    index = pd.date_range("2025-01-01", periods=10, freq="15min")
    sets = feature_set_columns(_exogenous(index))
    assert set(sets) == EXPECTED_SETS
    assert sets["endogenous_only"] == []
    assert sets["calendar"] == CALENDAR
    assert sets["weather"] == WEATHER
    assert sets["calendar_weather"] == CALENDAR + WEATHER
    assert sets["socioeconomic"] == SOCIOECONOMIC


def test_all_eligible_is_the_concatenation_of_every_exogenous_column():
    index = pd.date_range("2025-01-01", periods=10, freq="15min")
    exogenous = _exogenous(index)
    sets = feature_set_columns(exogenous)
    assert sets["all_eligible"] == list(exogenous.columns)
    assert set(sets["calendar"]) | set(sets["weather"]) | set(sets["socioeconomic"]) == set(exogenous.columns)


def test_ablation_reports_every_feature_set_on_every_fold():
    rng = np.random.default_rng(3)
    n = 400
    y = pd.Series(rng.normal(size=n))
    endogenous = pd.DataFrame({"lag_1": y.shift(1).fillna(0)})
    result = compare_feature_sets(
        FeatureSetBundle(endogenous, y), _exogenous(y.index), _folds(n), minimum_improvement=0.01
    )
    assert set(result.fold_metrics["feature_set"]) == EXPECTED_SETS
    assert len(result.fold_metrics) == 3 * len(EXPECTED_SETS)


def test_socioeconomic_signal_selects_its_group_and_keeps_its_columns():
    rng = np.random.default_rng(11)
    n = 400
    pbi = rng.normal(size=n)
    y = pd.Series(4 * pbi + rng.normal(scale=0.1, size=n))
    endogenous = pd.DataFrame({"lag_1": y.shift(1).fillna(0)})
    exogenous = _exogenous(y.index)
    exogenous["pbi_nacional"] = pbi
    result = compare_feature_sets(FeatureSetBundle(endogenous, y), exogenous, _folds(n), minimum_improvement=0.01)
    assert result.selected == "socioeconomic"
    assert "pbi_nacional" in result.selected_columns
    assert "temp_climatologica_c" not in result.selected_columns
