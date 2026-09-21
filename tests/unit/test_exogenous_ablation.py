import numpy as np
import pandas as pd

from prodma.modeling.ablation import FeatureSetBundle, compare_feature_sets


def _folds(n):
    return [
        (np.arange(0, 200), np.arange(200, 260)),
        (np.arange(0, 260), np.arange(260, 320)),
        (np.arange(0, 320), np.arange(320, n)),
    ]


def test_exogenous_set_is_rejected_when_it_does_not_improve_rolling_error():
    rng = np.random.default_rng(42)
    n = 400
    y = pd.Series(rng.normal(size=n))
    endogenous = pd.DataFrame({"lag_1": y.shift(1).fillna(0)})
    exogenous = pd.DataFrame({"feriado": rng.integers(0, 2, n), "temp_climatologica_c": rng.normal(size=n)})
    result = compare_feature_sets(FeatureSetBundle(endogenous, y), exogenous, _folds(n), minimum_improvement=0.01)
    assert result.selected == "endogenous_only"
    assert result.reason == "no_mejora_fuera_de_muestra"


def test_stable_exogenous_signal_is_selected():
    rng = np.random.default_rng(7)
    n = 400
    temperature = rng.normal(size=n)
    y = pd.Series(5 * temperature + rng.normal(scale=0.1, size=n))
    endogenous = pd.DataFrame({"lag_1": y.shift(1).fillna(0)})
    exogenous = pd.DataFrame({"feriado": np.zeros(n), "temp_climatologica_c": temperature})
    result = compare_feature_sets(FeatureSetBundle(endogenous, y), exogenous, _folds(n), minimum_improvement=0.01)
    assert result.selected == "calendar_weather"
    assert result.fold_metrics.query("feature_set == 'calendar_weather'").shape[0] == 3
