import pandas as pd
import pytest

from prodma.evaluacion.ranking import rank_candidates


def _candidates(**overrides):
    rows = [
        {"modelo": "Ridge", "estado": "TRAINED", "mae": 1.00, "estabilidad": 0.10,
         "cobertura_error": 0.01, "complejidad": 1, "segundos": 2.0},
        {"modelo": "XGBoost", "estado": "TRAINED", "mae": 0.95, "estabilidad": 0.20,
         "cobertura_error": 0.02, "complejidad": 5, "segundos": 30.0},
        {"modelo": "LSTM", "estado": "TIMEOUT", "mae": 0.50, "estabilidad": 0.05,
         "cobertura_error": 0.00, "complejidad": 9, "segundos": 900.0},
        {"modelo": "MLP", "estado": "TRAINED", "mae": 0.90, "estabilidad": 0.30,
         "cobertura_error": 0.30, "complejidad": 4, "segundos": 10.0},
    ]
    return pd.DataFrame(rows)


def test_ranking_orders_by_accuracy_among_admissible_candidates():
    result = rank_candidates(_candidates(), coverage_tolerance=0.05)
    admitted = result.loc[result["admitido"]]
    assert list(admitted["modelo"]) == ["XGBoost", "Ridge"]
    assert list(admitted["posicion"]) == [1, 2]


def test_every_candidate_keeps_its_exclusion_cause():
    result = rank_candidates(_candidates(), coverage_tolerance=0.05).set_index("modelo")
    assert result.loc["LSTM", "causa_exclusion"] == "estado=TIMEOUT"
    assert result.loc["MLP", "causa_exclusion"] == "calibracion_fuera_de_tolerancia"
    assert result.loc["Ridge", "causa_exclusion"] == ""
    assert set(result.index) == {"Ridge", "XGBoost", "LSTM", "MLP"}


def test_tie_is_broken_by_simplicity_then_runtime_and_keeps_all_candidates():
    frame = pd.DataFrame([
        {"modelo": "Complejo", "estado": "TRAINED", "mae": 1.0, "estabilidad": 0.1,
         "cobertura_error": 0.0, "complejidad": 5, "segundos": 5.0},
        {"modelo": "Simple", "estado": "TRAINED", "mae": 1.0, "estabilidad": 0.1,
         "cobertura_error": 0.0, "complejidad": 1, "segundos": 9.0},
    ])
    result = rank_candidates(frame, coverage_tolerance=0.05)
    assert list(result["modelo"]) == ["Simple", "Complejo"]


def test_missing_metric_is_an_explicit_exclusion_not_a_silent_drop():
    frame = _candidates()
    frame.loc[frame["modelo"] == "Ridge", "mae"] = float("nan")
    result = rank_candidates(frame, coverage_tolerance=0.05).set_index("modelo")
    assert result.loc["Ridge", "causa_exclusion"] == "metrica_faltante"
    assert len(result) == 4


def test_empty_candidate_table_is_rejected():
    with pytest.raises(ValueError):
        rank_candidates(pd.DataFrame(columns=["modelo"]), coverage_tolerance=0.05)
