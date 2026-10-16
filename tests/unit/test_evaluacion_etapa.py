import numpy as np
import pandas as pd

from prodma.evaluacion.etapa import evaluate_series


def _predictions(models, blocks=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for horizon in (24, 96):
        for fold in (0, 1):
            for position in range(blocks // 2):
                stamp = f"2026-01-{position + 1:02d}T{fold:02d}:00"
                y = 100 + rng.normal()
                for model, error_scale in models.items():
                    rows.append({
                        "timestamp": stamp, "modelo": model, "trial": 0, "fold": fold,
                        "horizonte": horizon, "y": y,
                        "prediccion": y + error_scale * rng.normal(scale=1.0),
                    })
    return pd.DataFrame(rows)


def test_evaluation_ranks_every_candidate_and_keeps_the_best_by_error():
    predictions = _predictions({"Ridge": 0.5, "SeasonalNaive": 3.0})
    trained = pd.DataFrame([
        {"modelo": "Ridge", "estado": "ENTRENADO", "parameters": "{\"alpha\": 1.0}"},
        {"modelo": "SeasonalNaive", "estado": "ENTRENADO", "parameters": "{}"},
    ])
    times = pd.DataFrame([{"modelo": "Ridge", "fase": "tuning_y_refit", "segundos": 2.0},
                          {"modelo": "SeasonalNaive", "fase": "tuning_y_refit", "segundos": 1.0}])
    prefilter = pd.DataFrame([{"modelo": "Ridge", "estado": "APROBADO"},
                              {"modelo": "LSTM", "estado": "RECHAZADO"}])
    result = evaluate_series(predictions, trained, times, prefilter, coverage_tolerance=0.05)

    ranking = result["ranking"].set_index("modelo")
    assert set(ranking.index) == {"Ridge", "SeasonalNaive", "LSTM"}
    assert ranking.loc["LSTM", "estado"] == "PREFILTER_REJECTED"
    assert ranking.loc["Ridge", "posicion"] == 1
    assert ranking.loc["Ridge", "calibracion"] == "NO_EVALUADA"


def test_pairwise_inference_is_reported_against_the_best_model():
    predictions = _predictions({"Ridge": 0.5, "SeasonalNaive": 3.0, "Other": 2.0}, blocks=60, seed=3)
    trained = pd.DataFrame([{"modelo": m, "estado": "ENTRENADO", "parameters": "{}"}
                            for m in ("Ridge", "SeasonalNaive", "Other")])
    times = pd.DataFrame([{"modelo": m, "fase": "tuning_y_refit", "segundos": 1.0}
                          for m in ("Ridge", "SeasonalNaive", "Other")])
    result = evaluate_series(predictions, trained, times, pd.DataFrame(columns=["modelo", "estado"]),
                             coverage_tolerance=0.05)
    inference = result["inferencia"]
    assert inference["mejor_modelo"] == "Ridge"
    assert set(inference["pairwise"]["modelo"]) == {"SeasonalNaive", "Other"}
    assert inference["bloques"] == 120  # 2 horizontes x 2 folds x 30 posiciones, todas alineadas


def test_misaligned_blocks_are_counted_not_silently_dropped():
    predictions = _predictions({"Ridge": 0.5, "SeasonalNaive": 3.0}, blocks=20)
    predictions = predictions.drop(predictions.index[0])
    trained = pd.DataFrame([{"modelo": m, "estado": "ENTRENADO", "parameters": "{}"}
                            for m in ("Ridge", "SeasonalNaive")])
    times = pd.DataFrame([{"modelo": m, "fase": "tuning_y_refit", "segundos": 1.0}
                          for m in ("Ridge", "SeasonalNaive")])
    result = evaluate_series(predictions, trained, times, pd.DataFrame(columns=["modelo", "estado"]),
                             coverage_tolerance=0.05)
    assert result["inferencia"]["bloques_no_alineados"] == 1


def test_mcs_keeps_the_best_model_and_eliminates_the_worst():
    predictions = _predictions({"Ridge": 0.5, "SeasonalNaive": 6.0, "Other": 0.6}, blocks=60, seed=9)
    trained = pd.DataFrame([{"modelo": m, "estado": "ENTRENADO", "parameters": "{}"}
                            for m in ("Ridge", "SeasonalNaive", "Other")])
    times = pd.DataFrame([{"modelo": m, "fase": "tuning_y_refit", "segundos": 1.0}
                          for m in ("Ridge", "SeasonalNaive", "Other")])
    result = evaluate_series(predictions, trained, times, pd.DataFrame(columns=["modelo", "estado"]),
                             coverage_tolerance=0.05)
    mcs = result["inferencia"]["mcs"]
    assert "Ridge" in mcs
    assert "SeasonalNaive" not in mcs
