import pandas as pd

from tools.validar_backends import run_backend_smoke


def test_backend_is_recorded_after_real_fit_predict(tmp_path):
    output = tmp_path / "backends_efectivos.csv"
    result = run_backend_smoke(output, seed=7)

    assert set(result.columns) >= {
        "modelo", "solicitado", "efectivo", "fit_ok", "predict_ok", "dispositivo"
    }
    assert set(result["modelo"]) == {"XGBoost", "CatBoost", "LightGBM", "LSTM"}
    assert result[["fit_ok", "predict_ok"]].all().all()
    assert result.loc[result.modelo == "LightGBM", "efectivo"].iloc[0] == "cpu"
    pd.testing.assert_frame_equal(result, pd.read_csv(output), check_dtype=False)
