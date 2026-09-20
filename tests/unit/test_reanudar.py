import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prodma.reanudar import (
    ESTADO_CONSTANTE,
    ESTADO_OK,
    ESTADO_SIN_DATOS,
    RestauracionInconsistente,
    cargar_estado,
    clasificar_serie,
    columnas_de_modelo,
    construir_ganadores,
    dividir_train_test,
    elegir_ganadores_ml,
    guardar_estado,
    restaurar_estado,
    verificar_config,
)
from prodma.transformaciones import _destransformar_y, _transformar_y

CFG = dict(modo="completo", seed=42, train_frac=0.8, min_train_days=2, test_days=28, cv_days=7, cv_folds=3,
           es_days=14, fecha_inicio="2023-01-01 00:15", usar_temperatura_climatologica=True, usar_pbi=True,
           usar_poblacion=True, train_days_completo=None, train_days_rapido=120)


def _y(n=1440):
    idx = pd.date_range("2026-01-01 00:15", periods=n, freq="15min")
    rng = np.random.RandomState(0)
    return pd.Series(100 + 20 * np.sin(2 * np.pi * np.arange(n) / 96) + rng.normal(scale=2, size=n), index=idx)


def _features(y):
    return pd.DataFrame({"lag_1": y.shift(1), "lag_96": y.shift(96), "lag_672": y.shift(672)})


def _metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(RMSE=float(np.sqrt(np.mean((y - p) ** 2))))


def test_dividir_train_test_es_cronologico_y_respeta_el_80_por_ciento():
    y = _y()
    X = _features(y)
    tr, te = dividir_train_test(X, y, CFG)
    n_good = int((X.notna().all(axis=1) & y.notna()).sum())
    assert tr.sum() == int(round(n_good * 0.8)) and te.sum() == n_good - tr.sum()
    assert X.index[tr].max() < X.index[te].min()
    assert not (tr & te).any()


def test_dividir_train_test_rechaza_historial_corto():
    y = _y(300)
    with pytest.raises(ValueError, match="insuficiente"):
        dividir_train_test(_features(y), y, CFG)


def test_el_ganador_operativo_nunca_es_un_benchmark():
    m = pd.DataFrame([
        dict(medidor="A", variable="P", modelo="Estacional_1", CV_RMSE=1.0),
        dict(medidor="A", variable="P", modelo="XGBoost", CV_RMSE=3.0),
        dict(medidor="A", variable="P", modelo="RandomForest", CV_RMSE=2.0),
        dict(medidor="A", variable="Q", modelo="Ridge", CV_RMSE=np.nan),
        dict(medidor="A", variable="Q", modelo="MLP", CV_RMSE=5.0),
    ])
    g = elegir_ganadores_ml(m).set_index(["medidor", "variable"])["modelo"].to_dict()
    assert g == {("A", "P"): "RandomForest", ("A", "Q"): "MLP"}


def test_clasificacion_de_series():
    ref = pd.Timestamp("2026-09-08")
    assert clasificar_serie(10.0, pd.Timestamp("2026-09-08"), ref)[0] == ESTADO_OK
    assert clasificar_serie(10.0, pd.Timestamp("2025-01-01"), ref)[0] == ESTADO_SIN_DATOS
    assert clasificar_serie(-0.0, pd.Timestamp("2026-09-08"), ref)[0] == ESTADO_CONSTANTE
    # sin datos recientes tiene prioridad sobre constante
    assert clasificar_serie(0.0, pd.Timestamp("2025-01-01"), ref)[0] == ESTADO_SIN_DATOS


def test_verificar_config_detecta_diferencias(tmp_path):
    ruta = tmp_path / "config.json"
    ruta.write_text(json.dumps(CFG), encoding="utf-8")
    verificar_config(ruta, CFG)
    with pytest.raises(ValueError, match="train_frac"):
        verificar_config(ruta, {**CFG, "train_frac": 0.7})


def test_columnas_de_modelo_para_pipeline_y_arboles():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = pd.Series(np.abs(X["a"] * 2 + rng.normal(scale=0.1, size=200)) + 1)
    ttr = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), Ridge()),
                                     func=_transformar_y, inverse_func=_destransformar_y).fit(X, y)
    assert columnas_de_modelo(ttr) == ["a", "b", "c"]
    assert columnas_de_modelo(RandomForestRegressor(n_estimators=3, random_state=0).fit(X, y)) == ["a", "b", "c"]
    import xgboost as xgb
    assert columnas_de_modelo(xgb.XGBRegressor(n_estimators=3, device="cpu").fit(X, y)) == ["a", "b", "c"]
    import lightgbm as lgb
    assert columnas_de_modelo(lgb.LGBMRegressor(n_estimators=3, verbose=-1).fit(X, y)) == ["a", "b", "c"]
    import catboost as cb
    assert columnas_de_modelo(cb.CatBoostRegressor(iterations=3, verbose=False).fit(X, y)) == ["a", "b", "c"]


@pytest.fixture()
def corrida_sintetica(tmp_path):
    """Una mini-corrida: 1 medidor, variable P, Ridge + RandomForest + un benchmark."""
    run = tmp_path / "corrida"
    (run / "modelos").mkdir(parents=True)
    (run / "tablas").mkdir()
    (run / "config.json").write_text(json.dumps(CFG), encoding="utf-8")
    y = _y()
    X = _features(y)
    tr, te = dividir_train_test(X, y, CFG)
    cols = ["lag_1", "lag_96"]
    filas = []
    for nombre, est in (("Ridge", TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), Ridge()),
                                                             func=_transformar_y, inverse_func=_destransformar_y)),
                        ("RandomForest", RandomForestRegressor(n_estimators=5, random_state=0))):
        est.fit(X.loc[tr, cols], y.loc[tr])
        joblib.dump(est, run / "modelos" / f"TOK_P_{nombre}.joblib")
        filas.append(dict(medidor="M1", variable="P", modelo=nombre, CV_RMSE=2.0 if nombre == "Ridge" else 1.5,
                          RMSE=_metrics(y.loc[te], est.predict(X.loc[te, cols]))["RMSE"]))
    filas.append(dict(medidor="M1", variable="P", modelo="Estacional_1", CV_RMSE=0.5,
                      RMSE=_metrics(y.loc[te], X.loc[te, "lag_1"])["RMSE"]))
    pd.DataFrame(filas).to_csv(run / "tablas" / "metricas.csv", index=False)
    return run, y


def _restaurar(run, y):
    return restaurar_estado(run, cfg=CFG, tokens={"M1": "TOK"}, load_meter=lambda m: pd.DataFrame({"P": y}),
                            features=_features, metrics=_metrics, imprimir=lambda *_: None)


def test_restauracion_reproduce_las_metricas_guardadas(corrida_sintetica):
    run, y = corrida_sintetica
    estado = _restaurar(run, y)
    assert set(estado["pred"]) == {("M1", "P", "Ridge"), ("M1", "P", "RandomForest"), ("M1", "P", "Estacional_1")}
    assert (estado["validacion"].dif_rel < 1e-9).all()
    assert estado["columnas"][("M1", "P", "Ridge")] == ["lag_1", "lag_96"]

    metr = pd.read_csv(run / "tablas" / "metricas.csv")
    winners = construir_ganadores(metr, estado, run, {"M1": "TOK"})
    w = winners[("M1", "P")]
    assert w["name"] == "RandomForest"  # el benchmark con menor CV_RMSE NO es el ganador operativo
    assert w["model"] is None and w["ruta_modelo"].endswith("TOK_P_RandomForest.joblib")
    assert w["estado"] == ESTADO_OK and w["last"] == y.index[-1]

    ruta = guardar_estado(run, estado, winners)
    est2, win2 = cargar_estado(run)
    assert ruta.exists() and win2[("M1", "P")]["name"] == "RandomForest"
    pd.testing.assert_frame_equal(est2["pred"][("M1", "P", "Ridge")], estado["pred"][("M1", "P", "Ridge")])


def test_restauracion_aborta_si_las_metricas_no_coinciden(corrida_sintetica):
    run, y = corrida_sintetica
    metr = pd.read_csv(run / "tablas" / "metricas.csv")
    metr.loc[metr.modelo == "Ridge", "RMSE"] *= 1.5
    metr.to_csv(run / "tablas" / "metricas.csv", index=False)
    with pytest.raises(RestauracionInconsistente, match="no reproducen"):
        _restaurar(run, y)


def test_cargar_estado_inexistente_devuelve_none(tmp_path):
    (tmp_path / "tablas").mkdir()
    assert cargar_estado(tmp_path) is None
