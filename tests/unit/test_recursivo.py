"""El pronóstico recursivo rápido debe reproducir EXACTAMENTE lo que daría ``features()``
del notebook (copiado aquí como referencia) aplicado paso a paso sobre el mismo buffer."""
import numpy as np
import pandas as pd
import pytest

from prodma.recursivo import (
    BUFFER_MINIMO,
    es_columna_dinamica,
    features_calendario,
    proyectar_recursivo,
    proyectar_series,
)
from prodma.transformaciones import _destransformar_y, _transformar_y


def _causal_fill(s):  # copia de la celda 10 del notebook
    s = s.replace([np.inf, -np.inf], np.nan).astype(float)
    out = s.ffill(limit=4)
    out = out.fillna(s.shift(672))
    flags = pd.Series(np.where(s.notna(), 0, np.where(out.notna(), 3, 5)), index=s.index)
    if s.first_valid_index() is not None:
        out.loc[out.index < s.first_valid_index()] = np.nan
    return out, flags


def _features_referencia(s, feriados):  # copia de la celda 12 del notebook, sin exógenas externas
    z, flags = _causal_fill(s)
    idx = z.index
    X = pd.DataFrame(index=idx)
    hora_val = idx.hour + idx.minute / 60
    for k in (1, 2, 3):
        X[f"hora_sin{k}"] = np.sin(2 * np.pi * k * hora_val / 24)
        X[f"hora_cos{k}"] = np.cos(2 * np.pi * k * hora_val / 24)
    X["anio_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    X["anio_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    for dia in range(1, 7):
        X[f"dia_{dia}"] = (idx.dayofweek == dia).astype(int)
    X["feriado"] = [int(t.date() in feriados) for t in idx]
    X["fin_semana"] = (idx.dayofweek >= 5).astype(int)
    for lag in (1, 4, 96, 672):
        X[f"lag_{lag}"] = z.shift(lag)
    for win in (4, 24, 96, 672):
        r = z.shift(1).rolling(win, min_periods=win)
        X[f"media_{win}"] = r.mean()
        X[f"std_{win}"] = r.std()
        X[f"mediana_{win}"] = r.median()
        X[f"min_{win}"] = r.min()
        X[f"max_{win}"] = r.max()
        X[f"p95_{win}"] = r.quantile(0.95)
    mismos = pd.concat([z.shift(672 * k) for k in (1, 2, 3, 4)], axis=1)
    X["media_mismo_dia_hora_4sem"] = mismos.mean(axis=1)
    X["std_mismo_dia_hora_4sem"] = mismos.std(axis=1)
    imputado = (flags > 0).astype(float)
    X["prop_imputado_96"] = imputado.shift(1).rolling(96, min_periods=1).mean()
    return X


class _ModeloLineal:
    """Modelo determinista con un coeficiente distinto por columna: cualquier columna mal
    calculada cambia la predicción."""

    def __init__(self, columnas, semilla=0):
        rng = np.random.RandomState(semilla)
        self.coef = rng.uniform(-1, 1, size=len(columnas)) * 0.03
        self.columnas = list(columnas)

    def predict(self, X):
        assert list(X.columns) == self.columnas
        return np.array([float(X.to_numpy(float)[0] @ self.coef) + 5.0])


@pytest.fixture(scope="module")
def feriados():
    import holidays

    return holidays.country_holidays("PE", years=range(2023, 2030))


def _serie(n, semilla=1):
    rng = np.random.RandomState(semilla)
    idx = pd.date_range("2026-01-01 00:15", periods=n, freq="15min")
    base = 100 + 30 * np.sin(2 * np.pi * np.arange(n) / 96) + rng.normal(scale=3, size=n)
    return pd.Series(base, index=idx)


def test_calendario_coincide_con_features_del_notebook(feriados):
    s = _serie(3000)
    ref = _features_referencia(s, feriados)
    cal = features_calendario(s.index, feriados)
    for c in cal.columns:
        np.testing.assert_allclose(cal[c].to_numpy(float), ref[c].to_numpy(float), rtol=0, atol=1e-12, err_msg=c)


def test_recursion_rapida_coincide_con_features_paso_a_paso(feriados):
    n_hist, pasos = BUFFER_MINIMO + 300, 40
    hist = _serie(n_hist)
    idx_fut = pd.date_range(hist.index[-1] + pd.Timedelta(minutes=15), periods=pasos, freq="15min")
    columnas = ["hora_sin1", "dia_3", "feriado", "lag_1", "lag_4", "lag_96", "lag_672", "media_4", "std_24",
                "mediana_96", "min_96", "max_672", "p95_672", "std_672", "media_mismo_dia_hora_4sem",
                "std_mismo_dia_hora_4sem", "prop_imputado_96", "fin_semana", "anio_cos"]
    modelo = _ModeloLineal(columnas)

    # referencia: features() del notebook sobre el buffer completo, un paso a la vez
    s = hist.copy()
    esperado = []
    for t in idx_fut:
        s.loc[t] = np.nan
        fila = _features_referencia(s, feriados).loc[[t], columnas]
        valor = float(modelo.predict(fila)[0])
        s.loc[t] = valor
        esperado.append(valor)

    cal = features_calendario(idx_fut, feriados)
    obtenido = proyectar_recursivo(modelo, hist.to_numpy(), columnas, cal)
    np.testing.assert_allclose(obtenido, esperado, rtol=1e-9, atol=1e-9)


def test_historia_corta_usa_las_semanas_disponibles_igual_que_pandas(feriados):
    hist = _serie(1500)  # menos de 4 semanas: mismo día/hora con 2 semanas, como pandas (skipna)
    idx_fut = pd.date_range(hist.index[-1] + pd.Timedelta(minutes=15), periods=10, freq="15min")
    columnas = ["lag_1", "media_mismo_dia_hora_4sem", "std_mismo_dia_hora_4sem", "media_672"]
    modelo = _ModeloLineal(columnas, semilla=3)
    s = hist.copy()
    esperado = []
    for t in idx_fut:
        s.loc[t] = np.nan
        valor = float(modelo.predict(_features_referencia(s, feriados).loc[[t], columnas])[0])
        s.loc[t] = valor
        esperado.append(valor)
    obtenido = proyectar_recursivo(modelo, hist.to_numpy(), columnas, features_calendario(idx_fut, feriados))
    np.testing.assert_allclose(obtenido, esperado, rtol=1e-9, atol=1e-9)


def test_falla_si_faltan_lags_en_vez_de_fabricar_pronostico(feriados):
    hist = np.full(100, 50.0)  # menos que la ventana de 672
    idx_fut = pd.date_range("2026-03-01", periods=3, freq="15min")
    with pytest.raises(ValueError, match="Lags incompletos"):
        proyectar_recursivo(_ModeloLineal(["media_672"]), hist, ["media_672"], features_calendario(idx_fut, feriados))


def test_columna_que_no_es_calendario_ni_dinamica_es_un_error_explicito(feriados):
    idx_fut = pd.date_range("2026-03-01", periods=3, freq="15min")
    with pytest.raises(ValueError, match="no trae las columnas"):
        proyectar_recursivo(_ModeloLineal(["inventada"]), np.ones(4000), ["inventada"],
                            features_calendario(idx_fut, feriados))


def test_es_columna_dinamica():
    assert all(es_columna_dinamica(c) for c in ["lag_96", "p95_24", "std_672", "prop_imputado_96",
                                                "media_mismo_dia_hora_4sem"])
    assert not any(es_columna_dinamica(c) for c in ["hora_sin1", "feriado", "temp_climatologica_c", "pbi_bcrp"])


def test_procesos_en_paralelo_dan_lo_mismo_que_en_serie(tmp_path, feriados):
    import joblib
    from sklearn.linear_model import Ridge

    hist = _serie(BUFFER_MINIMO + 200)
    idx_fut = pd.date_range(hist.index[-1] + pd.Timedelta(minutes=15), periods=25, freq="15min")
    columnas = ["lag_1", "lag_96", "media_24", "hora_sin1", "media_mismo_dia_hora_4sem"]
    cal = features_calendario(idx_fut, feriados)
    tareas = []
    for k in range(3):
        rng = np.random.RandomState(k)
        X = pd.DataFrame(rng.normal(size=(200, len(columnas))), columns=columnas)
        ruta = tmp_path / f"m{k}.joblib"
        joblib.dump(Ridge(alpha=1.0).fit(X, X.iloc[:, 0] * 0.5 + rng.normal(scale=0.01, size=200)), ruta)
        cols_cal = [c for c in columnas if not es_columna_dinamica(c)]
        tareas.append(dict(clave=k, ruta_modelo=ruta, columnas=columnas, historia=hist.to_numpy(),
                           calendario_columnas=cols_cal, calendario_valores=cal[cols_cal].to_numpy(float),
                           indice_futuro=idx_fut))
    serie = {k: p for k, p, _, err in proyectar_series(iter(tareas), n_jobs=1) if err is None}
    paralelo = {k: p for k, p, _, err in proyectar_series(iter(tareas), n_jobs=2) if err is None}
    assert serie.keys() == paralelo.keys() == {0, 1, 2}
    for k in serie:
        np.testing.assert_allclose(serie[k], paralelo[k], rtol=1e-12)


def test_transformacion_del_target_es_invertible_y_conserva_el_signo():
    v = np.array([-5000.0, -3.0, 0.0, 2.5, 4e4])
    np.testing.assert_allclose(_destransformar_y(_transformar_y(v)), v, rtol=1e-9)
    assert np.all(np.sign(_transformar_y(v)) == np.sign(v))


def test_un_error_en_una_serie_no_tumba_a_las_demas(tmp_path, feriados):
    hist = _serie(BUFFER_MINIMO + 100)
    idx_fut = pd.date_range(hist.index[-1] + pd.Timedelta(minutes=15), periods=5, freq="15min")
    tarea = dict(clave="sin_modelo", ruta_modelo=tmp_path / "no_existe.joblib", columnas=["lag_1"],
                 historia=hist.to_numpy(), calendario_columnas=[], calendario_valores=np.empty((5, 0)),
                 indice_futuro=idx_fut)
    (clave, pred, _seg, error), = list(proyectar_series([tarea], n_jobs=1))
    assert clave == "sin_modelo" and pred is None and error is not None
