import numpy as np
import pandas as pd

from prodma.sanidad import calendar_profile, evaluar_sanidad


def _historia(n=96 * 400, escala=100.0, base=1000.0, semilla=0):
    idx = pd.date_range("2025-01-01 00:15", periods=n, freq="15min")
    rng = np.random.RandomState(semilla)
    return pd.Series(base + escala * np.sin(2 * np.pi * np.arange(n) / 96) + rng.normal(scale=5, size=n), index=idx)


def _referencia(s, index):  # semántica original punto a punto (celda 22 del notebook)
    train = s.dropna().tail(366 * 96)
    tab = train.groupby([train.index.month, train.index.dayofweek, train.index.hour * 4 + train.index.minute // 15]).median()
    fb = train.groupby([train.index.dayofweek, train.index.hour * 4 + train.index.minute // 15]).median()
    return np.array([tab.get((t.month, t.dayofweek, t.hour * 4 + t.minute // 15),
                             fb.get((t.dayofweek, t.hour * 4 + t.minute // 15), train.median())) for t in index])


def test_perfil_calendario_vectorizado_coincide_con_la_referencia():
    s = _historia()
    for idx in (pd.date_range("2026-02-05", "2026-02-06", freq="15min")[:-1],
                pd.date_range("2025-11-05", "2025-11-06", freq="15min")[:-1]):
        np.testing.assert_allclose(calendar_profile(s, idx), _referencia(s, idx), equal_nan=True)


def test_proyeccion_razonable_queda_ok():
    h = _historia()
    idx = pd.date_range(h.index[-1] + pd.Timedelta(minutes=15), periods=96 * 200, freq="15min")
    p = pd.Series(calendar_profile(h, idx), index=idx)
    r = evaluar_sanidad(p, h)
    assert r["estado"] == "OK" and r["marcas"] == ""


def test_proyeccion_plana_se_marca_como_aplanada():
    h = _historia()
    idx = pd.date_range(h.index[-1] + pd.Timedelta(minutes=15), periods=96 * 60, freq="15min")
    r = evaluar_sanidad(pd.Series(1000.0, index=idx), h)
    assert r["estado"] == "REVISAR" and "aplanada" in r["marcas"]


def test_desvio_de_la_media_y_fuera_de_rango():
    h = _historia()
    idx = pd.date_range(h.index[-1] + pd.Timedelta(minutes=15), periods=96 * 60, freq="15min")
    base = pd.Series(calendar_profile(h, idx), index=idx)
    r = evaluar_sanidad(base * 1.6, h)
    assert "desvio_media" in r["marcas"] and "fuera_de_rango" in r["marcas"]


def test_sin_historia_suficiente_no_inventa_un_veredicto():
    h = _historia(n=96 * 5)
    idx = pd.date_range(h.index[-1] + pd.Timedelta(minutes=15), periods=96, freq="15min")
    assert evaluar_sanidad(pd.Series(1.0, index=idx), h)["estado"] == "SIN_BASE"
