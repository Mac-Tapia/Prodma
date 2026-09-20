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


# --- perfil anclado: forma estacional x nivel reciente --------------------------------------
from prodma.sanidad import perfil_anclado  # noqa: E402


def _serie_con_crecimiento(anios=3.0, crecimiento_anual=0.30, semilla=0):
    n = int(96 * 365 * anios)
    idx = pd.date_range("2023-01-01 00:15", periods=n, freq="15min")
    rng = np.random.RandomState(semilla)
    nivel = 1000 * (1 + crecimiento_anual) ** (np.arange(n) / (96 * 365))            # crece 30 % por año
    dia = 1 + 0.4 * np.sin(2 * np.pi * (np.arange(n) % 96) / 96 - 1.5)               # ciclo diario
    mes = 1 + 0.15 * np.cos(2 * np.pi * idx.dayofyear / 365.25)                      # estacionalidad anual
    return pd.Series(nivel * dia * mes * (1 + rng.normal(scale=0.02, size=n)), index=idx)


def test_perfil_anclado_no_subestima_cuando_la_demanda_crecio():
    s = _serie_con_crecimiento()
    cut = s.index[-96 * 60]
    hist, real = s.loc[:cut], s.iloc[-96 * 59:-96 * 59 + 96 * 28]
    idx = real.index
    err_simple = np.sqrt(np.mean((real.to_numpy() - calendar_profile(hist, idx)) ** 2))
    err_anclado = np.sqrt(np.mean((real.to_numpy() - perfil_anclado(hist, idx)) ** 2))
    assert err_anclado < 0.6 * err_simple  # el perfil de 12 meses arrastra el nivel del año anterior


def test_perfil_anclado_con_poca_historia_usa_el_perfil_simple():
    s = _historia(n=96 * 200)
    idx = pd.date_range(s.index[-1] + pd.Timedelta(minutes=15), periods=96 * 7, freq="15min")
    np.testing.assert_allclose(perfil_anclado(s, idx), calendar_profile(s, idx))


def test_perfil_anclado_admite_series_con_negativos_y_es_finito():
    s = _serie_con_crecimiento(anios=2.5) - 1300  # signo mixto -> modo aditivo
    idx = pd.date_range(s.index[-1] + pd.Timedelta(minutes=15), periods=96 * 30, freq="15min")
    p = perfil_anclado(s, idx)
    assert np.isfinite(p).all() and len(p) == len(idx)


def test_perfil_anclado_no_se_dispara_en_series_casi_nulas():
    rng = np.random.RandomState(1)
    n = 96 * 800
    idx = pd.date_range("2023-01-01 00:15", periods=n, freq="15min")
    s = pd.Series(np.where(rng.rand(n) < 0.001, rng.uniform(50, 400, n), 0.0) + 1e-3, index=idx)  # ráfagas raras
    fut = pd.date_range(idx[-1] + pd.Timedelta(minutes=15), periods=96 * 60, freq="15min")
    p = perfil_anclado(s, fut)
    base = calendar_profile(s, fut)
    assert np.isfinite(p).all() and p.mean() <= 2.5 * max(base.mean(), 1e-9) + 1e-9
