import numpy as np
import pandas as pd

from prodma.series import resolver_duplicados


def _referencia(df):  # el método original del notebook: agrupa TODAS las marcas
    return df.groupby(level=0).agg(lambda v: v.dropna().iloc[0] if v.dropna().nunique() == 1 else np.nan)


def _df_con_duplicados():
    idx = pd.date_range("2023-12-31 22:00", periods=12, freq="15min", name="Fecha")
    df = pd.DataFrame({"P": np.arange(12, dtype=float), "Q": np.arange(12, dtype=float) * 2,
                       "S": np.full(12, np.nan)}, index=idx)
    # empalme entre dos libros anuales: 3 marcas repetidas con casos distintos
    repetidas = pd.DataFrame({"P": [3.0, 4.0, 5.0], "Q": [6.0, 999.0, np.nan], "S": [np.nan, np.nan, np.nan]},
                             index=idx[3:6])  # Q: coincide / conflicto / uno nulo
    df.loc[idx[5], "Q"] = 10.0
    return pd.concat([df, repetidas]).sort_index(kind="stable")


def test_resultado_identico_al_metodo_original():
    df = _df_con_duplicados()
    esperado, obtenido = _referencia(df), resolver_duplicados(df)
    pd.testing.assert_frame_equal(obtenido, esperado, check_dtype=True, check_exact=True)


def test_conflicto_o_solo_nulos_dan_nan_y_coincidencia_conserva_el_valor():
    r = resolver_duplicados(_df_con_duplicados())
    assert r.index.is_unique and r.index.is_monotonic_increasing
    idx = r.index
    assert r.loc[idx[3], "Q"] == 6.0          # los dos coinciden
    assert np.isnan(r.loc[idx[4], "Q"])       # valores distintos: conflicto
    assert r.loc[idx[5], "Q"] == 10.0         # uno nulo y otro válido: el válido
    assert r["S"].isna().all()                # solo nulos


def test_sin_duplicados_devuelve_el_mismo_marco():
    df = pd.DataFrame({"P": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2, freq="15min"))
    assert resolver_duplicados(df) is df
