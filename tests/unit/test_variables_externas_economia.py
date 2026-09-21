import io

import numpy as np
import pandas as pd
import pytest

from prodma.variables_externas import cargar_pbi_mensual, pbi_en_fecha, poblacion_en_fecha
from prodma.exogenas.asof import asof_values, pbi_feature_frame


def _csv_pbi_temporal(tmp_path, filas):
    contenido = "serie_id,periodo,valor,fuente,fecha_descarga\n"
    for periodo, valor in filas:
        contenido += f"PN01773AM,{periodo},{valor},BCRPData,2026-09-17T00:00:00\n"
    ruta = tmp_path / "pbi.csv"
    ruta.write_text(contenido, encoding="utf-8")
    return ruta


def test_cargar_pbi_mensual_parsea_periodos_en_espanol(tmp_path):
    ruta = _csv_pbi_temporal(tmp_path, [("Ene.2023", 175.5), ("Feb.2023", 176.2), ("Dic.2023", 180.0)])
    serie = cargar_pbi_mensual(ruta)
    assert list(serie.index) == [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01"), pd.Timestamp("2023-12-01")]
    assert serie.iloc[0] == pytest.approx(175.5)
    assert serie.iloc[-1] == pytest.approx(180.0)


def test_pbi_en_fecha_usa_el_mes_correcto_y_manda_valor_real():
    pbi = pd.Series([100.0, 200.0, 300.0], index=pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]))
    idx = pd.date_range("2023-02-15", periods=3, freq="15min")  # todo dentro de febrero
    valores = pbi_en_fecha(idx, pbi)
    assert (valores == 200.0).all()


def test_pbi_en_fecha_mantiene_ultimo_valor_mas_alla_de_la_cobertura():
    pbi = pd.Series([100.0, 200.0], index=pd.to_datetime(["2023-01-01", "2023-02-01"]))
    idx_futuro = pd.date_range("2037-01-01", periods=5, freq="15min")  # muy fuera de rango
    valores = pbi_en_fecha(idx_futuro, pbi)
    assert (valores == 200.0).all(), "debe mantenerse el ultimo valor observado, no NaN ni inventar una tendencia"


def test_pbi_never_backfills_the_past_with_a_future_release():
    vintages = pd.DataFrame({
        "periodo": pd.to_datetime(["2023-01-01", "2023-02-01"]),
        "valor": [100.0, 200.0],
        "fecha_disponible": pd.to_datetime(["2023-03-15", "2023-04-15"]),
    })
    result = asof_values(pd.DatetimeIndex(["2023-02-01"]), vintages, origin=pd.Timestamp("2023-02-28"))
    assert result.isna().all()


def test_pbi_does_not_fill_dates_before_first_period():
    pbi = pd.Series([100.0, 200.0], index=pd.to_datetime(["2023-01-01", "2023-02-01"]))
    result = pbi_en_fecha(pd.DatetimeIndex(["2022-12-01"]), pbi)
    assert result.isna().all()


def test_pbi_scenario_is_disabled_unless_explicitly_supplied():
    vintages = pd.DataFrame({
        "periodo": pd.to_datetime(["2023-01-01"]), "valor": [100.0],
        "fecha_disponible": pd.to_datetime(["2023-02-15"]),
    })
    index = pd.DatetimeIndex(["2023-03-01"])
    default = pbi_feature_frame(index, vintages, origin=pd.Timestamp("2023-03-01"))
    scenario = pbi_feature_frame(index, vintages, origin=pd.Timestamp("2023-03-01"), scenario=pd.Series([110.0], index=index))
    assert "pbi_scenario" not in default
    assert scenario.loc[index[0], "pbi_scenario"] == 110.0


def test_poblacion_en_fecha_usa_el_valor_del_anio_correcto():
    poblacion = pd.Series([1_020_050, 1_041_198, 1_062_346], index=[2022, 2023, 2024])
    idx = pd.date_range("2023-06-01", periods=3, freq="15min")
    valores = poblacion_en_fecha(idx, poblacion)
    assert (valores == 1_041_198).all()


def test_poblacion_en_fecha_usa_el_extremo_mas_cercano_fuera_de_rango():
    poblacion = pd.Series([1_020_050, 1_062_346], index=[2022, 2024])
    idx_futuro = pd.date_range("2037-01-01", periods=3, freq="15min")
    valores = poblacion_en_fecha(idx_futuro, poblacion)
    assert (valores == 1_062_346).all(), "para anios posteriores al rango, se usa el ultimo disponible"
    idx_pasado = pd.date_range("2010-01-01", periods=3, freq="15min")
    valores_pasado = poblacion_en_fecha(idx_pasado, poblacion)
    assert (valores_pasado == 1_020_050).all(), "para anios anteriores al rango, se usa el primero disponible"
