import pandas as pd
import pytest

from prodma.external.connectors.senamhi import (
    BLOCKED_MISSING_SOURCE,
    source_status,
    parse_verified_export,
)


def test_senamhi_without_verified_export_is_an_explicit_blocked_status(tmp_path):
    status = source_status(tmp_path / "senamhi_ica.csv")

    assert status["estado"] == BLOCKED_MISSING_SOURCE
    assert "SENAMHI" in status["motivo"]


def test_verified_export_requires_station_identity_and_unit(tmp_path):
    export = tmp_path / "senamhi.csv"
    export.write_text("estacion,fecha_hora,temperatura_c\nIca,2023-01-01 00:00,21.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="codigo_estacion"):
        parse_verified_export(export)


def test_verified_export_is_labelled_observed_measurement(tmp_path):
    export = tmp_path / "senamhi.csv"
    export.write_text(
        "codigo_estacion,fecha_hora,temperatura_c,fuente_verificacion\n"
        "000000,2023-01-01 00:00,21.0,descarga_manual_verificada\n",
        encoding="utf-8",
    )

    frame = parse_verified_export(export)

    assert frame["calidad"].iloc[0] == "medicion_directa"
    assert frame["fecha_hora"].iloc[0] == pd.Timestamp("2023-01-01 00:00")
