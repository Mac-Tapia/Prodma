from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from prodma.custody.authorized import (
    AuthorizedCatalogError,
    build_authorized_catalog,
    write_authorized_catalog,
)


REAL_CATALOG = Path(__file__).parents[2] / "medidoralimentador.xlsx"


def _write_catalog(path: Path, data_rows, sheet="alimentador_medidor"):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append([None, None, None, None])
    ws.append(["codigo alimentador", "Medidor", "Nivel Tension", "siglas"])
    for row in data_rows:
        ws.append(list(row))
    wb.save(path)
    return path


def test_identifiers_keep_hyphens_and_duplicates_are_traced(tmp_path):
    source = _write_catalog(tmp_path / "medidoralimentador.xlsx", [
        ("LT", "L-6605", 60, None),
        ("LT", "L6605", 60, None),
        ("SI112", " L-SI112 ", 10, "SI"),
        ("SI112", "L-SI112", 10, "SI"),
    ])

    catalog = build_authorized_catalog(source)

    identifiers = set(catalog.table["identificador"])
    assert identifiers == {"L-6605", "L6605", "L-SI112"}
    dup = catalog.table.set_index("identificador").loc["L-SI112"]
    assert dup["filas_origen"] == [5, 6]
    assert catalog.summary["filas_origen"] == 4
    assert catalog.summary["identificadores_unicos"] == 3
    assert catalog.summary["duplicados_exactos"] == 1


def test_conflicting_duplicate_blocks_authorization(tmp_path):
    source = _write_catalog(tmp_path / "medidoralimentador.xlsx", [
        ("SI112", "L-SI112", 10, "SI"),
        ("SI213", "L-SI112", 22.9, "SI"),
    ])

    with pytest.raises(AuthorizedCatalogError, match="conflicto"):
        build_authorized_catalog(source)


def test_missing_sheet_is_rejected(tmp_path):
    source = _write_catalog(tmp_path / "medidoralimentador.xlsx", [("LT", "L-1", 60, None)], sheet="otra")

    with pytest.raises(AuthorizedCatalogError, match="alimentador_medidor"):
        build_authorized_catalog(source)


def test_missing_medidor_header_is_rejected(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "alimentador_medidor"
    ws.append(["codigo alimentador", "Equipo"])
    ws.append(["LT", "L-1"])
    path = tmp_path / "medidoralimentador.xlsx"
    wb.save(path)

    with pytest.raises(AuthorizedCatalogError, match="Medidor"):
        build_authorized_catalog(path)


def test_publication_reopens_with_hash_and_reconciliation(tmp_path):
    source = _write_catalog(tmp_path / "medidoralimentador.xlsx", [
        ("LT", "L-1", 60, None),
        ("LT", "L-1", 60, None),
        ("LT", "L-2", 60, None),
    ])
    out = tmp_path / "00_custodia"

    published = write_authorized_catalog(build_authorized_catalog(source), out)

    reopened = pd.read_parquet(published["parquet"])
    assert sorted(reopened["identificador"]) == ["L-1", "L-2"]
    assert published["sha256"]
    assert published["summary"]["filas_origen"] == 3


@pytest.mark.skipif(not REAL_CATALOG.exists(), reason="catálogo real no disponible")
def test_real_catalog_reconciles_every_source_row_read_only():
    before = REAL_CATALOG.stat().st_mtime_ns

    catalog = build_authorized_catalog(REAL_CATALOG)

    summary = catalog.summary
    assert summary["filas_origen"] == summary["identificadores_unicos"] + summary["duplicados_exactos"]
    assert summary["identificadores_unicos"] == len(catalog.table)
    assert REAL_CATALOG.stat().st_mtime_ns == before
    # Snapshot actual del catálogo (no es una constante permanente del dominio).
    assert summary["filas_origen"] == 112
    assert summary["identificadores_unicos"] == 110
