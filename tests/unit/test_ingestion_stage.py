from pathlib import Path

import pytest

from prodma.stages.ingestion import discover_historical_files, fingerprint_input


def test_changed_bytes_invalidate_same_named_input(tmp_path):
    path = tmp_path / "demanda2023.xlsx"
    path.write_bytes(b"primera-version")
    first = fingerprint_input(path)
    path.write_bytes(b"segunda-version")
    second = fingerprint_input(path)

    assert first.sha256 != second.sha256
    assert first.path == second.path == path


def test_ambiguous_annual_file_is_rejected(tmp_path):
    (tmp_path / "demanda2023.xls").write_bytes(b"a")
    (tmp_path / "demanda2023.xlsx").write_bytes(b"b")

    with pytest.raises(ValueError, match="ambiguos"):
        discover_historical_files(tmp_path, years=(2023,))


def test_missing_annual_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="2024"):
        discover_historical_files(tmp_path, years=(2024,))


def test_discovery_returns_year_order(tmp_path):
    for year in (2024, 2023):
        (tmp_path / f"demanda{year}.xlsx").write_bytes(str(year).encode())

    found = discover_historical_files(tmp_path, years=(2023, 2024))

    assert [Path(path).name for path in found] == ["demanda2023.xlsx", "demanda2024.xlsx"]
