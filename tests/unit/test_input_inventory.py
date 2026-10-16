import hashlib
from pathlib import Path

from prodma.custody.inventory import (
    DOCUMENTATION,
    DUPLICATE,
    MISSING_EXPECTED,
    QUARANTINED,
    SUPPORTED,
    discover_inputs,
    write_inventory,
)


XLSX_SIGNATURE = b"PK\x03\x04" + b"\x00" * 64
XLS_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64


def _roots(tmp_path):
    roots = {name: tmp_path / name for name in ("datahistorica", "nuevosclientes", "variables_externas")}
    for root in roots.values():
        root.mkdir()
    return roots


def test_discovers_future_and_nested_files_without_year_list(tmp_path):
    roots = _roots(tmp_path)
    (roots["datahistorica"] / "demanda2030.xlsx").write_bytes(XLSX_SIGNATURE)
    nested = roots["variables_externas"] / "economia" / "2031"
    nested.mkdir(parents=True)
    (nested / "pbi.csv").write_text("fecha,valor\n2031-01,1\n", encoding="utf-8")

    inventory = discover_inputs(roots.values())

    by_name = {Path(entry.path).name: entry for entry in inventory.entries}
    assert by_name["demanda2030.xlsx"].estado == SUPPORTED
    assert by_name["pbi.csv"].estado == SUPPORTED


def test_hash_is_sha256_of_exact_bytes(tmp_path):
    roots = _roots(tmp_path)
    payload = b"fecha,valor\n2026-01-01,5\n"
    (roots["variables_externas"] / "serie.csv").write_bytes(payload)

    entry = next(e for e in discover_inputs(roots.values()).entries if e.path.name == "serie.csv")

    assert entry.sha256 == hashlib.sha256(payload).hexdigest()
    assert entry.size == len(payload)


def test_unknown_extension_is_quarantined_and_blocks_acceptance(tmp_path):
    roots = _roots(tmp_path)
    (roots["datahistorica"] / "notas.bin").write_bytes(b"\x00\x01")

    inventory = discover_inputs(roots.values())

    assert inventory.by_state()[QUARANTINED] == 1
    assert inventory.accepted is False


def test_signature_mismatch_is_quarantined(tmp_path):
    roots = _roots(tmp_path)
    (roots["datahistorica"] / "demanda2027.xlsx").write_text("no soy un xlsx", encoding="utf-8")

    inventory = discover_inputs(roots.values())

    entry = inventory.entries[0]
    assert entry.estado == QUARANTINED
    assert "firma" in entry.motivo


def test_identical_bytes_are_marked_duplicate(tmp_path):
    roots = _roots(tmp_path)
    (roots["datahistorica"] / "demanda2025.xlsx").write_bytes(XLSX_SIGNATURE)
    (roots["datahistorica"] / "copia_demanda2025.xlsx").write_bytes(XLSX_SIGNATURE)

    inventory = discover_inputs(roots.values())

    assert inventory.by_state()[SUPPORTED] == 1
    assert inventory.by_state()[DUPLICATE] == 1


def test_documentation_and_placeholders_are_visible_but_not_blocking(tmp_path):
    roots = _roots(tmp_path)
    (roots["variables_externas"] / "README.md").write_text("# docs", encoding="utf-8")
    (roots["datahistorica"] / ".gitkeep").write_bytes(b"")

    inventory = discover_inputs(roots.values())

    assert inventory.by_state()[DOCUMENTATION] == 1
    assert inventory.ignored == [str(roots["datahistorica"] / ".gitkeep")]
    assert inventory.accepted is True


def test_declared_missing_source_is_counted_not_omitted(tmp_path):
    roots = _roots(tmp_path)
    expected = [roots["variables_externas"] / "temperatura" / "no_existe.csv"]

    inventory = discover_inputs(roots.values(), expected_paths=expected)

    assert inventory.by_state()[MISSING_EXPECTED] == 1
    assert inventory.accepted is False


def test_states_reconcile_with_total_discovered_plus_missing(tmp_path):
    roots = _roots(tmp_path)
    (roots["datahistorica"] / "demanda2024.xlsx").write_bytes(XLSX_SIGNATURE)
    (roots["datahistorica"] / "demanda2024_copia.xlsx").write_bytes(XLSX_SIGNATURE)
    (roots["datahistorica"] / "x.bin").write_bytes(b"1")
    expected = [tmp_path / "falta.csv"]

    inventory = discover_inputs(roots.values(), expected_paths=expected)

    counts = inventory.by_state()
    assert sum(counts.values()) == len(inventory.entries) + len(expected)
    assert sum(counts.values()) == 4


def test_inventory_does_not_modify_originals(tmp_path):
    roots = _roots(tmp_path)
    original = roots["datahistorica"] / "demanda2023.xlsx"
    original.write_bytes(XLSX_SIGNATURE)
    before = (original.stat().st_mtime_ns, original.read_bytes())

    discover_inputs(roots.values())

    assert (original.stat().st_mtime_ns, original.read_bytes()) == before


def test_publication_writes_jsonl_summary_and_reopens(tmp_path):
    roots = _roots(tmp_path)
    (roots["datahistorica"] / "demanda2029.xlsx").write_bytes(XLSX_SIGNATURE)
    out = tmp_path / "00_custodia"

    published = write_inventory(discover_inputs(roots.values()), out)

    lines = (out / "inventario.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert published["summary"]["por_estado"] == {SUPPORTED: 1}
    assert published["summary"]["aceptado"] is True
    assert (out / "input_summary.json").exists()
