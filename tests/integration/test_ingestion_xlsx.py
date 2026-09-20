import json
from pathlib import Path

import openpyxl
import pandas as pd

from prodma.artifacts import RunLayout
from prodma.config import PipelineConfig
from prodma.stages.ingestion import run_ingestion


def _make_demand_xlsx(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "MED-01"
    sheet.append(["Reporte de medidor"])
    sheet.append(["Local Time", "kW tot mean", "kVAR tot mean", "kVA tot mean"])
    sheet.append([pd.Timestamp("2023-01-01 00:15"), 100.0, 25.0, 103.08])
    sheet.append([pd.Timestamp("2023-01-01 00:30"), 110.0, 30.0, 114.02])
    workbook.save(path)
    return path


def _config(root: Path) -> PipelineConfig:
    raw = {"inputs": {"historical_dir": "datahistorica", "years": [2023]}}
    return PipelineConfig(
        project_root=root,
        output_root=root / "resultados",
        frequency="15min",
        timezone="America/Lima",
        seed=42,
        horizons=(96,),
        raw=raw,
    )


def test_stage_reads_sheet_preserves_values_and_separates_audit(tmp_path):
    _make_demand_xlsx(tmp_path / "datahistorica" / "demanda2023.xlsx")
    layout = RunLayout.create(tmp_path / "resultados", "r1")

    result = run_ingestion(_config(tmp_path), layout)

    parquet_paths = result.artifacts["parquet"]
    assert len(parquet_paths) == 1
    frame = pd.read_parquet(parquet_paths[0])
    assert frame.loc[pd.Timestamp("2023-01-01 00:15"), "P"] == 100.0
    assert frame.loc[pd.Timestamp("2023-01-01 00:30"), "Q"] == 30.0
    assert Path(parquet_paths[0]).parent == layout.stage("01_ingesta", "parquet")
    assert all(Path(path).parent == layout.stage("01_ingesta", "auditoria") for path in result.artifacts["auditoria"])
    assert result.status == "SUCCESS"

    catalog = json.loads((layout.stage("01_ingesta") / "catalogo.json").read_text(encoding="utf-8"))
    assert catalog["inputs"][0]["sha256"]
    assert catalog["sheets"][0]["medidor"] == "MED-01"
