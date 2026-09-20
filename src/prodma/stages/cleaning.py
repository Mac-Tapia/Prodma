"""Etapa 2: consolidacion y publicacion de la matriz limpia unica."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from prodma.artifacts import RunLayout
from prodma.limpieza.temporal import build_clean_variable
from prodma.limpieza.descriptive import describe_variable
from prodma.limpieza.electrical import ElectricalLimits, electrical_flags
from prodma.limpieza.outliers import detect_causal_outliers
from prodma.series import resolver_duplicados
from prodma.stages import StageResult


def matrix_output_path(layout: RunLayout) -> Path:
    return layout.stage("02_limpieza", "datos") / "matriz_limpia.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_cleaning(config, layout: RunLayout) -> StageResult:
    catalog_path = layout.stage("01_ingesta") / "catalogo.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Falta catalogo de ingesta: {catalog_path}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    rows = [row for row in catalog.get("sheets", []) if row.get("estado") == "ok"]
    by_meter: dict[str, list[pd.DataFrame]] = {}
    for row in rows:
        by_meter.setdefault(row["medidor"], []).append(pd.read_parquet(row["path"]))
    cleaning_cfg = getattr(config, "raw", {}).get("cleaning", {}) if config is not None else {}
    forward_limit = int(cleaning_cfg.get("forward_fill_limit", 4))
    weekly_lag = int(cleaning_cfg.get("weekly_lag", 672))
    columns: dict[str, pd.Series] = {}
    descriptions: list[dict] = []
    electrical_rows: list[pd.DataFrame] = []
    for meter, parts in sorted(by_meter.items()):
        raw = resolver_duplicados(pd.concat(parts).sort_index())
        raw = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="15min", name="Fecha"))
        flags = electrical_flags(raw, ElectricalLimits())
        if len(flags):
            flags = flags.assign(medidor=meter).reset_index()
            electrical_rows.append(flags)
        for variable in ("P", "Q", "S_medida"):
            if variable not in raw:
                continue
            clean = build_clean_variable(raw[variable], forward_limit, weekly_lag)
            detected = detect_causal_outliers(
                clean["observado"],
                int(cleaning_cfg.get("outlier_window", 672)),
                float(cleaning_cfg.get("outlier_threshold", 6.0)),
            )
            lower = detected.median - float(cleaning_cfg.get("outlier_threshold", 6.0)) * detected.scale
            upper = detected.median + float(cleaning_cfg.get("outlier_threshold", 6.0)) * detected.scale
            corrected = clean["observado"].clip(lower=lower, upper=upper)
            clean["es_atipico"] = detected.is_outlier
            clean["es_recortado"] = detected.is_outlier
            clean.loc[detected.is_outlier, "limpio"] = corrected.loc[detected.is_outlier]
            clean.loc[clean["es_observado"], "objetivo"] = clean.loc[clean["es_observado"], "limpio"]
            clean.loc[clean["es_observado"], "contexto"] = clean.loc[clean["es_observado"], "limpio"]
            descriptions.append({"medidor": meter, "variable": variable, **describe_variable(clean)})
            figure, axis = plt.subplots(figsize=(12, 4))
            axis.plot(clean.index, clean["observado"].to_numpy(), color="lightgray", linewidth=0.7, label="observado")
            axis.plot(clean.index, clean["limpio"].to_numpy(), color="tab:blue", linewidth=0.8, label="limpio")
            axis.set_title(f"{meter} - {variable}: observado y limpio")
            axis.legend()
            figure.tight_layout()
            figure.savefig(layout.stage("02_limpieza", "figuras") / f"{meter}_{variable}_crudo_limpio.png", dpi=120)
            plt.close(figure)
            for field in clean:
                columns[f"{meter}__{variable}__{field}"] = clean[field]
    if not columns:
        raise ValueError("La ingesta no contiene series limpiables")
    matrix = pd.concat(columns, axis=1)
    target = matrix_output_path(layout)
    matrix.to_parquet(target)
    digest = _sha256(target)
    metadata = {"matrix": str(target), "sha256": digest, "rows": len(matrix), "columns": len(matrix.columns)}
    metadata_path = layout.stage("02_limpieza", "auditoria") / "matriz_limpia.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(descriptions).to_csv(layout.stage("02_limpieza", "estadistica_descriptiva") / "resumen_por_medidor.csv", index=False)
    if electrical_rows:
        pd.concat(electrical_rows, ignore_index=True).to_csv(
            layout.stage("02_limpieza", "csv") / "calidad_electrica.csv", index=False
        )
    return StageResult("02_limpieza", "SUCCESS", layout.stage("02_limpieza"), metadata)
