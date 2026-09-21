"""Etapa 2: consolidacion y publicacion de la matriz limpia unica."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prodma.artifacts import RunLayout
from prodma.exogenas.catalog import build_exogenous_catalog
from prodma.exogenas.validation import validate_exogenous_sources
from prodma.limpieza.temporal import build_clean_variable
from prodma.limpieza.descriptive import describe_variable, normality_diagnostics, plot_distribution_diagnostics
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
    years = [int(row["anio"]) for row in rows if row.get("anio") is not None]
    canonical_year = max(years) if years else None
    canonical_meters = (
        {row["medidor"] for row in rows if row.get("anio") == canonical_year}
        if canonical_year is not None else {row["medidor"] for row in rows}
    )
    years_by_meter = {
        meter: sorted({int(row["anio"]) for row in rows if row["medidor"] == meter and row.get("anio") is not None})
        for meter in canonical_meters
    }
    recent_meters = {
        meter for meter, meter_years in years_by_meter.items()
        if canonical_year is not None and meter_years == [canonical_year]
    }
    by_meter: dict[str, list[pd.DataFrame]] = {}
    for row in rows:
        if row["medidor"] in canonical_meters:
            by_meter.setdefault(row["medidor"], []).append(pd.read_parquet(row["path"]))
    cleaning_cfg = getattr(config, "raw", {}).get("cleaning", {}) if config is not None else {}
    for stale_figure in layout.stage("02_limpieza", "figuras").glob("*_crudo_limpio.png"):
        stale_figure.unlink()
    forward_limit = int(cleaning_cfg.get("forward_fill_limit", 4))
    weekly_lag = int(cleaning_cfg.get("weekly_lag", 672))
    quality_columns: dict[str, pd.Series] = {}
    value_columns: dict[str, pd.Series] = {}
    recent_columns: dict[str, pd.Series] = {}
    meters_without_data: list[dict] = []
    descriptions: list[dict] = []
    normality_rows: list[dict] = []
    electrical_rows: list[pd.DataFrame] = []
    for meter, parts in sorted(by_meter.items()):
        raw = resolver_duplicados(pd.concat(parts).sort_index())
        raw = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="15min", name="Fecha"))
        valid_pq = sum(int(raw[column].notna().sum()) for column in ("P", "Q") if column in raw)
        if valid_pq == 0:
            meters_without_data.append({"medidor": meter, "motivo": "sin_mediciones_P_Q"})
            continue
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
            seasonal_difference = clean["observado"] - clean["observado"].shift(weekly_lag)
            diagnostic = normality_diagnostics(
                seasonal_difference,
                alpha=float(getattr(config, "raw", {}).get("evaluation", {}).get("alpha", 0.05))
                if config is not None else 0.05,
            )
            route = diagnostic["decision_inferencial"]
            recommendations = {
                "parametrica_candidata": ("t_pareada_sobre_diferencial_de_perdida", "anova_medidas_repetidas"),
                "no_parametrica": ("wilcoxon_pareado_con_ajuste_holm", "friedman_y_posthoc_holm"),
                "muestra_insuficiente": ("no_aplica", "no_aplica"),
            }
            two_models, multiple_models = recommendations[str(route)]
            normality_rows.append({
                "medidor": meter,
                "variable": variable,
                "serie_evaluada": "diferencia_estacional_observada",
                "rezago": weekly_lag,
                **diagnostic,
                "prueba_dos_modelos": two_models,
                "prueba_tres_o_mas_modelos": multiple_models,
                "nota": "ruta_preliminar; confirmar sobre diferenciales de error en modulo_04",
            })
            stats_figures = layout.stage("02_limpieza", "estadistica_descriptiva") / "figuras"
            stats_figures.mkdir(parents=True, exist_ok=True)
            plot_distribution_diagnostics(
                seasonal_difference,
                stats_figures / f"{meter}_{variable}_distribucion_qq.png",
                f"{meter} - {variable}: diferencia estacional observada",
            )
            output_variable = "S" if variable == "S_medida" else variable
            destination = recent_columns if meter in recent_meters else value_columns
            destination[f"{meter}_{output_variable}"] = clean["contexto"]
            figure, axis = plt.subplots(figsize=(12, 4))
            axis.plot(clean.index, clean["observado"].to_numpy(), color="lightgray", linewidth=0.7, label="observado")
            axis.plot(clean.index, clean["limpio"].to_numpy(), color="tab:blue", linewidth=0.8, label="limpio")
            axis.set_title(f"{meter} - {variable}: observado y limpio")
            axis.legend()
            figure.tight_layout()
            figure.savefig(layout.stage("02_limpieza", "figuras") / f"{meter}_{variable}_crudo_limpio.png", dpi=120)
            plt.close(figure)
            for field in clean:
                quality_columns[f"{meter}__{variable}__{field}"] = clean[field]
    if not value_columns and not recent_columns:
        raise ValueError("La ingesta no contiene series limpiables")
    matrix = pd.concat(value_columns, axis=1) if value_columns else pd.DataFrame()
    recent = pd.concat(recent_columns, axis=1) if recent_columns else pd.DataFrame()
    quality = pd.concat(quality_columns, axis=1)
    exogenous_report = None
    if config is not None and config.raw.get("exogenous", {}).get("sources"):
        exogenous_report = validate_exogenous_sources(
            build_exogenous_catalog(config), matrix.index, timezone=config.timezone
        )
        (layout.stage("02_limpieza", "auditoria") / "exogenas_validacion.json").write_text(
            json.dumps(exogenous_report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    target = matrix_output_path(layout)
    matrix.to_parquet(target)
    recent_target = layout.stage("02_limpieza", "datos") / "matriz_medidores_recientes.parquet"
    recent.to_parquet(recent_target)
    quality_target = layout.stage("02_limpieza", "datos") / "matriz_calidad.parquet"
    quality.to_parquet(quality_target)
    digest = _sha256(target)
    metadata = {
        "matrix": str(target), "recent": str(recent_target), "quality": str(quality_target), "sha256": digest,
        "rows": len(matrix), "columns": len(matrix.columns), "meters": len(by_meter),
        "canonical_year": canonical_year, "recent_meters": len(recent_meters),
        "meters_without_data": len(meters_without_data),
        "exogenous_sources_validated": len(exogenous_report.items) if exogenous_report else 0,
    }
    metadata_path = layout.stage("02_limpieza", "auditoria") / "matriz_limpia.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(descriptions).to_csv(layout.stage("02_limpieza", "estadistica_descriptiva") / "resumen_por_medidor.csv", index=False)
    normality = pd.DataFrame(normality_rows)
    normality.to_csv(layout.stage("02_limpieza", "estadistica_descriptiva") / "normalidad_inferencia.csv", index=False)
    decision_counts = normality["decision_inferencial"].value_counts().to_dict() if len(normality) else {}
    metadata["normality_decisions"] = {str(key): int(value) for key, value in decision_counts.items()}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if electrical_rows:
        pd.concat(electrical_rows, ignore_index=True).to_csv(
            layout.stage("02_limpieza", "csv") / "calidad_electrica.csv", index=False
        )
    pd.DataFrame(meters_without_data, columns=["medidor", "motivo"]).to_csv(
        layout.stage("02_limpieza", "csv") / "medidores_sin_datos.csv", index=False
    )
    pd.DataFrame({"medidor": sorted(recent_meters), "anio_incorporacion": canonical_year}).to_csv(
        layout.stage("02_limpieza", "csv") / "medidores_recientes.csv", index=False
    )
    return StageResult("02_limpieza", "SUCCESS", layout.stage("02_limpieza"), metadata)
