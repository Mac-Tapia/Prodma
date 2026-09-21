"""Construccion y publicacion de matrices exogenas alineadas a demanda."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prodma.artifacts import RunLayout
from prodma.config import PipelineConfig
from prodma.exogenas.asof import asof_values
from prodma.exogenas.contracts import ExogenousCatalog
from prodma.exogenas.weather import build_weather_features
from prodma.variables_externas import cargar_pbi_mensual, poblacion_en_fecha


@dataclass(frozen=True)
class ExogenousMatrices:
    historical: pd.DataFrame
    future: pd.DataFrame
    quality: pd.DataFrame
    source_hashes: dict[str, str]
    catalog: tuple[dict, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(catalog: ExogenousCatalog, name: str):
    return next(source for source in catalog.sources if source.name == name and source.enabled)


def _calendar_features(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    calendar = pd.read_csv(path)
    calendar["fecha"] = pd.to_datetime(calendar["fecha"]).dt.normalize()
    calendar = calendar.drop_duplicates("fecha").set_index("fecha")
    days = index.normalize()
    result = pd.DataFrame(index=index)
    for column in ("feriado", "fin_semana", "no_laborable"):
        result[column] = calendar[column].reindex(days).to_numpy(dtype=bool).astype(np.int8)
    return result


def _pbi_vintages(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    series = cargar_pbi_mensual(path)
    download = pd.to_datetime(raw["fecha_descarga"], errors="coerce")
    if download.isna().any():
        raise ValueError("PBI sin fecha_descarga valida")
    return pd.DataFrame({
        "periodo": series.index,
        "valor": series.to_numpy(dtype=float),
        "fecha_disponible": download.to_numpy(),
    })


def _historical_asof(index: pd.DatetimeIndex, vintages: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=index, dtype=float)
    ordered = vintages.sort_values(["periodo", "fecha_disponible"])
    for row in ordered.itertuples(index=False):
        available_from = max(pd.Timestamp(row.periodo), pd.Timestamp(row.fecha_disponible))
        result.loc[index >= available_from] = float(row.valor)
    return result


def _population(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    years = pd.to_datetime(frame["fecha"]).dt.year
    return pd.Series(frame["poblacion"].to_numpy(dtype=float), index=years).sort_index()


def build_exogenous_matrices(config: PipelineConfig, catalog: ExogenousCatalog,
                              historical_index, future_end: pd.Timestamp | str) -> ExogenousMatrices:
    historical_index = pd.DatetimeIndex(historical_index)
    if historical_index.empty or historical_index.has_duplicates:
        raise ValueError("indice historico vacio o duplicado")
    historical_index = historical_index.rename("Fecha")
    future_end = pd.Timestamp(future_end)
    future_index = pd.date_range(historical_index.max() + pd.Timedelta(config.frequency), future_end,
                                 freq=config.frequency, name="Fecha")

    calendar_source = _source(catalog, "calendario_pe")
    weather_source = _source(catalog, "temperatura_ica")
    pbi_source = _source(catalog, "pbi_bcrp")
    population_source = _source(catalog, "poblacion_ica")
    historical = _calendar_features(calendar_source.path, historical_index)
    future = _calendar_features(calendar_source.path, future_index)

    weather = pd.read_csv(weather_source.path)
    combined_index = historical_index.append(future_index)
    weather_features = build_weather_features(weather, combined_index, origin=historical_index.max())
    historical["temp_climatologica_c"] = weather_features.climatology.reindex(historical_index)
    future["temp_climatologica_c"] = weather_features.climatology.reindex(future_index)

    vintages = _pbi_vintages(pbi_source.path)
    historical["pbi_last_known"] = _historical_asof(historical_index, vintages)
    execution_origin = pd.Timestamp(vintages["fecha_disponible"].max())
    future["pbi_last_known"] = asof_values(future_index, vintages, execution_origin)

    population = _population(population_source.path)
    historical["poblacion_ica"] = poblacion_en_fecha(historical_index, population)
    future["poblacion_ica"] = poblacion_en_fecha(future_index, population)

    quality_historical = pd.DataFrame(index=historical_index)
    quality_historical["segmento"] = "historico"
    quality_historical["temperatura_tipo"] = "climatologia_ajustada_hasta_origen"
    quality_historical["pbi_disponible"] = historical["pbi_last_known"].notna()
    quality_historical["poblacion_tipo"] = "observada_o_interpolada_declarada"
    quality_future = pd.DataFrame(index=future_index)
    quality_future["segmento"] = "futuro"
    quality_future["temperatura_tipo"] = "climatologia_no_pronostico_meteorologico"
    quality_future["pbi_disponible"] = future["pbi_last_known"].notna()
    quality_future["poblacion_tipo"] = "ultimo_valor_o_proyeccion_disponible"
    quality = pd.concat([quality_historical, quality_future])
    records = tuple({**asdict(source), "path": str(source.path)} for source in catalog.sources)
    hashes = {source.name: source.sha256 for source in catalog.enabled_sources()}
    return ExogenousMatrices(historical, future, quality, hashes, records)


def _summary(frame: pd.DataFrame, demand: pd.DataFrame | None) -> pd.DataFrame:
    rows: list[dict] = []
    demand_mean = demand.mean(axis=1) if demand is not None and len(demand.columns) else None
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        correlation = np.nan
        aligned_demand = demand_mean.reindex(frame.index) if demand_mean is not None else None
        if (aligned_demand is not None and values.notna().sum() > 2
                and values.std() > 0 and aligned_demand.std() > 0):
            correlation = values.corr(aligned_demand)
        rows.append({
            "variable": column,
            "filas": len(values),
            "faltantes": int(values.isna().sum()),
            "cobertura_pct": float(values.notna().mean() * 100),
            "unicos": int(values.nunique(dropna=True)),
            "minimo": float(values.min()) if values.notna().any() else np.nan,
            "maximo": float(values.max()) if values.notna().any() else np.nan,
            "cambios": int(values.ne(values.shift()).sum() - 1),
            "correlacion_demanda_media": float(correlation) if pd.notna(correlation) else np.nan,
        })
    return pd.DataFrame(rows)


def publish_exogenous_matrices(matrices: ExogenousMatrices, layout: RunLayout,
                               demand: pd.DataFrame | None = None) -> dict[str, str]:
    data_dir = layout.stage("02_limpieza", "datos")
    audit_dir = layout.stage("02_limpieza", "auditoria")
    stats_dir = layout.stage("02_limpieza", "estadistica_descriptiva")
    figures_dir = stats_dir / "figuras"
    figures_dir.mkdir(parents=True, exist_ok=True)
    historical_path = data_dir / "matriz_exogenas_historicas.parquet"
    future_path = data_dir / "matriz_exogenas_futuras.parquet"
    quality_path = data_dir / "calidad_exogenas.parquet"
    catalog_path = audit_dir / "catalogo_exogenas.json"
    matrices.historical.to_parquet(historical_path)
    matrices.future.to_parquet(future_path)
    matrices.quality.to_parquet(quality_path)
    catalog_path.write_text(json.dumps({"sources": matrices.catalog, "hashes": matrices.source_hashes},
                                       ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary = _summary(matrices.historical, demand)
    summary.to_csv(stats_dir / "exogenas_resumen.csv", index=False)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(summary["variable"], summary["cobertura_pct"])
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylabel("Cobertura (%)")
    figure.tight_layout()
    figure.savefig(figures_dir / "exogenas_cobertura.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(matrices.historical.index, matrices.historical["temp_climatologica_c"].to_numpy(), linewidth=0.6)
    axis.set_ylabel("Temperatura climatologica (C)")
    figure.tight_layout()
    figure.savefig(figures_dir / "exogenas_temperatura.png", dpi=120)
    plt.close(figure)

    if demand is not None and len(demand.columns):
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.scatter(matrices.historical["temp_climatologica_c"], demand.mean(axis=1), s=2, alpha=0.2)
        axis.set_xlabel("Temperatura climatologica (C)")
        axis.set_ylabel("Demanda media entre series")
        figure.tight_layout()
        figure.savefig(figures_dir / "exogenas_demanda_temperatura.png", dpi=120)
        plt.close(figure)
    return {
        "exogenous_historical": str(historical_path),
        "exogenous_future": str(future_path),
        "exogenous_quality": str(quality_path),
        "exogenous_catalog": str(catalog_path),
        "exogenous_sha256": _sha256(historical_path),
    }
