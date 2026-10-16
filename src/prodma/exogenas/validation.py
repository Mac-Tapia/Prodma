"""Validacion estructural y temporal de fuentes exogenas.

Cada fuente habilitada se valida mediante el validador registrado para su nombre en
`DEFAULT_VALIDATORS`. Una fuente sin validador produce un error explícito; nunca se omite.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from prodma.exogenas.contracts import ExogenousCatalog, ExogenousSource
from prodma.external.registry import SourceRegistry


@dataclass(frozen=True)
class ValidationItem:
    source: str
    status: str
    rows: int
    start: str | None
    end: str | None
    warnings: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class ExogenousValidationReport:
    items: tuple[ValidationItem, ...]
    timezone: str

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(item.error for item in self.items if item.error)

    def to_dict(self) -> dict:
        return {"timezone": self.timezone, "items": [asdict(item) for item in self.items], "errors": list(self.errors)}


@dataclass(frozen=True)
class CalendarValidation:
    warnings: tuple[str, ...]


Validator = Callable[[ExogenousSource, pd.DatetimeIndex, str], ValidationItem]


def validate_frame(frame: pd.DataFrame, expected_frequency: str, timezone: str,
                   timestamp_column: str = "fecha_hora", value_columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    if timestamp_column not in frame:
        raise ValueError(f"falta columna temporal {timestamp_column}")
    checked = frame.copy()
    timestamps = pd.to_datetime(checked[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamp invalido")
    if timestamps.duplicated().any():
        raise ValueError("timestamp duplicado")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps no monotonicos")
    if len(timestamps) > 1:
        expected = pd.to_timedelta(expected_frequency)
        differences = timestamps.diff().dropna()
        if not (differences == expected).all():
            raise ValueError(f"frecuencia irregular; se esperaba {expected_frequency}")
    columns = value_columns or tuple(
        column for column in checked.select_dtypes(include=[np.number]).columns if column != timestamp_column
    )
    for column in columns:
        if column not in checked:
            raise ValueError(f"falta columna de valor {column}")
        numeric = pd.to_numeric(checked[column], errors="coerce")
        if (~np.isfinite(numeric.dropna())).any():
            raise ValueError(f"valor no finito en {column}")
    checked[timestamp_column] = timestamps
    checked.attrs["timezone"] = timezone
    return checked


def validate_calendar(frame: pd.DataFrame) -> CalendarValidation:
    required = {"fecha", "feriado", "norma"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"calendario sin columnas {sorted(missing)}")
    warnings: list[str] = []
    holidays_without_rule = frame["feriado"].astype(bool) & frame["norma"].fillna("").astype(str).str.strip().eq("")
    if holidays_without_rule.any():
        warnings.append("feriado_sin_norma")
    return CalendarValidation(tuple(warnings))


def _read_csv(source: ExogenousSource) -> pd.DataFrame:
    if source.path.suffix.lower() != ".csv":
        raise ValueError(f"formato no soportado para validacion: {source.path.suffix}")
    return pd.read_csv(source.path)


def _check_coverage(timestamps: pd.Series, demand_index: pd.DatetimeIndex) -> None:
    if timestamps.min() > demand_index.min().normalize() or timestamps.max() < demand_index.max().normalize():
        raise ValueError("cobertura temporal insuficiente")


def _validate_calendar(source: ExogenousSource, demand_index: pd.DatetimeIndex, timezone: str) -> ValidationItem:
    frame = _read_csv(source)
    warnings = list(validate_calendar(frame).warnings)
    checked = validate_frame(frame.rename(columns={"fecha": "fecha_hora"}), "1d", timezone,
                             value_columns=source.value_columns)
    timestamps = checked["fecha_hora"]
    _check_coverage(timestamps, demand_index)
    return ValidationItem(source.name, "VALID", len(frame), str(timestamps.min()), str(timestamps.max()), tuple(warnings))


def _validate_temperature(source: ExogenousSource, demand_index: pd.DatetimeIndex, timezone: str) -> ValidationItem:
    frame = _read_csv(source)
    checked = validate_frame(frame, source.frequency, timezone, value_columns=source.value_columns)
    allowed = {"reanalisis_no_es_medicion_directa", "interpolado_lineal_desde_horario"}
    if "calidad" in checked and not set(checked["calidad"].dropna().unique()) <= allowed:
        raise ValueError("etiqueta de calidad no permitida")
    timestamps = checked["fecha_hora"]
    _check_coverage(timestamps, demand_index)
    return ValidationItem(source.name, "VALID", len(frame), str(timestamps.min()), str(timestamps.max()))


def _validate_pbi(source: ExogenousSource, demand_index: pd.DatetimeIndex, timezone: str) -> ValidationItem:
    frame = _read_csv(source)
    missing = {"periodo", "valor", "fecha_descarga"} - set(frame)
    if missing:
        raise ValueError(f"PBI sin columnas {sorted(missing)}")
    if not np.isfinite(pd.to_numeric(frame["valor"], errors="coerce")).all():
        raise ValueError("PBI contiene valor no finito")
    timestamps = pd.to_datetime(frame["periodo"].str.extract(r"(?P<m>[A-Za-z]{3})\.(?P<y>\d{4})")["y"] + "-01-01")
    return ValidationItem(source.name, "VALID", len(frame), str(timestamps.min()), str(timestamps.max()),
                          ("fecha_disponible_ausente",))


def _validate_population(source: ExogenousSource, demand_index: pd.DatetimeIndex, timezone: str) -> ValidationItem:
    frame = _read_csv(source)
    missing = {"fecha", "poblacion", "dato", "fuente"} - set(frame)
    if missing:
        raise ValueError(f"poblacion sin columnas {sorted(missing)}")
    if not np.isfinite(pd.to_numeric(frame["poblacion"], errors="coerce")).all():
        raise ValueError("poblacion contiene valor no finito")
    timestamps = pd.to_datetime(frame["fecha"])
    return ValidationItem(source.name, "VALID", len(frame), str(timestamps.min()), str(timestamps.max()))


DEFAULT_VALIDATORS: SourceRegistry[Validator] = SourceRegistry()
DEFAULT_VALIDATORS.register("calendario_pe", _validate_calendar)
DEFAULT_VALIDATORS.register("temperatura_ica", _validate_temperature)
DEFAULT_VALIDATORS.register("pbi_bcrp", _validate_pbi)
DEFAULT_VALIDATORS.register("poblacion_ica", _validate_population)


def validate_exogenous_sources(catalog: ExogenousCatalog, demand_index, timezone: str = "America/Lima",
                               registry: SourceRegistry[Validator] | None = None) -> ExogenousValidationReport:
    registry = registry or DEFAULT_VALIDATORS
    index = pd.DatetimeIndex(demand_index)
    items: list[ValidationItem] = []
    for source in catalog.enabled_sources():
        try:
            validator = registry.resolve(source.name)
            items.append(validator(source, index, timezone))
        except Exception as exc:
            items.append(ValidationItem(source.name, "INVALID", 0, None, None, error=f"{type(exc).__name__}: {exc}"))
    report = ExogenousValidationReport(tuple(items), timezone)
    if report.errors:
        raise ValueError("; ".join(report.errors))
    return report
