"""Particiones anidadas con test final inaccesible al entrenamiento."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from prodma.evaluacion.backtesting import TemporalSplit, rolling_origins


class InsufficientHistory(ValueError):
    state = "HISTORIA_INSUFICIENTE"


@dataclass(frozen=True)
class TrainingPartition:
    inner_folds: tuple[TemporalSplit, ...]
    calibration_index: pd.DatetimeIndex
    final_test_index: pd.DatetimeIndex
    horizons: tuple[int, ...]


def nested_temporal_partition(index, horizons, folds: int, min_train: int,
                              frequency: str = "15min") -> TrainingPartition:
    ordered = pd.DatetimeIndex(index).sort_values().unique()
    horizons = tuple(int(horizon) for horizon in horizons)
    if not horizons or folds <= 0 or min_train <= 0:
        raise ValueError("horizontes, folds y min_train deben ser positivos")
    final_size = 2 * max(horizons)  # test final: dos ventanas del horizonte mayor para calibrar y medir
    if len(ordered) <= final_size:
        raise InsufficientHistory("HISTORIA_INSUFICIENTE: no alcanza para reservar test final")
    final_test = ordered[-final_size:]
    expected = pd.Timedelta(frequency)
    if len(final_test) > 1 and not (final_test.to_series().diff().dropna() == expected).all():
        raise ValueError("test final no contiguo")
    calibration = ordered[:-final_size]
    try:
        inner = rolling_origins(calibration, horizons=horizons, folds=folds, min_train=min_train)
    except ValueError as exc:
        if "insuficiente" in str(exc).lower():
            raise InsufficientHistory(f"HISTORIA_INSUFICIENTE: {exc}") from exc
        raise
    if any(set(split.validation_index).intersection(final_test) for split in inner):
        raise RuntimeError("fuga: fold interno intersecta test final")
    return TrainingPartition(tuple(inner), calibration, final_test, horizons)
