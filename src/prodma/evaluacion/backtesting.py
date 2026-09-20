"""Particiones rolling-origin por horizonte."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train_index: pd.DatetimeIndex
    validation_index: pd.DatetimeIndex
    horizon: int


def rolling_origins(index, horizons, folds: int, min_train: int) -> list[TemporalSplit]:
    ordered = pd.DatetimeIndex(index).sort_values().unique()
    splits: list[TemporalSplit] = []
    for horizon in horizons:
        for fold in range(folds, 0, -1):
            stop = len(ordered) - horizon * (fold - 1)
            start = stop - horizon
            if start < min_train:
                raise ValueError("Historia insuficiente para rolling-origin")
            train = ordered[:start]
            validation = ordered[start:stop]
            if not len(train) or len(validation) != horizon or train[-1] >= validation[0]:
                raise ValueError("Particion temporal invalida")
            splits.append(TemporalSplit(train, validation, int(horizon)))
    return splits
