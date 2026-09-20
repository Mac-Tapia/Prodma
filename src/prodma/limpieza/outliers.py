"""Deteccion causal robusta de atipicos puntuales."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutlierResult:
    is_outlier: pd.Series
    median: pd.Series
    scale: pd.Series


def detect_causal_outliers(series: pd.Series, window: int = 672, threshold: float = 6.0) -> OutlierResult:
    shifted = series.replace([np.inf, -np.inf], np.nan).shift(1)
    minimum = min(window, max(24, window // 7))
    median = shifted.rolling(window, min_periods=minimum).median()
    residual = (shifted - median).abs()
    scale = 1.4826 * residual.rolling(window, min_periods=minimum).median().replace(0, np.nan)
    candidate = ((series - median).abs() > threshold * scale).fillna(False)
    prior_density = candidate.shift(1, fill_value=False).rolling(17, min_periods=1).mean()
    isolated = candidate & prior_density.lt(0.5)
    return OutlierResult(isolated.astype(bool), median, scale)
