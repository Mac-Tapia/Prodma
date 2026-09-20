"""Perfil calendario (plan B sin entrenamiento) y guardas de sanidad de las proyecciones.

Un pronóstico recursivo a 15 min sobre 16+ meses acumula error sin una cota validada
(ver sección 20 del notebook). Estas guardas no corrigen nada en silencio: comparan cada
proyección con lo observado en los últimos 12 meses, la MARCAN si se aparta demasiado y,
solo entonces, la sustituyen por el perfil calendario, dejando ambas columnas en el archivo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PASOS_ANIO = 96 * 365


def calendar_profile(s: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Perfil (mes, día de semana, bloque de 15 min) por mediana de los últimos 366 días.

    Vectorizado; con reserva (día de semana, bloque) y mediana global cuando el mes no está
    en la ventana. Equivale a mapear punto a punto (verificado en la sección 22).
    """
    train = s.dropna().tail(366 * 96)
    key_full = pd.DataFrame({"mes": train.index.month, "dow": train.index.dayofweek,
                             "bloque": train.index.hour * 4 + train.index.minute // 15, "v": train.to_numpy()})
    tab = key_full.groupby(["mes", "dow", "bloque"])["v"].median()
    fallback = key_full.groupby(["dow", "bloque"])["v"].median()
    global_median = train.median()
    idx_keys = pd.DataFrame({"mes": index.month, "dow": index.dayofweek,
                             "bloque": index.hour * 4 + index.minute // 15})
    out = pd.Series(pd.MultiIndex.from_frame(idx_keys).map(tab), index=index)
    missing = out.isna()
    if missing.any():
        fb_vals = pd.MultiIndex.from_frame(idx_keys.loc[missing.values, ["dow", "bloque"]]).map(fallback)
        out.loc[missing] = np.where(pd.notna(fb_vals), fb_vals, global_median)
    return out.to_numpy(dtype=float)


def evaluar_sanidad(pred: pd.Series, historia: pd.Series, umbral: float = 0.25,
                    ratio_std_minimo: float = 0.3, factor_rango: float = 1.5) -> dict:
    """Compara la proyección con los últimos 12 meses de historia (observada/imputada).

    Marcas: ``desvio_media`` (la media proyectada difiere > ``umbral`` de la media de los
    últimos 12 meses), ``aplanada`` (la variabilidad proyectada es < ``ratio_std_minimo`` de
    la histórica: la recursión colapsó a una constante) y ``fuera_de_rango`` (máximo
    > ``factor_rango`` × máximo histórico, o mínimo muy por debajo del histórico).
    """
    h = historia.dropna().tail(PASOS_ANIO)
    p = pred.dropna()
    if len(h) < 96 * 30 or p.empty:
        return dict(estado="SIN_BASE", marcas="", desvio_media=np.nan, ratio_std=np.nan,
                    media_hist_12m=np.nan, media_proyectada=np.nan)
    media_h, media_p = float(h.mean()), float(p.mean())
    escala = max(abs(media_h), float(h.std()), 1e-9)
    desvio = (media_p - media_h) / escala
    std_h = float(h.std())
    ratio_std = float(p.std() / std_h) if std_h > 1e-12 else np.nan
    marcas = []
    if abs(desvio) > umbral:
        marcas.append("desvio_media")
    if np.isfinite(ratio_std) and ratio_std < ratio_std_minimo:
        marcas.append("aplanada")
    rango = float(h.max() - h.min())
    if p.max() > factor_rango * max(float(h.max()), 1e-9) or p.min() < float(h.min()) - 0.5 * rango:
        marcas.append("fuera_de_rango")
    return dict(estado="REVISAR" if marcas else "OK", marcas=",".join(marcas), desvio_media=float(desvio),
                ratio_std=ratio_std, media_hist_12m=media_h, media_proyectada=media_p)
