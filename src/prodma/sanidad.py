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


def _claves(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"mes": index.month, "dow": index.dayofweek, "bloque": index.hour * 4 + index.minute // 15})


def _mapear(tabla: pd.Series, reserva: pd.Series, mediana_global: float, index: pd.DatetimeIndex) -> np.ndarray:
    """Valor de ``tabla`` (mes, dow, bloque) para cada timestamp; reserva (dow, bloque); mediana global."""
    claves = _claves(index)
    out = pd.Series(pd.MultiIndex.from_frame(claves).map(tabla), index=index)
    faltan = out.isna()
    if faltan.any():
        fb = pd.MultiIndex.from_frame(claves.loc[faltan.values, ["dow", "bloque"]]).map(reserva)
        out.loc[faltan] = np.where(pd.notna(fb), fb, mediana_global)
    return out.to_numpy(dtype=float)


def calendar_profile(s: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Perfil (mes, día de semana, bloque de 15 min) por mediana de los últimos 366 días.

    Vectorizado; con reserva (día de semana, bloque) y mediana global cuando el mes no está
    en la ventana. Equivale a mapear punto a punto (verificado en la sección 22). No corrige
    cambios de nivel: si la demanda creció respecto al año anterior, subestima.
    """
    train = s.dropna().tail(366 * 96)
    k = _claves(train.index)
    k["v"] = train.to_numpy()
    tab = k.groupby(["mes", "dow", "bloque"])["v"].median()
    reserva = k.groupby(["dow", "bloque"])["v"].median()
    return _mapear(tab, reserva, train.median(), index)


def perfil_anclado(s: pd.Series, index: pd.DatetimeIndex, dias_nivel: int = 90,
                   ventana_tendencia: int = 365) -> np.ndarray:
    """Forma estacional (mes, día de semana, bloque) x nivel reciente desestacionalizado.

    Sin entrenamiento. Separa la tendencia (media móvil centrada de ``ventana_tendencia``
    días) de la estacionalidad usando TODO el historial disponible, y ancla el nivel a los
    últimos ``dias_nivel`` días una vez quitada la estacionalidad. A diferencia de
    ``calendar_profile`` (mediana de los últimos 12 meses, que arrastra el nivel del año
    anterior) no subestima cuando la demanda creció, ni sobreestima cuando cayó.
    Multiplicativo si la serie es positiva; aditivo si tiene valores negativos (p. ej. Q).
    Con menos de ``ventana_tendencia`` + 60 días de historia no se puede separar tendencia de
    estacionalidad y se devuelve ``calendar_profile``.
    """
    s = s.dropna()
    if len(s) < 96 * (ventana_tendencia + 60):
        return calendar_profile(s, index)
    multiplicativo = bool((s > 0).mean() > 0.99 and s.mean() > 0)
    nivel = s.rolling(96 * ventana_tendencia, min_periods=96 * ventana_tendencia // 2, center=True).mean()
    forma_obs = (s / nivel) if multiplicativo else (s - nivel)
    margen = 96 * (ventana_tendencia // 2)  # solo filas con ventana centrada completa (sin sesgo en los extremos)
    forma_obs = forma_obs.iloc[margen:len(forma_obs) - margen].replace([np.inf, -np.inf], np.nan).dropna()
    if len(forma_obs) < 96 * 90:
        return calendar_profile(s, index)
    k = _claves(forma_obs.index)
    k["v"] = forma_obs.to_numpy()
    tab = k.groupby(["mes", "dow", "bloque"])["v"].median()
    reserva = k.groupby(["dow", "bloque"])["v"].median()
    glob = float(forma_obs.median())
    reciente = s.tail(96 * dias_nivel)
    forma_rec = _mapear(tab, reserva, glob, reciente.index)
    forma_fut = _mapear(tab, reserva, glob, index)
    if multiplicativo:
        resultado = float(reciente.mean()) / float(np.mean(forma_rec)) * forma_fut
    else:
        resultado = float(reciente.mean()) - float(np.mean(forma_rec)) + forma_fut
    # Tope de seguridad: en series casi nulas o a ráfagas el cociente forma/nivel se dispara.
    # Si el nivel anclado se aparta demasiado del perfil simple, no se confía en el ajuste.
    plano = calendar_profile(s, index)
    if multiplicativo:
        razon = float(np.mean(resultado)) / float(np.mean(plano)) if np.mean(plano) > 0 else np.nan
        confiable = bool(np.isfinite(razon) and 0.4 <= razon <= 2.5)
    else:
        confiable = bool(abs(np.mean(resultado) - np.mean(plano)) <= 3 * max(float(s.tail(PASOS_ANIO).std()), 1e-9))
    return resultado if confiable else plano


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
