"""Pronóstico recursivo rápido (paso a paso, cada 15 min) con modelos ya entrenados.

El notebook original recalculaba ``features(serie)`` completo (pandas, ~50 columnas, un
bucle de Python por fila para los feriados) sobre un buffer de 1500 filas EN CADA PASO,
y lo hacía en serie. Aquí se separa lo que sí es recursivo de lo que no:

* Columnas de **calendario/exógenas** (hora, día, feriado, temperatura climatológica, PBI,
  población): dependen solo del timestamp, así que se calculan UNA vez, vectorizadas, para
  todo el horizonte (``features_calendario``).
* Columnas **dinámicas** (rezagos, rolling, mismo día/hora de las últimas 4 semanas): se
  calculan con numpy sobre un buffer que crece con los propios pronósticos
  (``proyectar_recursivo``), solo las que el modelo realmente usa.

Diferencia deliberada frente a la versión anterior: el buffer conserva ≥ 4 semanas
(2688 pasos). Antes se recortaba a 1500 filas, así que ``media/std_mismo_dia_hora_4sem``
se calculaban con 2 semanas en el pronóstico y con 4 en el entrenamiento.

Las funciones de nivel de módulo son importables por procesos trabajadores (joblib/loky),
lo que permite pronosticar varias series en paralelo (``proyectar_series``).
"""
from __future__ import annotations

import re
import time
import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

BUFFER_MINIMO = 672 * 4  # 4 semanas: alcance máximo de las features dinámicas
LAGS = (1, 4, 96, 672)
VENTANAS = (4, 24, 96, 672)
_RE_ROLLING = re.compile(r"^(media|std|mediana|min|max|p95)_(\d+)$")
_RE_LAG = re.compile(r"^lag_(\d+)$")
_MISMO_DIA_HORA = ("media_mismo_dia_hora_4sem", "std_mismo_dia_hora_4sem")


# ---------------------------------------------------------------------------
# Columnas de calendario / exógenas (dependen solo del timestamp)
# ---------------------------------------------------------------------------
def features_calendario(idx: pd.DatetimeIndex, feriados, climatologia=None, pbi=None,
                        poblacion=None) -> pd.DataFrame:
    """Mismas columnas, orden y definición que la parte de calendario de ``features()``."""
    from prodma.variables_externas import pbi_en_fecha, poblacion_en_fecha, temperatura_climatologica

    X = pd.DataFrame(index=idx)
    hora = idx.hour + idx.minute / 60
    for k in (1, 2, 3):
        X[f"hora_sin{k}"] = np.sin(2 * np.pi * k * hora / 24)
        X[f"hora_cos{k}"] = np.cos(2 * np.pi * k * hora / 24)
    X["anio_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    X["anio_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    for dia in range(1, 7):
        X[f"dia_{dia}"] = (idx.dayofweek == dia).astype(int)
    dias = idx.normalize()
    es_feriado = {d: int(d.date() in feriados) for d in dias.unique()}
    X["feriado"] = dias.map(es_feriado).to_numpy()
    X["fin_semana"] = (idx.dayofweek >= 5).astype(int)
    if climatologia is not None:
        X["temp_climatologica_c"] = temperatura_climatologica(idx, climatologia)
    if pbi is not None:
        X["pbi_bcrp"] = pbi_en_fecha(idx, pbi)
    if poblacion is not None:
        X["poblacion_ica"] = poblacion_en_fecha(idx, poblacion)
    return X


def es_columna_dinamica(nombre: str) -> bool:
    return bool(
        _RE_LAG.match(nombre)
        or _RE_ROLLING.match(nombre)
        or nombre in _MISMO_DIA_HORA
        or nombre == "prop_imputado_96"
    )


# ---------------------------------------------------------------------------
# Recursión con numpy
# ---------------------------------------------------------------------------
def _plan_columnas(columnas: Sequence[str], calendario_cols: Sequence[str]):
    """Para cada columna del modelo: (tipo, argumento_a, argumento_b)."""
    posicion_cal = {c: i for i, c in enumerate(calendario_cols)}
    plan = []
    for c in columnas:
        if c in posicion_cal:
            plan.append(("cal", posicion_cal[c], 0))
        elif (m := _RE_LAG.match(c)):
            plan.append(("lag", int(m.group(1)), 0))
        elif (m := _RE_ROLLING.match(c)):
            plan.append((m.group(1), int(m.group(2)), 0))
        elif c == "media_mismo_dia_hora_4sem":
            plan.append(("mdh_media", 0, 0))
        elif c == "std_mismo_dia_hora_4sem":
            plan.append(("mdh_std", 0, 0))
        elif c == "prop_imputado_96":
            plan.append(("cero", 0, 0))  # en recursión no hay imputados: todo es observado o pronosticado
        else:
            raise ValueError(f"Columna de modelo desconocida para el pronóstico recursivo: {c}")
    return plan


def _valor_dinamico(tipo: str, a: int, buf: np.ndarray, p: int) -> float:
    if tipo == "lag":
        return buf[p - a] if p - a >= 0 else np.nan
    if tipo in ("media", "std", "mediana", "min", "max", "p95"):
        if p - a < 0:
            return np.nan
        w = buf[p - a:p]  # equivale a serie.shift(1).rolling(a): los `a` valores previos a t
        if tipo == "media":
            return w.mean()
        if tipo == "std":
            return w.std(ddof=1)
        if tipo == "mediana":
            return float(np.median(w))
        if tipo == "min":
            return w.min()
        if tipo == "max":
            return w.max()
        return float(np.quantile(w, 0.95))
    # mismo día y hora de las últimas hasta 4 semanas (t-672k), ignorando las que no existen
    vals = [buf[p - 672 * k] for k in (1, 2, 3, 4) if p - 672 * k >= 0]
    if tipo == "mdh_media":
        return float(np.mean(vals)) if vals else np.nan
    if tipo == "mdh_std":
        return float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
    return 0.0  # "cero"


def preparar_para_inferencia(modelo) -> None:
    """Ajustes solo de velocidad para predecir UNA fila por vez, sin cambiar el resultado.

    Un ``n_jobs`` alto reparte cada fila entre hilos (puro costo de gestión), y XGBoost
    entrenado en GPU no debe usarla para inferencia de una fila (ni competir entre procesos).
    """
    interno = getattr(modelo, "regressor_", modelo)
    final = list(interno.named_steps.values())[-1] if hasattr(interno, "named_steps") else interno
    for est in (final,):
        try:
            nombre = type(est).__name__
            if nombre == "XGBRegressor":
                est.set_params(n_jobs=1, device="cpu")
                est.get_booster().set_param({"device": "cpu", "nthread": 1})
            elif hasattr(est, "n_jobs"):
                est.n_jobs = 1
        except Exception:  # pragma: no cover - optimización opcional, nunca debe romper el pronóstico
            pass


def proyectar_recursivo(modelo, historia: np.ndarray, columnas: Sequence[str],
                        calendario: pd.DataFrame, *, progreso: int = 0) -> np.ndarray:
    """Pronóstico recursivo a 15 min.

    ``historia``: serie observada/imputada (sin NaN), termina justo antes del primer paso.
    ``calendario``: columnas de calendario/exógenas indexadas por los timestamps futuros
    (una fila por paso); debe contener todas las columnas no dinámicas de ``columnas``.
    """
    historia = np.asarray(historia, dtype=float)
    pasos = len(calendario)
    n = len(historia)
    cal_cols = [c for c in columnas if not es_columna_dinamica(c)]
    faltantes = [c for c in cal_cols if c not in calendario.columns]
    if faltantes:
        raise ValueError(f"El calendario no trae las columnas del modelo: {faltantes}")
    cal_vals = calendario[cal_cols].to_numpy(dtype=float)
    plan = _plan_columnas(columnas, cal_cols)
    buf = np.empty(n + pasos)
    buf[:n] = historia
    fila = np.empty(len(columnas))
    salida = np.empty(pasos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(pasos):
            p = n + i
            for j, (tipo, a, _b) in enumerate(plan):
                fila[j] = cal_vals[i, a] if tipo == "cal" else _valor_dinamico(tipo, a, buf, p)
            if not np.isfinite(fila).all():
                raise ValueError("Lags incompletos; no fabricar pronóstico")
            valor = float(modelo.predict(pd.DataFrame(fila.reshape(1, -1), columns=list(columnas)))[0])
            if not np.isfinite(valor):
                raise ValueError("Pronóstico no finito")
            buf[p] = valor
            salida[i] = valor
            if progreso and (i + 1) % progreso == 0:
                print(f"  ...paso {i + 1}/{pasos}", flush=True)
    return salida


# ---------------------------------------------------------------------------
# Paralelismo por serie (procesos)
# ---------------------------------------------------------------------------
def cargar_modelo(ruta):
    """Carga un ``.joblib`` (también los entrenados antes de mover las transformaciones)."""
    import joblib

    from prodma.transformaciones import registrar_en_main

    registrar_en_main()
    return joblib.load(ruta)


def proyectar_desde_disco(clave, ruta_modelo, columnas, historia, calendario_columnas,
                          calendario_valores, indice_futuro):
    """Tarea de un proceso trabajador: carga el modelo del disco y proyecta una serie.

    Devuelve ``(clave, pred, segundos, error)``. Un fallo en una serie no debe tumbar a las
    demás: se devuelve ``pred=None`` y el texto del error.
    """
    inicio = time.time()
    try:
        modelo = cargar_modelo(ruta_modelo)
        preparar_para_inferencia(modelo)
        calendario = pd.DataFrame(calendario_valores, index=indice_futuro, columns=list(calendario_columnas))
        pred = proyectar_recursivo(modelo, historia, columnas, calendario)
        return clave, pred, time.time() - inicio, None
    except Exception as exc:  # noqa: BLE001 - se informa por serie, no se propaga
        return clave, None, time.time() - inicio, f"{type(exc).__name__}: {exc}"


def proyectar_series(tareas: Iterable[dict], n_jobs: int = 4):
    """Genera ``(clave, pred, segundos, error)`` a medida que cada serie termina.

    ``tareas`` puede ser un generador: se consume de forma perezosa (unas pocas tareas por
    proceso a la vez), de modo que no se acumulan en RAM los calendarios/historias de todas
    las series. Con ``n_jobs=1`` se ejecuta en el proceso actual (útil para depurar); con
    más, en procesos separados (cada uno carga su propio modelo: no se acumulan los modelos
    en la RAM del kernel).
    """
    from joblib import Parallel, delayed

    if n_jobs <= 1:
        for t in tareas:
            yield proyectar_desde_disco(**t)
        return
    yield from Parallel(n_jobs=n_jobs, backend="loky", return_as="generator_unordered")(
        delayed(proyectar_desde_disco)(**t) for t in tareas
    )
