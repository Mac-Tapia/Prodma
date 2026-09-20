"""Reanudar el pipeline desde una corrida previa SIN volver a entrenar.

La corrida completa guarda cada modelo (`modelos/*.joblib`) y las métricas
(`tablas/metricas.csv`), pero el estado que el resto del notebook necesita en memoria
(`PRED`, `WINNERS`, `RESULTS`) se perdía al caer el kernel. Aquí se reconstruye:

* ``PRED``: se vuelve a predecir el tramo de prueba con cada modelo guardado, sobre EL MISMO
  corte train/test de la sección 14 (``dividir_train_test``).
* Validación: el RMSE recalculado debe coincidir con el guardado en ``metricas.csv``; si no,
  la restauración no es fiel y se aborta (``RestauracionInconsistente``).
* ``WINNERS``: ganador operativo = menor ``CV_RMSE`` entre los 6 modelos ML. Los benchmarks
  ``Estacional_*`` siguen en las tablas, pero NO se usan como pronóstico operativo: en
  recursión a meses/años "Estacional_1" es una línea plana.

Las funciones reciben ``features``, ``load_meter`` y ``metrics`` del notebook como
parámetros (inyección de dependencias), de modo que el módulo se puede probar sin él.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

MODELOS_ML = ("Ridge", "RandomForest", "XGBoost", "LightGBM", "CatBoost", "MLP")
LAGS_BENCHMARK = (1, 96, 672)
ESTADO_OK = "OK"
ESTADO_CONSTANTE = "SERIE_CONSTANTE"
ESTADO_SIN_DATOS = "SIN_DATOS_RECIENTES"
ARCHIVO_ESTADO = "estado_restaurado.joblib"

CLAVES_CONFIG_COMPARADAS = (
    "modo", "seed", "train_frac", "min_train_days", "test_days", "cv_days", "cv_folds", "es_days",
    "fecha_inicio", "usar_temperatura_climatologica", "usar_pbi", "usar_poblacion",
)


class RestauracionInconsistente(RuntimeError):
    """Los modelos guardados no reproducen las métricas guardadas."""


def verificar_config(ruta_config_origen: Path, cfg: dict) -> None:
    """La configuración actual debe ser compatible con la que entrenó los modelos."""
    origen = json.loads(Path(ruta_config_origen).read_text(encoding="utf-8"))
    claves = list(CLAVES_CONFIG_COMPARADAS) + [f"train_days_{cfg['modo']}"]
    difieren = {k: (origen.get(k), cfg.get(k)) for k in claves if origen.get(k) != cfg.get(k)}
    if difieren:
        raise ValueError(
            "La configuración actual no coincide con la de la corrida que entrenó los modelos "
            f"(clave: (origen, actual)): {difieren}"
        )


def columnas_de_modelo(estimador) -> list[str]:
    """Columnas (y orden) con las que se entrenó un modelo guardado."""
    externo = getattr(estimador, "regressor_", estimador)
    candidatos = [externo]
    if hasattr(externo, "named_steps"):  # un Pipeline expone las columnas en el propio Pipeline
        candidatos.append(list(externo.named_steps.values())[-1])
    for interno in candidatos:
        for atributo in ("feature_names_in_", "feature_names_"):
            cols = getattr(interno, atributo, None)
            if cols is not None:
                return [str(c) for c in cols]
        if hasattr(interno, "get_booster"):
            cols = interno.get_booster().feature_names
            if cols:
                return list(cols)
    raise ValueError(f"No se pueden recuperar las columnas de {type(externo).__name__}")


def dividir_train_test(X: pd.DataFrame, y: pd.Series, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Corte cronológico de la sección 14 (mismas reglas, mismos redondeos)."""
    good = X.notna().all(axis=1) & y.notna()
    pos = np.flatnonzero(good.to_numpy())
    n_good = len(pos)
    if n_good < cfg["min_train_days"] * 96 + 96:
        raise ValueError("Historial insuficiente")
    n_tr = int(round(n_good * cfg["train_frac"]))
    n_tr = min(max(n_tr, cfg["min_train_days"] * 96), n_good - 96)
    tr = np.zeros(len(X), dtype=bool)
    tr[pos[:n_tr]] = True
    te = np.zeros(len(X), dtype=bool)
    te[pos[n_tr:]] = True
    if tr.sum() < cfg["min_train_days"] * 96 or te.sum() < 96:
        raise ValueError("Historial insuficiente")
    return tr, te


def elegir_ganadores_ml(metricas: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (medidor, variable): el modelo ML con menor CV_RMSE (nunca un benchmark)."""
    ml = metricas[metricas["modelo"].isin(MODELOS_ML) & metricas["CV_RMSE"].notna()]
    if ml.empty:
        raise ValueError("metricas.csv no tiene ningún modelo ML con CV_RMSE")
    idx = ml.groupby(["medidor", "variable"])["CV_RMSE"].idxmin()
    return ml.loc[idx].reset_index(drop=True)


def clasificar_serie(cv_rmse: float, ultimo_dato: pd.Timestamp, fecha_referencia: pd.Timestamp,
                     max_dias_sin_datos: int = 180, tol_constante: float = 1e-6) -> tuple[str, str]:
    """(estado, motivo). Las series sin datos recientes o constantes no se proyectan con ML."""
    dias = (fecha_referencia - ultimo_dato).days
    if dias > max_dias_sin_datos:
        return ESTADO_SIN_DATOS, (f"Último dato {ultimo_dato:%Y-%m-%d}, {dias} días antes del último dato "
                                  f"global ({fecha_referencia:%Y-%m-%d}); umbral {max_dias_sin_datos} días")
    if abs(cv_rmse) <= tol_constante:
        return ESTADO_CONSTANTE, "Objetivo constante en las ventanas de validación (CV_RMSE ≈ 0)"
    return ESTADO_OK, ""


def restaurar_estado(run_dir, *, cfg: dict, tokens: dict, load_meter: Callable, features: Callable,
                     metrics: Callable, tolerancia_rel: float = 1e-3,
                     imprimir: Callable[[str], None] = print) -> dict:
    """Reconstruye ``RESULTS``, ``PRED`` y las columnas/metadatos de cada serie.

    Devuelve un dict con: ``results`` (lista de filas), ``pred`` (dict), ``columnas``
    ({(medidor, variable, modelo): [...]}) , ``series`` ({(medidor, variable): {...}}) y
    ``validacion`` (DataFrame con el RMSE guardado vs. restaurado de cada modelo).
    """
    from prodma.recursivo import cargar_modelo

    run_dir = Path(run_dir)
    verificar_config(run_dir / "config.json", cfg)
    metr = pd.read_csv(run_dir / "tablas" / "metricas.csv")
    dias = cfg.get("train_days_" + cfg["modo"])
    pred, columnas, series, validacion = {}, {}, {}, []
    inicio = time.time()
    grupos = list(metr.groupby(["medidor", "variable"], sort=False))
    for n, ((m, var), sub) in enumerate(grupos, 1):
        y = load_meter(m)[var].replace([np.inf, -np.inf], np.nan)
        last = y.last_valid_index()
        y = y.loc[:last] if dias is None else y.loc[last - pd.Timedelta(days=dias + cfg["test_days"] + 10):last]
        X = features(y)
        tr, te = dividir_train_test(X, y, cfg)
        Xe, ye = X.loc[te], y.loc[te]
        series[(m, var)] = dict(last=last, test_inicio=Xe.index[0], test_fin=Xe.index[-1],
                                n_train=int(tr.sum()), n_test=int(te.sum()))
        for fila in sub.itertuples(index=False):
            nombre = fila.modelo
            if nombre.startswith("Estacional_"):
                pr = Xe[f"lag_{int(nombre.split('_')[1])}"]
            else:
                ruta = run_dir / "modelos" / f"{tokens[m]}_{var}_{nombre}.joblib"
                est = cargar_modelo(ruta)
                cols = columnas_de_modelo(est)
                columnas[(m, var, nombre)] = cols
                pr = est.predict(Xe[cols])
                del est
            pred[(m, var, nombre)] = pd.DataFrame({"real": ye, "pred": pr})
            rmse = metrics(ye, pr)["RMSE"]
            guardado = float(fila.RMSE)
            dif = abs(rmse - guardado) / max(abs(guardado), 1e-12) if np.isfinite(guardado) else np.nan
            validacion.append(dict(medidor=m, variable=var, modelo=nombre, RMSE_guardado=guardado,
                                   RMSE_restaurado=rmse, dif_rel=dif))
        imprimir(f"Restaurado {m}/{var} ({n}/{len(grupos)}); {time.time() - inicio:.0f} s")
    validacion = pd.DataFrame(validacion)
    malas = validacion[(validacion.dif_rel > tolerancia_rel) | validacion.RMSE_restaurado.isna()]
    if len(malas):
        raise RestauracionInconsistente(
            f"{len(malas)} de {len(validacion)} modelos no reproducen su RMSE guardado (tolerancia relativa "
            f"{tolerancia_rel:g}); primeros casos:\n{malas.head(10).to_string(index=False)}"
        )
    return dict(results=metr.to_dict("records"), pred=pred, columnas=columnas, series=series,
                validacion=validacion)


def construir_ganadores(metricas: pd.DataFrame, estado: dict, run_dir, tokens: dict,
                        fecha_referencia: pd.Timestamp | None = None, max_dias_sin_datos: int = 180) -> dict:
    """``WINNERS`` operativo: {(medidor, variable): {...}} sin cargar los modelos en memoria.

    ``model`` queda en ``None`` y ``ruta_modelo`` apunta al ``.joblib``: cada modelo se carga
    solo cuando se usa (en el proceso que lo necesita), en vez de mantener en RAM hasta 145
    modelos —algunos RandomForest de >180 MB—, que fue la causa probable del ``MemoryError``.
    """
    run_dir = Path(run_dir)
    ganadores = elegir_ganadores_ml(metricas)
    if fecha_referencia is None:
        fecha_referencia = max(s["last"] for s in estado["series"].values())
    winners = {}
    for g in ganadores.itertuples(index=False):
        m, var, nombre = g.medidor, g.variable, g.modelo
        meta = estado["series"][(m, var)]
        est, motivo = clasificar_serie(g.CV_RMSE, meta["last"], fecha_referencia, max_dias_sin_datos)
        winners[(m, var)] = dict(
            name=nombre, model=None, ruta_modelo=str(run_dir / "modelos" / f"{tokens[m]}_{var}_{nombre}.joblib"),
            cv_rmse=float(g.CV_RMSE), last=meta["last"], columns=estado["columnas"][(m, var, nombre)],
            test_inicio=meta["test_inicio"], estado=est, motivo=motivo,
        )
    return winners


def guardar_estado(run_dir, estado: dict, winners: dict) -> Path:
    import joblib

    ruta = Path(run_dir) / "tablas" / ARCHIVO_ESTADO
    joblib.dump(dict(estado=estado, winners=winners), ruta, compress=3)
    return ruta


def cargar_estado(run_dir) -> tuple[dict, dict] | None:
    import joblib

    ruta = Path(run_dir) / "tablas" / ARCHIVO_ESTADO
    if not ruta.exists():
        return None
    d = joblib.load(ruta)
    return d["estado"], d["winners"]
