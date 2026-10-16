"""Etapa 5: prediccion operativa recursiva del mejor modelo admitido por serie.

- Modelos sin exogenas: pronostico recursivo.
- Modelos con exogenas: pronostico solo si se proveen las exogenas futuras de todo el
  horizonte; si faltan o estan incompletas, el estado lista cuales faltan.
- La cobertura con exogenas realizadas siempre se etiqueta EXOGENAS_REALIZADAS.
Ninguna situacion se omite en silencio.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from prodma.artifacts import ManifestStore, RunLayout
from prodma.exogenas.futuras import open_meteo_future_exog
from prodma.modeling.features import make_features
from prodma.modeling.intervals import calibrate_rolling
from prodma.modeling.origenes import origin_forecasts, recursive_forecast
from prodma.modeling.registry import model_registry
from prodma.prediccion.escenarios import apply_scenarios, customer_increments
from prodma.stages import StageResult
from prodma.stages.training import _quality_for_target

STAGE = "05_prediccion"


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _fail(layout: RunLayout, message: str) -> StageResult:
    ManifestStore(layout).update_stage(STAGE, "FAILED", error=message)
    return StageResult(STAGE, "FAILED", layout.stage(STAGE), warnings=(message,))


def _future_stamps(clean_index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    step = clean_index[1] - clean_index[0]
    return pd.DatetimeIndex([clean_index[-1] + step * k for k in range(1, horizon + 1)])


def _future_status(needed: list[str], future_exog: pd.DataFrame | None, stamps: pd.DatetimeIndex,
                   error: str | None = None) -> str:
    if not needed:
        return "NO_REQUIERE"
    if future_exog is None and error:
        return f"ERROR_EXOG_FUTURA:{error}"
    if future_exog is None:
        return f"NO_PROVISTAS:{','.join(needed)}"
    missing = [column for column in needed if column not in future_exog.columns]
    if missing:
        return f"NO_PROVISTAS:{','.join(missing)}"
    aligned = future_exog.reindex(stamps)
    if aligned[needed].isna().any(axis=None):
        return f"INCOMPLETAS:{','.join(needed)}"
    return "PROVISTAS"


def run_prediction(config, layout: RunLayout, future_exog: pd.DataFrame | None = None) -> StageResult:
    """`future_exog`: exogenas futuras indexadas por los pasos del horizonte (ver `_future_stamps`)."""
    manifest = ManifestStore(layout).read().get("stages", {})
    if manifest.get("04_evaluacion", {}).get("status") not in {"SUCCESS", "PARTIAL"}:
        return _fail(layout, "La etapa 4 no tiene estado SUCCESS o PARTIAL")
    stage_two = manifest.get("02_limpieza", {})
    quality_path = Path(stage_two.get("artifacts", {}).get("quality", ""))
    if not quality_path.is_file():
        return _fail(layout, "Falta matriz de calidad de la etapa 2")
    quality = pd.read_parquet(quality_path)
    training_root = layout.stage("03_entrenamiento")
    evaluation_root = layout.stage("04_evaluacion")
    output_root = layout.stage(STAGE)
    horizon = int(max(config.horizons))
    alpha = float(config.raw.get("evaluation", {}).get("alpha", 0.1))
    registry = model_registry()
    exog_path = Path(stage_two.get("artifacts", {}).get("exogenous_historical", ""))
    future_error = None
    section = config.raw.get("exogenas_futuras", {}).get("open_meteo", {})
    if future_exog is None and section.get("habilitado"):
        try:
            future_exog = open_meteo_future_exog(section, _future_stamps(quality.index, horizon))
        except Exception as exc:
            future_error = f"{type(exc).__name__}: {exc}"
    try:
        summary: list[dict] = []
        for series_dir in sorted(p for p in (evaluation_root / "series").glob("*") if p.is_dir()):
            target = series_dir.name
            ranking = pd.read_csv(series_dir / "csv" / "ranking.csv")
            admitted = ranking.loc[ranking["admitido"] == True].sort_values("posicion")  # noqa: E712
            if admitted.empty:
                summary.append({"serie": target, "estado": "SIN_MODELO_ADMITIDO", "modelo": ""})
                continue
            model_name = str(admitted.iloc[0]["modelo"])
            if registry[model_name].family != "sklearn":
                summary.append({"serie": target, "estado": "NO_SOPORTADO_T16", "modelo": model_name})
                continue
            training_dir = training_root / "series" / target
            envelope = joblib.load(training_dir / "checkpoints" / f"{model_name}.joblib")
            columns = list(envelope["key"].columns)
            model = joblib.load(training_dir / "modelos" / f"{model_name}.joblib")
            clean = _quality_for_target(quality, target)
            endogenous = list(make_features(clean).columns)
            needed = [column for column in columns if column not in endogenous]
            stamps = _future_stamps(clean.index, horizon)
            future_state = _future_status(needed, future_exog, stamps, future_error)
            final_stamps = pd.to_datetime(pd.read_csv(training_dir / "csv" / "predicciones_test_final.csv")["timestamp"])
            exog_history = None
            if needed:
                if not exog_path.is_file():
                    summary.append({"serie": target, "estado": "REQUIERE_EXOGENAS_FUTURAS", "modelo": model_name,
                                    "exogenas_futuras": "SIN_EXOG_HISTORICA"})
                    continue
                exog_history = pd.read_parquet(exog_path)[needed]
                conditional = pd.concat([
                    calibrate_rolling(
                        origin_forecasts(model, columns, clean, final_stamps, int(h), exog=exog_history),
                        int(h), alpha)
                    for h in sorted(config.horizons)
                ], ignore_index=True)
                conditional["condicion"] = "EXOGENAS_REALIZADAS"
                _atomic_csv(conditional, output_root / "series" / target / "csv" / "calibracion_condicionada_exogenas.csv")
            if future_state not in {"NO_REQUIERE", "PROVISTAS"}:
                summary.append({"serie": target, "estado": "REQUIERE_EXOGENAS_FUTURAS", "modelo": model_name,
                                "exogenas_futuras": future_state})
                continue
            exog_all = None
            if needed:
                future_part = future_exog.loc[stamps, needed]
                exog_all = pd.concat([exog_history, future_part]).sort_index()
                exog_all = exog_all[~exog_all.index.duplicated(keep="last")]
            forecast = recursive_forecast(model, columns, clean, horizon, exog=exog_all)
            _atomic_csv(forecast, output_root / "series" / target / "csv" / "pronostico_operativo.csv")
            scenario_state = "SIN_TABLA_ESCENARIOS"
            scenario_path = Path(stage_two.get("artifacts", {}).get("new_customer_scenario", ""))
            meter = target.rsplit("_", 1)[0]
            if scenario_path.is_file():
                customer_table = pd.read_parquet(scenario_path)
                if meter in set(customer_table["medidor"].astype(str)):
                    base = pd.DataFrame({"timestamp": pd.to_datetime(forecast["timestamp"]),
                                         "prediccion": forecast["prediccion"].to_numpy()})
                    customers_cfg = config.raw.get("clientes", {})
                    increments = customer_increments(
                        customer_table, meter, base["timestamp"],
                        ramp_months=float(customers_cfg.get("rampa_meses", 0.0)),
                        coincidence=float(customers_cfg.get("coincidencia", 1.0)),
                    )
                    scenarios = apply_scenarios(base, increments)
                    _atomic_csv(scenarios, output_root / "series" / target / "csv" / "pronostico_escenarios.csv")
                    scenario_state = "ESCENARIOS_APLICADOS"
                else:
                    scenario_state = "SIN_ESCENARIO_MEDIDOR"
            if not needed:
                coverage = pd.concat([
                    calibrate_rolling(origin_forecasts(model, columns, clean, final_stamps, int(h)), int(h), alpha)
                    for h in sorted(config.horizons)
                ], ignore_index=True)
                _atomic_csv(coverage, output_root / "series" / target / "csv" / "calibracion_rolling.csv")
            summary.append({"serie": target, "estado": "PRONOSTICADA", "modelo": model_name,
                            "escenarios": scenario_state, "exogenas_futuras": future_state,
                            "horizonte": horizon, "inicio": forecast["timestamp"].iloc[0]})
        summary_frame = pd.DataFrame(summary)
        _atomic_csv(summary_frame, output_root / "csv" / "resumen_prediccion.csv")
        done = summary_frame.loc[summary_frame["estado"] == "PRONOSTICADA", "serie"].tolist()
        status = "SUCCESS" if len(done) == len(summary_frame) and done else ("PARTIAL" if done else "FAILED")
        artifacts = {"pronosticadas": done, "series_totales": len(summary_frame)}
        ManifestStore(layout).update_stage(STAGE, status, artifacts=artifacts)
        return StageResult(STAGE, status, output_root, artifacts)
    except Exception as exc:
        return _fail(layout, f"{type(exc).__name__}: {exc}")
