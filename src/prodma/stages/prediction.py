"""Etapa 5: prediccion operativa recursiva del mejor modelo admitido por serie.

Solo pronostica cuando el modelo no requiere exogenas futuras. Cualquier otra
situacion queda con un estado explicito en el resumen; nunca se omite en silencio.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prodma.artifacts import ManifestStore, RunLayout
from prodma.modeling.origenes import origin_forecasts, recursive_forecast
from prodma.modeling.intervals import calibrate_rolling
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


def run_prediction(config, layout: RunLayout) -> StageResult:
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
    registry = model_registry()
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
            training_dir = training_root / "series" / target
            compatibility = pd.read_csv(training_dir / "csv" / "compatibilidad_exogenas.csv")
            row = compatibility.loc[compatibility["modelo"] == model_name].iloc[0]
            if int(row["exogenas"]) > 0 and bool(row["exogenas_usadas"]):
                summary.append({"serie": target, "estado": "REQUIERE_EXOGENAS_FUTURAS", "modelo": model_name})
                continue
            if registry[model_name].family != "sklearn":
                summary.append({"serie": target, "estado": "NO_SOPORTADO_T16", "modelo": model_name})
                continue
            envelope = joblib.load(training_dir / "checkpoints" / f"{model_name}.joblib")
            columns = list(envelope["key"].columns)
            model = joblib.load(training_dir / "modelos" / f"{model_name}.joblib")
            clean = _quality_for_target(quality, target)
            forecast = recursive_forecast(model, columns, clean, horizon)
            _atomic_csv(forecast, output_root / "series" / target / "csv" / "pronostico_operativo.csv")
            scenario_state = "SIN_TABLA_ESCENARIOS"
            scenario_path = Path(stage_two.get("artifacts", {}).get("new_customer_scenario", ""))
            meter = target.rsplit("_", 1)[0]
            if scenario_path.is_file():
                customer_table = pd.read_parquet(scenario_path)
                if meter in set(customer_table["medidor"].astype(str)):
                    base = pd.DataFrame({"timestamp": pd.to_datetime(forecast["timestamp"]),
                                         "prediccion": forecast["prediccion"].to_numpy()})
                    increments = customer_increments(customer_table, meter, base["timestamp"])
                    scenarios = apply_scenarios(base, increments)
                    _atomic_csv(scenarios, output_root / "series" / target / "csv" / "pronostico_escenarios.csv")
                    scenario_state = "ESCENARIOS_APLICADOS"
                else:
                    scenario_state = "SIN_ESCENARIO_MEDIDOR"
            final_stamps = pd.to_datetime(pd.read_csv(training_dir / "csv" / "predicciones_test_final.csv")["timestamp"])
            alpha = float(config.raw.get("evaluation", {}).get("alpha", 0.1))
            coverage = pd.concat([
                calibrate_rolling(origin_forecasts(model, columns, clean, final_stamps, int(h)), int(h), alpha)
                for h in sorted(config.horizons)
            ], ignore_index=True)
            _atomic_csv(coverage, output_root / "series" / target / "csv" / "calibracion_rolling.csv")
            summary.append({"serie": target, "estado": "PRONOSTICADA", "modelo": model_name, "escenarios": scenario_state,
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
