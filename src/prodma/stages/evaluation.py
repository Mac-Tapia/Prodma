"""Etapa 4: evaluacion fuera de muestra sobre las predicciones de validacion de la etapa 3."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prodma.artifacts import ManifestStore, RunLayout
from prodma.evaluacion.etapa import evaluate_series
from prodma.modeling.intervals import calibrate_final_test, calibrate_rolling
from prodma.modeling.origenes import origin_forecasts
from prodma.modeling.registry import model_registry
from prodma.stages import StageResult
from prodma.stages.training import _quality_for_target

STAGE = "04_evaluacion"


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _fail(layout: RunLayout, message: str) -> StageResult:
    ManifestStore(layout).update_stage(STAGE, "FAILED", error=message)
    return StageResult(STAGE, "FAILED", layout.stage(STAGE), warnings=(message,))


def _calibration_errors(config, layout: RunLayout, series: str, training_dir: Path, quality: pd.DataFrame,
                        alpha: float) -> dict[str, float]:
    """Desviacion media absoluta de la cobertura respecto a 1 - alpha, por modelo sin exogenas futuras.

    Los modelos que requieren exogenas futuras o no son sklearn quedan fuera (NO_EVALUADA en el ranking).
    """
    compatibility = pd.read_csv(training_dir / "csv" / "compatibilidad_exogenas.csv")
    trained = pd.read_csv(training_dir / "csv" / "entrenamientos.csv")
    registry = model_registry()
    final_stamps = pd.to_datetime(pd.read_csv(training_dir / "csv" / "predicciones_test_final.csv")["timestamp"])
    clean = _quality_for_target(quality, series)
    errors: dict[str, float] = {}
    for name in trained.loc[trained["estado"] == "ENTRENADO", "modelo"]:
        row = compatibility.loc[compatibility["modelo"] == name].iloc[0]
        if int(row["exogenas"]) > 0 and bool(row["exogenas_usadas"]):
            continue
        if registry[name].family != "sklearn":
            continue
        columns = list(joblib.load(training_dir / "checkpoints" / f"{name}.joblib")["key"].columns)
        model = joblib.load(training_dir / "modelos" / f"{name}.joblib")
        tables = [
            calibrate_rolling(origin_forecasts(model, columns, clean, final_stamps, int(h)), int(h), alpha)
            for h in sorted(config.horizons)
        ]
        coverage = pd.concat(tables)["cobertura"].to_numpy(dtype=float)
        errors[name] = float(np.abs(coverage - (1 - alpha)).mean())
    return errors


def run_evaluation(config, layout: RunLayout) -> StageResult:
    manifest = ManifestStore(layout).read().get("stages", {})
    stage_three = manifest.get("03_entrenamiento", {})
    if stage_three.get("status") not in {"SUCCESS", "PARTIAL"}:
        return _fail(layout, "La etapa 3 no tiene estado SUCCESS o PARTIAL")
    training_root = layout.stage("03_entrenamiento")
    output_root = layout.stage(STAGE)
    tolerance = float(config.raw.get("evaluation", {}).get("coverage_tolerance", 0.05))
    alpha = float(config.raw.get("evaluation", {}).get("alpha", 0.1))
    stage_two_artifacts = manifest.get("02_limpieza", {}).get("artifacts", {})
    quality_path = Path(stage_two_artifacts.get("quality", ""))
    if not quality_path.is_file():
        return _fail(layout, "Falta matriz de calidad de la etapa 2")
    quality = pd.read_parquet(quality_path)
    try:
        series_dirs = sorted(path for path in (training_root / "series").glob("*") if path.is_dir())
        if not series_dirs:
            return _fail(layout, "La etapa 3 no tiene series entrenadas que evaluar")
        summary: list[dict] = []
        for series_dir in series_dirs:
            predictions_path = series_dir / "csv" / "predicciones_validacion.csv"
            if not predictions_path.is_file():
                summary.append({"serie": series_dir.name, "estado": "SIN_PREDICCIONES"})
                continue
            result = evaluate_series(
                pd.read_csv(predictions_path),
                pd.read_csv(series_dir / "csv" / "entrenamientos.csv"),
                pd.read_csv(series_dir / "times" / "tiempos.csv"),
                pd.read_csv(series_dir / "csv" / "prefiltro.csv"),
                coverage_tolerance=tolerance,
                calibration_errors=_calibration_errors(config, layout, series_dir.name, series_dir, quality, alpha),
            )
            target = output_root / "series" / series_dir.name
            _atomic_csv(result["ranking"], target / "csv" / "ranking.csv")
            _atomic_csv(result["inferencia"]["pairwise"], target / "csv" / "inferencia_pareada.csv")
            final_path = series_dir / "csv" / "predicciones_test_final.csv"
            if final_path.is_file():
                intervals = calibrate_final_test(pd.read_csv(final_path), config.horizons, alpha)
                _atomic_csv(intervals, target / "csv" / "calibracion_intervalos.csv")
            admitted = result["ranking"].loc[result["ranking"]["admitido"], "modelo"].tolist()
            summary.append({"serie": series_dir.name, "estado": "EVALUADA",
                            "mejor_modelo": result["inferencia"]["mejor_modelo"],
                            "mcs": ";".join(result["inferencia"]["mcs"]),
                            "admitidos": ";".join(admitted),
                            "bloques_no_alineados": result["inferencia"]["bloques_no_alineados"]})
        summary_frame = pd.DataFrame(summary)
        _atomic_csv(summary_frame, output_root / "csv" / "resumen_evaluacion.csv")
        evaluated = summary_frame.loc[summary_frame["estado"] == "EVALUADA", "serie"].tolist()
        status = "SUCCESS" if len(evaluated) == len(series_dirs) else ("PARTIAL" if evaluated else "FAILED")
        artifacts = {"evaluadas": evaluated, "series_totales": len(series_dirs)}
        ManifestStore(layout).update_stage(STAGE, status, artifacts=artifacts)
        return StageResult(STAGE, status, output_root, artifacts)
    except Exception as exc:
        return _fail(layout, f"{type(exc).__name__}: {exc}")
