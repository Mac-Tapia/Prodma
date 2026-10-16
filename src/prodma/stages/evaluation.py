"""Etapa 4: evaluacion fuera de muestra sobre las predicciones de validacion de la etapa 3."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from prodma.artifacts import ManifestStore, RunLayout
from prodma.evaluacion.etapa import evaluate_series
from prodma.stages import StageResult

STAGE = "04_evaluacion"


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _fail(layout: RunLayout, message: str) -> StageResult:
    ManifestStore(layout).update_stage(STAGE, "FAILED", error=message)
    return StageResult(STAGE, "FAILED", layout.stage(STAGE), warnings=(message,))


def run_evaluation(config, layout: RunLayout) -> StageResult:
    manifest = ManifestStore(layout).read().get("stages", {})
    stage_three = manifest.get("03_entrenamiento", {})
    if stage_three.get("status") not in {"SUCCESS", "PARTIAL"}:
        return _fail(layout, "La etapa 3 no tiene estado SUCCESS o PARTIAL")
    training_root = layout.stage("03_entrenamiento")
    output_root = layout.stage(STAGE)
    tolerance = float(config.raw.get("evaluation", {}).get("coverage_tolerance", 0.05))
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
            )
            target = output_root / "series" / series_dir.name
            _atomic_csv(result["ranking"], target / "csv" / "ranking.csv")
            _atomic_csv(result["inferencia"]["pairwise"], target / "csv" / "inferencia_pareada.csv")
            admitted = result["ranking"].loc[result["ranking"]["admitido"], "modelo"].tolist()
            summary.append({"serie": series_dir.name, "estado": "EVALUADA",
                            "mejor_modelo": result["inferencia"]["mejor_modelo"],
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
