"""Etapa 3: entrenamiento temporal aislado, trazable y reanudable."""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prodma.artifacts import ManifestStore, RunLayout, sha256_file
from prodma.modeling.adapters import TrainingSlice
from prodma.modeling.features import join_exogenous_features, make_features
from prodma.modeling.prefilter import PrefilterPolicy, run_prefilter
from prodma.modeling.registry import model_registry
from prodma.modeling.splits import nested_temporal_partition
from prodma.modeling.train import build_training_slice, train_approved_models
from prodma.stages import StageResult


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _fail(layout: RunLayout, message: str) -> StageResult:
    ManifestStore(layout).update_stage("03_entrenamiento", "FAILED", error=message)
    return StageResult("03_entrenamiento", "FAILED", layout.stage("03_entrenamiento"), warnings=(message,))


def _quality_for_target(quality: pd.DataFrame, target: str) -> pd.DataFrame:
    meter, variable = target.rsplit("_", 1)
    source_variable = "S_medida" if variable == "S" else variable
    prefix = f"{meter}__{source_variable}__"
    fields = {column[len(prefix):]: quality[column] for column in quality if column.startswith(prefix)}
    required = {"limpio", "contexto", "es_observado", "es_imputado"}
    if missing := required - set(fields):
        raise ValueError(f"{target}: faltan campos de calidad {sorted(missing)}")
    return pd.DataFrame(fields, index=quality.index)


def _compatible_restart(output: Path, names: tuple[str, ...], matrix_hash: str,
                        exogenous_hash: str, columns: tuple[str, ...], seed: int) -> bool:
    table_path = output / "csv" / "entrenamientos.csv"
    if not table_path.is_file():
        return False
    table = pd.read_csv(table_path)
    if set(table.get("modelo", ())) != set(names) or not (table.get("estado") == "ENTRENADO").all():
        return False
    for name in names:
        checkpoint = output / "checkpoints" / f"{name}.joblib"
        model = output / "modelos" / f"{name}.joblib"
        if not checkpoint.is_file() or not model.is_file():
            return False
        envelope = joblib.load(checkpoint)
        key = envelope.get("key")
        if key is None or (key.matrix_sha256, key.exogenous_sha256, key.columns, key.seed) != (
            matrix_hash, exogenous_hash, columns, seed
        ):
            return False
    return True


def run_training(config, layout: RunLayout, model_names: tuple[str, ...] | None = None) -> StageResult:
    """Entrena candidatos sin tocar el test final y conserva fallos por modelo."""
    output = layout.stage("03_entrenamiento")
    manifest = ManifestStore(layout)
    stage_two = manifest.read().get("stages", {}).get("02_limpieza", {})
    if stage_two.get("status") != "SUCCESS":
        return _fail(layout, "La etapa 2 no tiene estado SUCCESS")
    artifacts = stage_two.get("artifacts", {})
    matrix_path = Path(artifacts.get("matrix", layout.stage("02_limpieza", "datos") / "matriz_limpia.parquet"))
    quality_path = Path(artifacts.get("quality", layout.stage("02_limpieza", "datos") / "matriz_calidad.parquet"))
    exogenous_path = Path(artifacts.get("exogenous_historical", layout.stage("02_limpieza", "datos") / "matriz_exogenas_historicas.parquet"))
    try:
        if not matrix_path.is_file() or sha256_file(matrix_path) != artifacts.get("sha256"):
            return _fail(layout, "Hash de matriz limpia no coincide con etapa 2")
        if not quality_path.is_file():
            return _fail(layout, "Falta matriz de calidad de etapa 2")
        exogenous_hash = "none"
        if exogenous_path.is_file():
            exogenous_hash = sha256_file(exogenous_path)
            if artifacts.get("exogenous_sha256") != exogenous_hash:
                return _fail(layout, "Hash de matriz exogena no coincide con etapa 2")
        matrix_hash = sha256_file(matrix_path)
        matrix, quality = pd.read_parquet(matrix_path), pd.read_parquet(quality_path)
        exogenous = pd.read_parquet(exogenous_path) if exogenous_path.is_file() else pd.DataFrame(index=matrix.index)
        if not matrix.index.equals(quality.index) or not matrix.index.equals(exogenous.index):
            return _fail(layout, "Indices de demanda, calidad y exogenas desalineados")
        registry = model_registry()
        names = tuple(model_names or registry)
        unknown = sorted(set(names) - set(registry))
        if unknown:
            return _fail(layout, f"Modelos desconocidos: {unknown}")
        training_cfg = config.raw.get("training", {})
        targets = list(matrix.columns[: int(training_cfg.get("max_series", 1))])
        if not targets:
            return _fail(layout, "No hay series establecidas para entrenar")
        target = targets[0]
        clean = _quality_for_target(quality, target)
        endogenous = make_features(clean)
        safe_exogenous = exogenous.drop(columns=list(set(endogenous).intersection(exogenous)), errors="ignore")
        features = join_exogenous_features(endogenous, safe_exogenous)
        data = build_training_slice(clean, features)
        partition = nested_temporal_partition(
            data.y.index, config.horizons, int(training_cfg.get("folds", 3)),
            int(training_cfg.get("min_train", max(config.horizons))), config.frequency,
        )
        columns = tuple(data.x.columns)
        if _compatible_restart(output, names, matrix_hash, exogenous_hash, columns, config.seed):
            manifest.update_stage("03_entrenamiento", "SUCCESS", artifacts={"reused": True, "target": target})
            return StageResult("03_entrenamiento", "SUCCESS", output, {"reused": True, "target": target})
        first = partition.inner_folds[0]
        positions = data.y.index.get_indexer(first.train_index)
        validation = data.y.index.get_indexer(first.validation_index)
        prefilter = run_prefilter(
            {name: registry[name] for name in names},
            TrainingSlice(data.x, data.y, positions, validation),
            PrefilterPolicy(float(training_cfg.get("prefilter_max_seconds", 300))), seed=config.seed,
        )
        _atomic_csv(prefilter, output / "csv" / "prefiltro.csv")
        approved_names = tuple(prefilter.loc[prefilter["estado"] == "APROBADO", "modelo"])
        if not approved_names:
            return _fail(layout, "Ningun modelo supero el prefiltro")
        trained = train_approved_models(
            {name: registry[name] for name in approved_names}, data, partition,
            output / "checkpoints", matrix_hash, exogenous_hash, config.seed,
            int(training_cfg.get("n_trials", 5)), selected_feature_set="endogenas_mas_exogenas_asof",
        )
        _atomic_csv(trained.table, output / "csv" / "entrenamientos.csv")
        _atomic_csv(trained.timings, output / "times" / "tiempos.csv")
        _atomic_csv(trained.traces, output / "traces" / "optuna_trials.csv")
        events = pd.concat([
            prefilter.assign(evento="prefiltro"),
            trained.table.assign(evento="entrenamiento"),
        ], ignore_index=True, sort=False)
        _atomic_csv(events, output / "logs" / "eventos.csv")
        for checkpoint in trained.checkpoints:
            envelope = joblib.load(checkpoint)
            model_target = output / "modelos" / checkpoint.name
            temporary = model_target.with_suffix(".joblib.tmp")
            joblib.dump(envelope["payload"]["model"], temporary)
            temporary.replace(model_target)
        if not trained.traces.empty:
            figure, axis = plt.subplots(figsize=(8, 4))
            for name, group in trained.traces.groupby("modelo"):
                axis.plot(group.index, group["RMSE"], marker="o", label=name)
            axis.set(title=f"RMSE interno - {target}", ylabel="RMSE", xlabel="evaluacion")
            axis.legend()
            figure.tight_layout()
            figure.savefig(output / "figuras" / f"{target}_inner_rmse.png", dpi=120)
            plt.close(figure)
        trained_ok = int((trained.table["estado"] == "ENTRENADO").sum())
        status = "SUCCESS" if trained_ok == len(names) else ("PARTIAL" if trained_ok else "FAILED")
        result_artifacts = {
            "target": target, "matrix_sha256": matrix_hash, "exogenous_sha256": exogenous_hash,
            "requested": len(names), "approved": len(approved_names), "trained": trained_ok,
            "final_test_start": str(partition.final_test_index[0]),
        }
        manifest.update_stage("03_entrenamiento", status, artifacts=result_artifacts)
        return StageResult("03_entrenamiento", status, output, result_artifacts)
    except Exception as exc:
        return _fail(layout, f"{type(exc).__name__}: {exc}")
