"""Etapa 3: entrenamiento temporal aislado por serie, trazable y reanudable.

Cada serie candidata se prepara por separado. Una serie no elegible queda registrada con su
motivo; ninguna se omite en silencio. Las elegibles se entrenan en `series/<serie>/` con sus
propios modelos, checkpoints, tiempos y trazas. El test final no se usa en la selección.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prodma.artifacts import ManifestStore, RunLayout, sha256_file
from prodma.modeling.adapters import TrainingSlice
from prodma.modeling.ablation import FeatureSetBundle, compare_feature_sets
from prodma.modeling.features import join_exogenous_features, make_features
from prodma.modeling.prefilter import PrefilterPolicy, run_prefilter
from prodma.modeling.registry import exogenous_compatibility, exogenous_used, model_registry
from prodma.modeling.splits import nested_temporal_partition
from prodma.modeling.train import build_training_slice, train_approved_models
from prodma.stages import StageResult

SERIES_DIR = "series"
PARTS = ("csv", "modelos", "checkpoints", "times", "traces", "figuras", "logs")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _prepare(candidate: str, matrix: pd.DataFrame, quality: pd.DataFrame,
             exogenous: pd.DataFrame, config) -> dict:
    """Prepara una serie; lanza ValueError con el motivo si no es elegible."""
    training_cfg = config.raw.get("training", {})
    clean = _quality_for_target(quality, candidate)
    endogenous = make_features(clean)
    safe_exogenous = exogenous.drop(columns=list(set(endogenous).intersection(exogenous)), errors="ignore")
    features = join_exogenous_features(endogenous, safe_exogenous).dropna(axis=1, how="all")
    data = build_training_slice(clean, features)
    partition = nested_temporal_partition(
        data.y.index, config.horizons, int(training_cfg.get("folds", 3)),
        int(training_cfg.get("min_train", max(config.horizons))), config.frequency,
    )
    endogenous_columns = [column for column in data.x.columns if column in endogenous.columns]
    exogenous_columns = [column for column in data.x.columns if column not in endogenous.columns]
    ablation = None
    if exogenous_columns:
        positional_folds = [
            (data.y.index.get_indexer(fold.train_index), data.y.index.get_indexer(fold.validation_index))
            for fold in partition.inner_folds
        ]
        ablation = compare_feature_sets(
            FeatureSetBundle(endogenous=data.x[endogenous_columns], y=data.y),
            data.x[exogenous_columns], positional_folds,
            minimum_improvement=float(training_cfg.get("ablation_minimum_improvement", 0.0)),
        )
        selected = [column for column in ablation.selected_columns if column in features.columns]
        data = build_training_slice(clean, features.loc[:, selected])
    return {"clean": clean, "features": features, "data": data, "partition": partition, "ablation": ablation,
            "endogenous_columns": list(endogenous.columns)}


def _train_series(target: str, prepared: dict, names: tuple[str, ...], registry: dict, config,
                  output: Path, matrix_hash: str, exogenous_hash: str) -> dict:
    training_cfg = config.raw.get("training", {})
    data, partition = prepared["data"], prepared["partition"]
    columns = tuple(data.x.columns)
    ablation = prepared.get("ablation")
    if ablation is not None:
        _atomic_csv(ablation.fold_metrics, output / "csv" / "ablacion.csv")
    selected_exogenous = tuple(column for column in columns if column not in prepared["endogenous_columns"])
    compatibility = pd.DataFrame([
        {"serie": target, "modelo": name, "estado": exogenous_compatibility(registry[name], bool(selected_exogenous)),
         "exogenas": len(selected_exogenous), "exogenas_usadas": exogenous_used(registry[name])}
        for name in names
    ])
    _atomic_csv(compatibility, output / "csv" / "compatibilidad_exogenas.csv")
    incompatible = tuple(compatibility.loc[compatibility["estado"] != "COMPATIBLE", "modelo"])
    names = tuple(name for name in names if name not in incompatible)
    if not names:
        return {"serie": target, "estado": "FALLIDA", "motivo": "ningun modelo compatible con exogenas",
                "entrenados": 0, "incompatibles": len(incompatible)}
    if _compatible_restart(output, names, matrix_hash, exogenous_hash, columns, config.seed):
        return {"serie": target, "estado": "REUTILIZADA", "motivo": "checkpoints compatibles", "entrenados": len(names)}
    first = partition.inner_folds[0]
    positions = data.y.index.get_indexer(first.train_index)
    validation = data.y.index.get_indexer(first.validation_index)
    prefilter = run_prefilter(
        {name: registry[name] for name in names},
        TrainingSlice(data.x, data.y, positions, validation),
        PrefilterPolicy(float(training_cfg.get("prefilter_max_seconds", 300))), seed=config.seed,
    )
    _atomic_csv(prefilter, output / "csv" / "prefiltro.csv")
    approved = tuple(prefilter.loc[prefilter["estado"] == "APROBADO", "modelo"])
    if not approved:
        return {"serie": target, "estado": "FALLIDA", "motivo": "ningun modelo supero el prefiltro", "entrenados": 0}
    trained = train_approved_models(
        {name: registry[name] for name in approved}, data, partition,
        output / "checkpoints", matrix_hash, exogenous_hash, config.seed,
        int(training_cfg.get("n_trials", 5)), selected_feature_set="endogenas_mas_exogenas_asof",
    )
    _atomic_csv(trained.table, output / "csv" / "entrenamientos.csv")
    _atomic_csv(trained.timings, output / "times" / "tiempos.csv")
    _atomic_csv(trained.traces, output / "traces" / "optuna_trials.csv")
    _atomic_csv(trained.predictions, output / "csv" / "predicciones_validacion.csv")
    _atomic_csv(trained.final_test, output / "csv" / "predicciones_test_final.csv")
    _atomic_csv(pd.concat([prefilter.assign(evento="prefiltro"), trained.table.assign(evento="entrenamiento")],
                          ignore_index=True, sort=False), output / "logs" / "eventos.csv")
    for checkpoint in trained.checkpoints:
        envelope = joblib.load(checkpoint)
        model_target = output / "modelos" / checkpoint.name
        model_target.parent.mkdir(parents=True, exist_ok=True)
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
        (output / "figuras").mkdir(parents=True, exist_ok=True)
        figure.savefig(output / "figuras" / "inner_rmse.png", dpi=120)
        plt.close(figure)
    ok = int((trained.table["estado"] == "ENTRENADO").sum())
    estado = "ENTRENADA" if ok == len(names) else ("PARCIAL" if ok else "FALLIDA")
    return {"serie": target, "estado": estado, "motivo": "", "entrenados": ok,
            "solicitados": len(names), "aprobados_prefiltro": len(approved),
            "final_test_start": str(partition.final_test_index[0])}


def run_training(config, layout: RunLayout, model_names: tuple[str, ...] | None = None) -> StageResult:
    """Entrena todas las series elegibles; cada no elegible queda con su motivo."""
    output_root = layout.stage("03_entrenamiento")
    manifest = ManifestStore(layout)
    stage_two = manifest.read().get("stages", {}).get("02_limpieza", {})
    if stage_two.get("status") != "SUCCESS":
        return _fail(layout, "La etapa 2 no tiene estado SUCCESS")
    artifacts = stage_two.get("artifacts", {})
    data_dir = layout.stage("02_limpieza", "datos")
    matrix_path = Path(artifacts.get("matrix", data_dir / "matriz_limpia.parquet"))
    quality_path = Path(artifacts.get("quality", data_dir / "matriz_calidad.parquet"))
    exogenous_path = Path(artifacts.get("exogenous_historical", data_dir / "matriz_exogenas_historicas.parquet"))
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
        if unknown := sorted(set(names) - set(registry)):
            return _fail(layout, f"Modelos desconocidos: {unknown}")
        if not len(matrix.columns):
            return _fail(layout, "No hay series establecidas para entrenar")

        rows: list[dict] = []
        prepared: dict[str, dict] = {}
        for candidate in matrix.columns:
            try:
                prepared[candidate] = _prepare(candidate, matrix, quality, exogenous, config)
            except (ValueError, KeyError) as exc:
                rows.append({"serie": candidate, "estado": "NO_ELEGIBLE", "motivo": str(exc), "entrenados": 0})
        if not prepared:
            return _fail(layout, "Ninguna serie tiene test final observado y contiguo")
        for target, item in prepared.items():
            series_dir = output_root / SERIES_DIR / target
            rows.append(_train_series(target, item, names, registry, config, series_dir,
                                      matrix_hash, exogenous_hash))
        summary = pd.DataFrame(rows)
        _atomic_csv(summary, output_root / "csv" / "resumen_series.csv")
        reconciled = set(summary["serie"]) == set(matrix.columns)
        if not reconciled:
            return _fail(layout, "Conteo de series no reconciliado con la matriz")
        trained_series = summary.loc[summary["estado"].isin(["ENTRENADA", "REUTILIZADA"]), "serie"].tolist()
        partial = summary.loc[summary["estado"] == "PARCIAL", "serie"].tolist()
        status = ("SUCCESS" if len(trained_series) == len(prepared) and not partial
                  else ("PARTIAL" if trained_series or partial else "FAILED"))
        result_artifacts = {
            "matrix_sha256": matrix_hash, "exogenous_sha256": exogenous_hash,
            "series_totales": len(matrix.columns), "series_elegibles": len(prepared),
            "entrenadas": trained_series, "parciales": partial,
            "no_elegibles": summary.loc[summary["estado"] == "NO_ELEGIBLE", "serie"].tolist(),
            "reused": bool(len(trained_series) and set(summary["estado"]) == {"REUTILIZADA"}),
        }
        manifest.update_stage("03_entrenamiento", status, artifacts=result_artifacts)
        return StageResult("03_entrenamiento", status, output_root, result_artifacts)
    except Exception as exc:
        return _fail(layout, f"{type(exc).__name__}: {exc}")
