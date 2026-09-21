"""Ejecuta solamente el prefiltro aislado de todos los candidatos registrados."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prodma.artifacts import RunLayout
from prodma.config import load_config
from prodma.modeling.adapters import TrainingSlice
from prodma.modeling.features import join_exogenous_features, make_features
from prodma.modeling.prefilter import PrefilterPolicy, run_prefilter
from prodma.modeling.registry import model_registry
from prodma.modeling.splits import nested_temporal_partition
from prodma.modeling.train import build_training_slice
from prodma.stages.training import _quality_for_target


def run_catalog_prefilter(config, layout: RunLayout, output: Path, timeout: float = 60) -> pd.DataFrame:
    data_dir = layout.stage("02_limpieza", "datos")
    matrix = pd.read_parquet(data_dir / "matriz_limpia.parquet")
    quality = pd.read_parquet(data_dir / "matriz_calidad.parquet")
    exogenous = pd.read_parquet(data_dir / "matriz_exogenas_historicas.parquet")
    for target in matrix:
        clean = _quality_for_target(quality, target)
        endogenous = make_features(clean)
        external = exogenous.drop(columns=list(set(endogenous).intersection(exogenous)), errors="ignore")
        features = join_exogenous_features(endogenous, external).dropna(axis=1, how="all")
        data = build_training_slice(clean, features)
        try:
            partition = nested_temporal_partition(data.y.index, config.horizons, 3,
                                                  max(config.horizons), config.frequency)
            break
        except ValueError:
            continue
    else:
        raise ValueError("ninguna serie admite el prefiltro temporal")
    fold = partition.inner_folds[0]
    training = TrainingSlice(
        data.x, data.y, data.y.index.get_indexer(fold.train_index),
        data.y.index.get_indexer(fold.validation_index),
    )
    result = run_prefilter(model_registry(), training, PrefilterPolicy(timeout), seed=config.seed)
    result.insert(0, "serie", target)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_catalog_prefilter(config, RunLayout.create(config.output_root, args.run_id),
                                   args.output, args.timeout)
    print(result[["modelo", "estado", "segundos", "backend_efectivo", "error"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
