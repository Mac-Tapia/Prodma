from dataclasses import dataclass

import numpy as np
import pandas as pd

from prodma.modeling.registry import model_registry
from prodma.modeling.splits import nested_temporal_partition
from prodma.modeling.train import build_training_slice, train_approved_models


def test_imputed_rows_are_context_but_never_training_targets():
    index = pd.date_range("2025-01-01", periods=10, freq="15min")
    clean = pd.DataFrame({
        "limpio": np.arange(10, dtype=float), "contexto": np.arange(10, dtype=float),
        "es_observado": [True] * 8 + [False, True], "es_imputado": [False] * 8 + [True, False],
    }, index=index)
    features = pd.DataFrame({"lag_1": np.arange(10, dtype=float)}, index=index)
    result = build_training_slice(clean, features)
    assert result.y.index.equals(clean.index[clean.es_observado])
    assert not result.y.index.intersection(clean.index[clean.es_imputado]).size
    assert result.context.index.equals(clean.index)


def test_training_uses_inner_folds_and_never_final_test(tmp_path):
    index = pd.date_range("2025-01-01", periods=1400, freq="15min")
    y = pd.Series(50 + np.sin(np.arange(1400) / 10), index=index)
    clean = pd.DataFrame({"limpio": y, "contexto": y, "es_observado": True, "es_imputado": False}, index=index)
    features = pd.DataFrame({"lag_1": y.shift(1).bfill(), "lag_4": y.shift(4).bfill()}, index=index)
    data = build_training_slice(clean, features)
    partition = nested_temporal_partition(data.y.index, horizons=(96,), folds=2, min_train=672)
    result = train_approved_models(
        {"Ridge": model_registry()["Ridge"]}, data, partition, tmp_path,
        matrix_sha256="matrix", exogenous_sha256="exog", seed=42, n_trials=2,
    )
    assert result.table.loc[0, "estado"] == "ENTRENADO"
    assert result.table.loc[0, "matrix_sha256"] == "matrix"
    assert pd.to_datetime(result.traces["validation_end"]).max() < partition.final_test_index.min()
    assert result.checkpoints[0].is_file()


def test_training_fits_missing_feature_transform_only_inside_each_fold(tmp_path):
    index = pd.date_range("2025-01-01", periods=1400, freq="15min")
    y = pd.Series(np.sin(np.arange(1400) / 10), index=index)
    clean = pd.DataFrame({"limpio": y, "contexto": y, "es_observado": True, "es_imputado": False}, index=index)
    feature = y.shift(1)
    feature.iloc[700] = np.nan
    data = build_training_slice(clean, pd.DataFrame({"lag_1": feature}, index=index))
    partition = nested_temporal_partition(data.y.index, horizons=(96,), folds=2, min_train=672)
    result = train_approved_models({"Ridge": model_registry()["Ridge"]}, data, partition, tmp_path,
                                   "matrix", "exog", seed=1, n_trials=1)
    assert result.table.loc[0, "estado"] == "ENTRENADO", result.table.loc[0, "error"]
