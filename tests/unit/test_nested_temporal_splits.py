import pandas as pd
import pytest

from prodma.modeling.splits import InsufficientHistory, nested_temporal_partition


INDEX = pd.date_range("2025-01-01", periods=20_000, freq="15min")


def test_final_test_is_disjoint_from_prefilter_selection_and_tuning():
    partition = nested_temporal_partition(INDEX, horizons=(96, 672, 2688), folds=3, min_train=2688)
    used = set().union(*(set(fold.train_index) | set(fold.validation_index) for fold in partition.inner_folds))
    assert used.isdisjoint(partition.final_test_index)
    assert partition.inner_folds[-1].validation_index[-1] < partition.final_test_index[0]
    assert len(partition.final_test_index) == 2688


def test_insufficient_history_has_explicit_state():
    with pytest.raises(InsufficientHistory) as error:
        nested_temporal_partition(INDEX[:3000], horizons=(96, 672, 2688), folds=3, min_train=2688)
    assert error.value.state == "HISTORIA_INSUFICIENTE"


def test_non_contiguous_final_test_is_rejected():
    broken = INDEX.delete(-50)
    with pytest.raises(ValueError, match="contiguo"):
        nested_temporal_partition(broken, horizons=(96,), folds=2, min_train=672)
