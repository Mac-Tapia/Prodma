import numpy as np
import pandas as pd
import pytest

from prodma.evaluacion.mcs import model_confidence_set


def _losses(columns, n=300, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(1.0, 2.0, size=n)
    return pd.DataFrame({name: base + offset for name, offset in columns.items()})


def test_clearly_worse_models_are_eliminated_and_best_is_kept():
    losses = _losses({"A": 0.0, "B": 0.02, "C": 1.5, "D": 1.6})
    result = model_confidence_set(losses, alpha=0.1, block_length=10, replications=500, seed=1)
    assert "A" in result["mcs"]
    assert "D" not in result["mcs"]
    assert {"C", "D"} <= set(result["eliminados"])


def test_indistinguishable_models_all_remain_in_the_set():
    losses = _losses({"A": 0.0, "B": 0.0, "C": 0.0}, seed=2)
    result = model_confidence_set(losses, alpha=0.1, block_length=10, replications=500, seed=2)
    assert set(result["mcs"]) == {"A", "B", "C"}
    assert result["eliminados"] == []


def test_elimination_order_and_pvalues_are_reported():
    losses = _losses({"A": 0.0, "B": 0.5, "C": 2.0}, seed=3)
    result = model_confidence_set(losses, alpha=0.1, block_length=10, replications=500, seed=3)
    assert result["p_valores"]["A"] >= 0.1
    assert len(result["orden_eliminacion"]) == len(result["eliminados"])


def test_requires_at_least_two_models_and_enough_blocks():
    with pytest.raises(ValueError):
        model_confidence_set(pd.DataFrame({"A": [1.0, 2.0, 3.0]}), alpha=0.1)
    with pytest.raises(ValueError):
        model_confidence_set(pd.DataFrame({"A": [1.0, 2.0], "B": [1.0, 2.0]}), alpha=0.1)
