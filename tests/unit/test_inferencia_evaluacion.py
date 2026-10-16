import numpy as np
import pandas as pd
import pytest

from prodma.evaluacion.inferencia import diebold_mariano, friedman_holm, paired_wilcoxon


def test_paired_wilcoxon_detects_consistently_better_candidate():
    rng = np.random.default_rng(1)
    base = rng.uniform(1.0, 2.0, size=40)
    better = base * 0.8
    result = paired_wilcoxon(pd.Series(better), pd.Series(base))
    assert result["n"] == 40
    assert result["mejor_diferencia"] < 0
    assert result["p_valor"] < 0.01


def test_paired_wilcoxon_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        paired_wilcoxon(pd.Series([1.0, 2.0], index=[0, 1]), pd.Series([1.0, 2.0], index=[0, 2]))


def test_diebold_mariano_is_significant_for_systematically_lower_loss():
    rng = np.random.default_rng(2)
    loss_b = pd.Series(rng.uniform(1.0, 2.0, size=300))
    loss_a = loss_b * 0.7
    result = diebold_mariano(loss_a, loss_b, horizon=4)
    assert result["estadistico"] < 0
    assert result["p_valor"] < 0.01
    assert result["lags"] == 3


def test_diebold_mariano_is_not_significant_for_identical_losses():
    rng = np.random.default_rng(3)
    loss = pd.Series(rng.uniform(1.0, 2.0, size=200))
    result = diebold_mariano(loss, loss.copy(), horizon=1)
    assert result["p_valor"] == 1.0


def test_friedman_holm_compares_each_model_against_the_best_mean():
    rng = np.random.default_rng(4)
    blocks = 30
    table = pd.DataFrame({
        "mejor": rng.uniform(1.0, 1.2, size=blocks),
        "medio": rng.uniform(1.5, 1.7, size=blocks),
        "peor": rng.uniform(2.0, 2.2, size=blocks),
    })
    result = friedman_holm(table)
    assert result["p_friedman"] < 0.001
    assert result["mejor_modelo"] == "mejor"
    pairwise = result["pairwise"].set_index("modelo")
    assert set(pairwise.index) == {"medio", "peor"}
    assert (pairwise["p_holm"] >= pairwise["p_valor"]).all()
    assert (pairwise["p_holm"] <= 1.0).all()


def test_friedman_holm_needs_at_least_three_models_and_blocks():
    with pytest.raises(ValueError):
        friedman_holm(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}))
