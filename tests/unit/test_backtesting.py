import pandas as pd
import pytest

from prodma.evaluacion.backtesting import rolling_origins


def test_rolling_origins_preserve_order_and_horizon():
    idx = pd.date_range("2025-01-01", periods=1000, freq="15min")
    splits = rolling_origins(idx, horizons=(96,), folds=2, min_train=672)
    assert len(splits) == 2
    assert all(len(split.validation_index) == 96 for split in splits)
    assert all(split.train_index[-1] < split.validation_index[0] for split in splits)


def test_insufficient_history_is_rejected():
    idx = pd.date_range("2025-01-01", periods=100, freq="15min")
    with pytest.raises(ValueError, match="insuficiente"):
        rolling_origins(idx, horizons=(96,), folds=2, min_train=96)
