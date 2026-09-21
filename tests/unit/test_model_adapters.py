import numpy as np
import pandas as pd
import pytest

from prodma.modeling.adapters import ForecastAdapter, ForecastRequest, TrainingSlice
from prodma.modeling.prefilter import PrefilterPolicy, run_prefilter
from prodma.modeling.registry import model_registry


@pytest.fixture
def training_slice():
    index = pd.date_range("2025-01-01", periods=96 * 22, freq="15min")
    y = pd.Series(100 + 10 * np.sin(2 * np.pi * np.arange(len(index)) / 96), index=index)
    x = pd.DataFrame({"lag_1": y.shift(1), "hora": index.hour}, index=index).bfill()
    return TrainingSlice(x=x, y=y, train_positions=np.arange(96 * 21),
                         validation_positions=np.arange(96 * 21, 96 * 22))


@pytest.mark.parametrize("name", ["Ridge", "SeasonalNaive"])
def test_prefilter_accepts_sklearn_and_series_forecasters(name, training_slice):
    result = run_prefilter({name: model_registry()[name]}, training_slice,
                           PrefilterPolicy(max_seconds=15), seed=42)
    assert result.iloc[0].estado == "APROBADO", result.iloc[0].error
    assert result.iloc[0].backend_efectivo in {"cpu", "cuda:0"}
    assert result.iloc[0].memoria_pico_mb > 0


def test_adapter_returns_timestamped_prediction(training_slice):
    spec = model_registry()["Ridge"]
    adapter = ForecastAdapter(spec, seed=42).fit(training_slice)
    positions = training_slice.validation_positions
    request = ForecastRequest(
        x=training_slice.x.iloc[positions], index=training_slice.y.index[positions], horizon=len(positions)
    )
    prediction = adapter.predict(request)
    assert prediction.index.equals(request.index)
    assert np.isfinite(prediction).all()


def test_registered_prefilter_enforces_wall_clock_timeout(training_slice):
    result = run_prefilter(
        {"Ridge": model_registry()["Ridge"]}, training_slice,
        PrefilterPolicy(max_seconds=0.001), seed=42,
    )
    assert result.iloc[0].estado == "TIMEOUT"
