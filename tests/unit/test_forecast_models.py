import importlib.util

import numpy as np
import pandas as pd
import pytest

from prodma.forecast_models import (
    MissingOptionalDependency,
    available_model_names,
    create_forecaster,
    resolve_torch_device,
)


@pytest.fixture
def monthly_series():
    index = pd.date_range("2018-01-01", periods=72, freq="MS")
    values = 100 + np.arange(72) * 0.4 + 8 * np.sin(2 * np.pi * np.arange(72) / 12)
    return pd.Series(values, index=index, name="energia_mwh")


def test_registry_contains_every_model_required_by_source_plan():
    required = {
        "SeasonalNaive",
        "ARIMA",
        "SARIMA",
        "HoltWinters",
        "SARIMAX",
        "Prophet",
        "RandomForest",
        "XGBoost",
        "LSTM",
    }
    assert required <= set(available_model_names())


def test_seasonal_naive_repeats_last_complete_season(monthly_series):
    model = create_forecaster("SeasonalNaive", seasonal_periods=12)
    model.fit(monthly_series)
    prediction = model.predict(12)
    np.testing.assert_allclose(prediction, monthly_series.iloc[-12:].to_numpy())


@pytest.mark.parametrize("name", ["ARIMA", "SARIMA", "HoltWinters"])
def test_statistical_forecasters_return_finite_horizon(name, monthly_series):
    model = create_forecaster(name, seasonal_periods=12)
    model.fit(monthly_series)
    prediction = model.predict(6)
    assert prediction.shape == (6,)
    assert np.isfinite(prediction).all()


def test_sarimax_requires_and_uses_future_exogenous_values(monthly_series):
    exog = pd.DataFrame(
        {"temperatura_c": np.linspace(18, 28, len(monthly_series))},
        index=monthly_series.index,
    )
    model = create_forecaster("SARIMAX", seasonal_periods=12)
    model.fit(monthly_series, exog=exog)
    with pytest.raises(ValueError, match="exógenas futuras"):
        model.predict(3)
    prediction = model.predict(
        3,
        exog=pd.DataFrame(
            {"temperatura_c": [24.0, 25.0, 26.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="MS"),
        ),
    )
    assert np.isfinite(prediction).all()


def test_prophet_dependency_is_explicit(monthly_series):
    model = create_forecaster("Prophet", seasonal_periods=12)
    if importlib.util.find_spec("prophet") is None:
        with pytest.raises(MissingOptionalDependency, match="prophet"):
            model.fit(monthly_series)


def test_prophet_returns_finite_monthly_forecast_when_installed(monthly_series):
    if importlib.util.find_spec("prophet") is None:
        pytest.skip("prophet no instalado")
    model = create_forecaster("Prophet", seasonal_periods=12)
    model.fit(monthly_series)
    prediction = model.predict(3)
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()


def test_prophet_enables_weekly_seasonality_only_for_daily_or_finer_data():
    if importlib.util.find_spec("prophet") is None:
        pytest.skip("prophet no instalado")
    daily_index = pd.date_range("2023-01-01", periods=400, freq="D")
    daily_values = 100 + 5 * np.sin(2 * np.pi * np.arange(400) / 7)
    daily_series = pd.Series(daily_values, index=daily_index)

    model_diario = create_forecaster("Prophet", seasonal_periods=7)
    model_diario.fit(daily_series)
    assert model_diario.model_.weekly_seasonality is True

    model_mensual = create_forecaster("Prophet", seasonal_periods=12)
    model_mensual.fit(pd.Series(
        100 + np.arange(72) * 0.4,
        index=pd.date_range("2018-01-01", periods=72, freq="MS"),
    ))
    assert model_mensual.model_.weekly_seasonality is False


def test_lstm_dependency_and_device_resolution_are_explicit(monthly_series):
    assert resolve_torch_device("cpu") == "cpu"
    if importlib.util.find_spec("torch") is None:
        with pytest.raises(MissingOptionalDependency, match="torch"):
            resolve_torch_device("gpu")
        model = create_forecaster("LSTM", seasonal_periods=12, device="gpu")
        with pytest.raises(MissingOptionalDependency, match="torch"):
            model.fit(monthly_series)


def test_lstm_runs_on_cuda_when_available(monthly_series):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch no instalado")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA no disponible para torch")
    model = create_forecaster(
        "LSTM", seasonal_periods=12, device="gpu", epochs=2, hidden_size=8
    )
    model.fit(monthly_series)
    prediction = model.predict(3)
    assert model.device_ == "cuda:0"
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Modelo desconocido"):
        create_forecaster("modelo_inventado")
