import numpy as np
import pandas as pd
import pytest

from prodma.forecast_models import ARIMAForecaster, SARIMAForecaster
from prodma.modeling.registry import exogenous_used, model_registry


def _series_with_exogenous(n=240, seed=4):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    driver = rng.normal(size=n)
    y = pd.Series(10 + 3 * driver + rng.normal(scale=0.2, size=n), index=index)
    exog = pd.DataFrame({"temp": driver}, index=index)
    return y, exog


def test_arima_uses_future_exogenous_values_in_its_forecast():
    y, exog = _series_with_exogenous()
    model = ARIMAForecaster(order=(1, 0, 0)).fit(y, exog=exog)
    future_index = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=5, freq="D")
    low = pd.DataFrame({"temp": np.full(5, -2.0)}, index=future_index)
    high = pd.DataFrame({"temp": np.full(5, 2.0)}, index=future_index)
    assert not np.allclose(model.predict(5, exog=low), model.predict(5, exog=high))


def test_arima_rejects_future_exogenous_with_wrong_columns_or_length():
    y, exog = _series_with_exogenous()
    model = ARIMAForecaster(order=(1, 0, 0)).fit(y, exog=exog)
    future_index = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=5, freq="D")
    with pytest.raises(ValueError):
        model.predict(5, exog=pd.DataFrame({"otra": np.zeros(5)}, index=future_index))
    with pytest.raises(ValueError):
        model.predict(5, exog=pd.DataFrame({"temp": np.zeros(3)}, index=future_index[:3]))


def test_arima_without_exogenous_still_fits_and_predicts():
    y, _ = _series_with_exogenous()
    model = ARIMAForecaster(order=(1, 0, 0)).fit(y)
    assert len(model.predict(4)) == 4


def test_sarima_uses_future_exogenous_values_in_its_forecast():
    y, exog = _series_with_exogenous(n=120)
    model = SARIMAForecaster(seasonal_periods=7).fit(y, exog=exog)
    future_index = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=3, freq="D")
    low = pd.DataFrame({"temp": np.full(3, -2.0)}, index=future_index)
    high = pd.DataFrame({"temp": np.full(3, 2.0)}, index=future_index)
    assert not np.allclose(model.predict(3, exog=low), model.predict(3, exog=high))


def test_registry_reports_exogenous_use_per_forecaster_family():
    registry = model_registry()
    assert exogenous_used(registry["ARIMA"]) is True
    assert exogenous_used(registry["SARIMA"]) is True
    assert exogenous_used(registry["SARIMAX"]) is True
    assert exogenous_used(registry["SeasonalNaive"]) is False
    assert exogenous_used(registry["HoltWinters"]) is False
    assert exogenous_used(registry["Prophet"]) is False
