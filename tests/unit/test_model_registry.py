from prodma.modeling.registry import model_registry


def test_registry_contains_all_requested_candidates():
    expected = {"SeasonalNaive", "Ridge", "RandomForest", "XGBoost", "LightGBM", "CatBoost",
                "MLP", "ARIMA", "SARIMA", "SARIMAX", "HoltWinters", "Prophet", "LSTM"}
    assert expected <= set(model_registry())


def test_each_model_declares_resolution_backend_and_factory():
    for spec in model_registry().values():
        assert spec.resolution in {"15min", "1h", "1d"}
        assert spec.backend in {"cpu", "auto"}
        assert callable(spec.factory)
