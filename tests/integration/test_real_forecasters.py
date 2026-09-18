from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

from prodma.forecast_models import create_forecaster


ROOT = Path(__file__).resolve().parents[2]


def _real_monthly_power(sheet: str = "L-6605") -> pd.Series:
    pieces = []
    for year in range(2023, 2027):
        path = ROOT / "datahistorica" / f"demanda{year}.xlsx"
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            rows = workbook[sheet].iter_rows(values_only=True)
            header = list(next(rows))
            date_col = header.index("Local Time")
            power_col = header.index("kW tot mean")
            values = [(row[date_col], row[power_col]) for row in rows if row[date_col] is not None]
            pieces.append(pd.Series(
                [value for _, value in values],
                index=pd.to_datetime([date for date, _ in values]),
                dtype=float,
            ))
        finally:
            workbook.close()
    series = pd.concat(pieces).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    return series.resample("MS").mean().dropna()


@pytest.mark.integration
def test_required_forecasters_run_on_real_excel_monthly_series():
    series = _real_monthly_power()
    assert len(series) >= 40
    train, test = series.iloc[:-6], series.iloc[-6:]
    names = ["SeasonalNaive", "ARIMA", "SARIMA", "HoltWinters", "Prophet", "LSTM"]
    for name in names:
        kwargs = {"seasonal_periods": 12}
        if name == "LSTM":
            kwargs.update(device="gpu", epochs=2, hidden_size=8)
        model = create_forecaster(name, **kwargs)
        model.fit(train)
        prediction = model.predict(len(test))
        assert prediction.shape == (len(test),), name
        assert np.isfinite(prediction).all(), name
        if name == "LSTM":
            assert model.device_ == "cuda:0"
