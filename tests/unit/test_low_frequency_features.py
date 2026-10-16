import pandas as pd

from prodma.external.low_frequency import step_feature


def test_step_feature_uses_only_points_published_before_each_instant():
    points = pd.DataFrame({
        "periodo": [pd.Timestamp("2024-06-30")],
        "valor": [1062346.0],
        "fecha_disponible": [pd.Timestamp("2024-08-01")],
    })
    index = pd.DatetimeIndex(["2024-07-15 00:00", "2024-08-02 00:00"])

    values = step_feature(index, points)

    assert pd.isna(values.iloc[0])
    assert values.iloc[1] == 1062346.0


def test_step_feature_is_never_back_filled_before_first_point():
    points = pd.DataFrame({
        "periodo": [pd.Timestamp("2025-06-30")],
        "valor": [1083494.0],
        "fecha_disponible": [pd.Timestamp("2025-08-01")],
    })

    values = step_feature(pd.DatetimeIndex(["2023-01-01", "2025-07-31"]), points)

    assert values.isna().all()
