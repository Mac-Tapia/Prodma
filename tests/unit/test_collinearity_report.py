from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prodma.exogenas.matrix import collinearity_report


def test_duplicated_information_is_flagged_as_collinear():
    rng = np.random.default_rng(0)
    base = rng.normal(size=500)
    frame = pd.DataFrame({"a": base, "a_copia": base * 2 + 1, "independiente": rng.normal(size=500)})

    report = collinearity_report(frame, threshold=0.9).set_index(["variable_a", "variable_b"])

    assert report.loc[("a", "a_copia"), "estado"] == "REVISAR_COLINEALIDAD"
    assert report.loc[("a", "independiente"), "estado"] == "ACEPTABLE"


def test_missing_pairs_are_not_reported_as_correlated():
    frame = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "y": [np.nan, np.nan, 3.0, 4.0]})

    report = collinearity_report(frame, threshold=0.9)

    assert report.empty or report["correlacion"].isna().all()


REAL = Path(__file__).parents[2] / "variables_externas" / "_raw"


@pytest.mark.skipif(not (REAL / "open_meteo_archive").exists(), reason="snapshots completos no descargados")
def test_full_snapshots_reconcile_with_project_csvs():
    import json

    from prodma.external.connectors.bcrp import parse_bcrp_periods
    from prodma.external.connectors.nasa_power import parse_power_hourly
    from prodma.external.connectors.open_meteo import parse_archive

    bcrp = parse_bcrp_periods(json.loads((REAL / "bcrp_pbi" / "2026-10-15" / "respuesta.raw").read_text("utf-8")))
    old_pbi = pd.read_csv(Path(__file__).parents[2] / "variables_externas/economia/pbi_mensual_bcrp_2023_2026.csv")
    assert np.nanmax(np.abs(bcrp["valor"].to_numpy()[: len(old_pbi)] - old_pbi["valor"].to_numpy())) < 0.05

    om = parse_archive(json.loads((REAL / "open_meteo_archive" / "2026-10-15" / "respuesta.raw").read_text("utf-8")))
    assert len(om) == 32400
    nasa = parse_power_hourly(json.loads((REAL / "nasa_power" / "2026-10-15" / "respuesta.raw").read_text("utf-8")))
    assert len(nasa) == 32400
    assert (nasa["calidad"] == "faltante").sum() == 0
