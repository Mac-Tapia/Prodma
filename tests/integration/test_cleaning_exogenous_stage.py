from pathlib import Path

import pandas as pd

from prodma.artifacts import RunLayout
from prodma.config import load_config
from prodma.exogenas.catalog import build_exogenous_catalog
from prodma.exogenas.matrix import build_exogenous_matrices, publish_exogenous_matrices


ROOT = Path(__file__).parents[2]


def test_exogenous_join_preserves_demand_window_and_provenance(tmp_path):
    config = load_config(ROOT / "config/default.yaml")
    index = pd.date_range("2023-01-01 00:15", periods=96 * 10, freq="15min", name="Fecha")
    matrices = build_exogenous_matrices(config, build_exogenous_catalog(config), index,
                                        future_end=pd.Timestamp("2023-01-15 00:00"))
    assert matrices.historical.index.equals(index)
    assert not matrices.historical.index.duplicated().any()
    assert {"feriado", "fin_semana", "temp_climatologica_c", "pbi_last_known", "poblacion_ica"} <= set(matrices.historical)
    assert matrices.source_hashes

    layout = RunLayout.create(tmp_path, "run")
    artifacts = publish_exogenous_matrices(matrices, layout, demand=pd.DataFrame({"M1_P": range(len(index))}, index=index))
    assert Path(artifacts["exogenous_historical"]).is_file()
    assert Path(artifacts["exogenous_future"]).is_file()
    assert Path(artifacts["exogenous_quality"]).is_file()
    assert len(artifacts["exogenous_sha256"]) == 64
    assert (layout.stage("02_limpieza", "estadistica_descriptiva") / "exogenas_resumen.csv").is_file()
    assert list((layout.stage("02_limpieza", "estadistica_descriptiva") / "figuras").glob("exogenas_*.png"))
