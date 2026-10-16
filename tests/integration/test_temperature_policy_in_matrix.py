from pathlib import Path

import pandas as pd
import pytest

from prodma.config import load_config
from prodma.exogenas.catalog import build_exogenous_catalog
from prodma.exogenas.matrix import build_exogenous_matrices, publish_exogenous_matrices
from prodma.artifacts import RunLayout


ROOT = Path(__file__).parents[2]


def _matrices(config=None, index=None):
    config = config or load_config(ROOT / "config/default.yaml")
    index = index if index is not None else pd.date_range("2024-01-01 00:00", periods=96, freq="15min", name="Fecha")
    return build_exogenous_matrices(config, build_exogenous_catalog(config), index,
                                    future_end=pd.Timestamp("2024-01-02 00:00"))


def test_temperature_comes_from_the_hashed_open_meteo_snapshot_and_is_traced():
    matrices = _matrices()

    assert "temperatura_fuente" in matrices.quality.columns
    historical = matrices.quality[matrices.quality["segmento"] == "historico"]
    assert set(historical["temperatura_fuente"]) == {"open_meteo_archive"}


def test_temperature_without_any_usable_source_fails_closed(tmp_path):
    config = load_config(ROOT / "config/default.yaml")
    raw = dict(config.raw)
    raw["exogenous"] = {"sources": {k: dict(v) for k, v in config.raw["exogenous"]["sources"].items()}}
    raw["exogenous"]["sources"]["temperatura_ica"]["raw_snapshot"] = "variables_externas/_raw/no_existe/respuesta.raw"
    broken = type(config)(project_root=config.project_root, output_root=config.output_root,
                          frequency=config.frequency, timezone=config.timezone, seed=config.seed,
                          horizons=config.horizons, raw=raw)
    index = pd.date_range("2024-01-01 00:00", periods=8, freq="15min", name="Fecha")

    with pytest.raises(ValueError, match="temperatura"):
        build_exogenous_matrices(broken, build_exogenous_catalog(broken), index,
                                 future_end=pd.Timestamp("2024-01-02 00:00"))
