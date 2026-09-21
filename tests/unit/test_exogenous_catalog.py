from pathlib import Path

from prodma.config import load_config
from prodma.exogenas.catalog import build_exogenous_catalog


ROOT = Path(__file__).parents[2]


def test_every_exogenous_source_declares_availability_and_future_policy():
    catalog = build_exogenous_catalog(load_config(ROOT / "config/default.yaml"))
    assert {source.name for source in catalog.sources} >= {
        "calendario_pe", "temperatura_ica", "pbi_bcrp", "poblacion_ica"
    }
    assert all(source.availability in {"known_future", "publication_lag", "historical_only", "scenario"}
               for source in catalog.sources)
    assert all(source.future_policy in {"calendar", "climatology", "last_known", "official_projection", "scenario_only"}
               for source in catalog.sources)


def test_enabled_sources_exist_and_have_sha256():
    catalog = build_exogenous_catalog(load_config(ROOT / "config/default.yaml"))
    enabled = [source for source in catalog.sources if source.enabled]
    assert enabled
    assert all(source.path.is_file() for source in enabled)
    assert all(len(source.sha256) == 64 for source in enabled)


def test_disabled_scenario_sources_do_not_require_fake_files():
    catalog = build_exogenous_catalog(load_config(ROOT / "config/default.yaml"))
    disabled = {source.name for source in catalog.sources if not source.enabled}
    assert {"clientes_confirmados", "produccion_regional"} <= disabled
