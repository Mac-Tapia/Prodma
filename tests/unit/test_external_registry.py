import pandas as pd
import pytest

from prodma.exogenas.contracts import ExogenousCatalog, ExogenousSource
from prodma.exogenas.validation import ValidationItem, validate_exogenous_sources
from prodma.external.registry import (
    DuplicateSourceError,
    SourceRegistry,
    UnsupportedSourceError,
)


def _validator_ok(source, demand_index, timezone):
    return ValidationItem(source.name, "VALID", 1, "2023-01-01", "2023-12-31")


def test_registry_rejects_duplicate_registration():
    registry = SourceRegistry()
    registry.register("fuente_a", _validator_ok)

    with pytest.raises(DuplicateSourceError, match="fuente_a"):
        registry.register("fuente_a", _validator_ok)


def test_unknown_source_is_an_explicit_error_not_a_skip():
    registry = SourceRegistry()

    with pytest.raises(UnsupportedSourceError, match="fuente_nueva"):
        registry.resolve("fuente_nueva")


def test_new_source_is_validated_by_registered_plugin_only(tmp_path):
    csv = tmp_path / "nueva.csv"
    csv.write_text("fecha_hora,valor\n2023-01-01 00:00,1\n", encoding="utf-8")
    source = ExogenousSource(
        name="fuente_nueva", path=csv, frequency="1h", value_columns=("valor",),
        availability="known_future", future_policy="known", source_url="", sha256="x", enabled=True,
    )
    catalog = ExogenousCatalog((source,))
    index = pd.DatetimeIndex(["2023-01-01 00:00"])

    with pytest.raises(ValueError, match="sin validador"):
        validate_exogenous_sources(catalog, index, registry=SourceRegistry())

    registry = SourceRegistry()
    registry.register("fuente_nueva", _validator_ok)
    report = validate_exogenous_sources(catalog, index, registry=registry)

    assert [item.status for item in report.items] == ["VALID"]


def test_disabled_sources_keep_terminal_status_and_do_not_need_validators():
    source = ExogenousSource(
        name="produccion_regional", path=None, frequency="1M", value_columns=("indice",),
        availability="scenario", future_policy="scenario_only", source_url="", sha256="",
        enabled=False,
    )
    report = validate_exogenous_sources(ExogenousCatalog((source,)), pd.DatetimeIndex(["2023-01-01"]),
                                        registry=SourceRegistry())

    assert report.items == ()
    assert report.errors == ()
