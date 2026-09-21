import json

import pandas as pd

from prodma.artifacts import RunLayout
from prodma.exogenas.new_customers import ingest_new_customers, publish_new_customer_scenario


def _workbook(path):
    frame = pd.DataFrame({
        "nombre del cliente nuevo": ["CLIENTE A", "CLIENTE B"],
        "Demanda kw": [100.0, 50.0],
        "factor de potencia": [0.95, 0.90],
        "aliementador": ["AL1", "AL2"],
        "Medidor": ["M1", "DESCONOCIDO"],
        2026: [60.0, 10.0],
        2027: [40.0, 40.0],
    })
    frame.to_excel(path, sheet_name="demandanueva.", index=False)


def test_new_customer_annual_load_is_not_added_to_history(tmp_path):
    workbook = tmp_path / "nuevosclientes.xlsx"
    _workbook(workbook)
    result = ingest_new_customers(workbook, known_meters={"M1"})
    assert {"cliente", "demanda_kw", "factor_potencia", "alimentador", "medidor", "anio",
            "incremento_kw", "fecha_entrada", "estado_mapeo"} <= set(result.table)
    assert result.table.groupby("cliente").incremento_kw.sum().le(
        result.table.groupby("cliente").demanda_kw.first() + 1e-9
    ).all()
    assert result.table["fecha_entrada"].isna().all()
    assert set(result.table.loc[result.table.medidor == "DESCONOCIDO", "estado_mapeo"]) == {"SIN_MAPEO"}

    layout = RunLayout.create(tmp_path, "run")
    artifacts = publish_new_customer_scenario(result, layout)
    scenario = pd.read_parquet(artifacts["new_customer_scenario"])
    assert {"limite_inferior_kw", "central_kw", "limite_superior_kw", "calidad"} <= set(scenario)
    assert (scenario["calidad"] == "scenario_no_observed").all()
    assert json.loads((layout.stage("01_ingesta", "auditoria") / "nuevos_clientes.json").read_text())["rows"] == 4


def test_inconsistent_customer_capacity_is_quarantined_not_silently_corrected(tmp_path):
    workbook = tmp_path / "nuevosclientes.xlsx"
    frame = pd.DataFrame({
        "nombre del cliente nuevo": ["CLIENTE X"], "Demanda kw": [100.0],
        "factor de potencia": [0.95], "aliementador": ["AL1"], "Medidor": ["M1"],
        2026: [70.0], 2027: [40.0],
    })
    frame.to_excel(workbook, sheet_name="demandanueva.", index=False)
    result = ingest_new_customers(workbook, known_meters={"M1"})
    assert set(result.table["estado_capacidad"]) == {"REVISAR_CAPACIDAD"}
    layout = RunLayout.create(tmp_path, "run")
    scenario = pd.read_parquet(publish_new_customer_scenario(result, layout)["new_customer_scenario"])
    assert scenario["central_kw"].isna().all()
