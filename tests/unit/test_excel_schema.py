import pytest

from prodma.excel_schema import find_first_columns


ALIASES = {
    "P": {"kw tot", "kw tot mean"},
    "Q": {"kvar tot", "kvar tot mean"},
    "Fecha": {"local time"},
}

ALIASES_CON_S_MEDIDA = {
    **ALIASES,
    "S_medida": {"kva tot", "kva tot mean", "kva"},
}


def test_duplicate_headers_keep_the_first_column_from_left_to_right():
    headers = [
        "Local Time",
        "Timestamp",
        "kW tot mean",
        "kVAR tot mean",
        "Local Time",
        "kW tot mean",
        "kVAR tot mean",
    ]

    columns = find_first_columns(headers, ALIASES, required={"Fecha", "P", "Q"})

    assert columns == {"P": 2, "Q": 3, "Fecha": 0}


def test_s_medida_kva_keeps_first_matching_alias_spelling_from_left_to_right():
    # No es un duplicado EXACTO (mismo texto repetido): son dos variantes distintas
    # de alias para el mismo campo S_medida ("kVA tot" y "kVA tot mean") -- confirma
    # que la politica de "primera columna" aplica a cualquier combinacion de alias
    # aceptados para el campo, no solo a texto identico repetido.
    headers = [
        "Local Time",
        "kW tot mean",
        "kVAR tot mean",
        "kVA tot",
        "PF lag mean",
        "kVA tot mean",
    ]

    columns = find_first_columns(headers, ALIASES_CON_S_MEDIDA, required={"Fecha", "P", "Q"})

    assert columns["S_medida"] == 3  # "kVA tot", no "kVA tot mean" (posicion 5)


def test_s_medida_kva_no_se_confunde_con_kvar():
    # "kva" es substring de "kvar" en texto plano; la normalizacion (espacios en vez
    # de simbolos) y la comparacion por token completo deben evitar que una columna
    # kVAR se cuele como si fuera KVA.
    headers = ["Local Time", "kW tot mean", "kVAR tot mean"]

    columns = find_first_columns(headers, ALIASES_CON_S_MEDIDA, required={"Fecha", "P", "Q"})

    assert "S_medida" not in columns


def test_missing_required_headers_remain_an_error():
    with pytest.raises(ValueError, match="Faltan columnas requeridas: Q"):
        find_first_columns(
            ["Local Time", "kW tot mean"],
            ALIASES,
            required={"Fecha", "P", "Q"},
        )
