import pytest

from prodma.excel_schema import find_first_columns


ALIASES = {
    "P": {"kw tot", "kw tot mean"},
    "Q": {"kvar tot", "kvar tot mean"},
    "Fecha": {"local time"},
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


def test_missing_required_headers_remain_an_error():
    with pytest.raises(ValueError, match="Faltan columnas requeridas: Q"):
        find_first_columns(
            ["Local Time", "kW tot mean"],
            ALIASES,
            required={"Fecha", "P", "Q"},
        )
