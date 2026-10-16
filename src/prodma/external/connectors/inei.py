"""INEI: población oficial vs. supuesto de escenario, identidad de tablas y producción regional bloqueada."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BLOCKED_MISSING_SOURCE = "BLOCKED_MISSING_SOURCE"
ESTADO_OBSERVADO = "OBSERVADO_OFICIAL"
ESTADO_ESCENARIO = "SUPUESTO_ESCENARIO"
USO_OBSERVADO = "covariable_observada"
USO_ESCENARIO = "escenario"
ETIQUETAS = {
    "observado_inei": (ESTADO_OBSERVADO, USO_OBSERVADO),
    "interpolado_lineal": (ESTADO_ESCENARIO, USO_ESCENARIO),
    "extrapolado_lineal": (ESTADO_ESCENARIO, USO_ESCENARIO),
}
REQUIRED_IDENTITY = ("ambito", "tabla_id", "unidad", "anio_base", "fecha_publicacion", "frecuencia")
AMBITO_ICA = "Ica"


def classify_population(frame: pd.DataFrame) -> pd.DataFrame:
    """Cada fila recibe estado y uso; un valor interpolado o extrapolado nunca es observación."""
    labels = frame["dato"].astype(str).str.strip()
    unknown = sorted(set(labels) - set(ETIQUETAS))
    if unknown:
        raise ValueError(f"dato desconocido en población: {unknown}")
    result = frame.copy()
    result["estado"] = labels.map(lambda label: ETIQUETAS[label][0])
    result["uso"] = labels.map(lambda label: ETIQUETAS[label][1])
    official = pd.to_numeric(result.loc[result["uso"] == USO_OBSERVADO, "poblacion"], errors="coerce")
    bounds_valid = official.dropna().nunique() >= 2
    result["cota_valida"] = bounds_valid
    return result


def validate_official_table_identity(meta: dict) -> None:
    for field in REQUIRED_IDENTITY:
        if meta.get(field) in (None, ""):
            raise ValueError(f"identidad de tabla oficial sin {field}")
    if str(meta["ambito"]).strip() != AMBITO_ICA:
        raise ValueError(f"ámbito fuera de Ica: {meta['ambito']!r}")


def production_status(path: Path | str, meta: dict | None) -> dict:
    """Producción regional: sin archivo verificado queda bloqueada con motivo explícito."""
    if not Path(path).is_file() or meta is None:
        return {"estado": BLOCKED_MISSING_SOURCE,
                "motivo": "falta tabla oficial verificada de produccion regional de Ica"}
    validate_official_table_identity(meta)
    return {"estado": "CANDIDATO_VERIFICADO", "motivo": "identidad oficial validada"}
