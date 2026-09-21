"""Puerta de calidad para la trazabilidad cientifica de cada modulo."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml


REQUIRED_SOURCE_TYPES = {"articulo_indexado", "tesis_maestria", "tesis_doctoral"}


def validate_evidence_matrix(path: Path | str) -> list[str]:
    """Devuelve incumplimientos; una lista vacia permite declarar validado el modulo."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    errors: list[str] = []
    for module, record in payload.get("modulos", {}).items():
        if record.get("estado") != "validado":
            continue
        decisions = record.get("decisiones", [])
        if not decisions:
            errors.append(f"{module}: no tiene decisiones trazables")
            continue
        for decision in decisions:
            prefix = f"{module}/{decision.get('id', 'sin_id')}"
            for field in ("decision", "codigo", "prueba", "artefacto"):
                if not decision.get(field):
                    errors.append(f"{prefix}: falta {field}")
            sources = decision.get("fuentes", [])
            missing = REQUIRED_SOURCE_TYPES - {source.get("tipo") for source in sources}
            if missing:
                errors.append(f"{prefix}: faltan fuentes {sorted(missing)}")
            languages = {source.get("idioma") for source in sources if source.get("idioma")}
            if len(languages) < 2:
                errors.append(f"{prefix}: requiere fuentes en al menos dos idiomas")
            for source in sources:
                parsed = urlparse(str(source.get("url", "")))
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    errors.append(f"{prefix}: URL cientifica invalida")
    return errors
