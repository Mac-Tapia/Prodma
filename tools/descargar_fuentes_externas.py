"""Descarga reproducible de fuentes externas como snapshots crudos verificados.

Uso (desde la raíz del proyecto):
    .venv\\Scripts\\python.exe tools\\descargar_fuentes_externas.py --fecha 2026-10-15

Cada respuesta se guarda en variables_externas/_raw/<fuente>/<fecha>/respuesta.raw con su
metadato (URL, licencia, hash, tamaño). Sin red, use los snapshots ya guardados.
"""
from __future__ import annotations

import argparse
import urllib.request
from datetime import date
from pathlib import Path

from prodma.external.connectors.bcrp import bcrp_series_url
from prodma.external.connectors.open_meteo import archive_url
from prodma.external.contracts import ExternalSourceSpec
from prodma.external.snapshots import store_snapshot

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "variables_externas" / "_raw"
ICA_LAT, ICA_LON = -14.0678, -75.7286
TIMEOUT_S = 120


def default_fetcher(url: str, timeout: float):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read()
        headers = {"content-length": str(len(body))}
    return headers, iter([body])


SPECS = {
    "bcrp_pbi": ExternalSourceSpec(
        name="bcrp_pbi", schema_version="1", licence="BCRPData: términos de uso del BCRP",
        source_url="https://estadisticas.bcrp.gob.pe/estadisticas/series/"),
    "open_meteo_archive": ExternalSourceSpec(
        name="open_meteo_archive", schema_version="1",
        licence="Open-Meteo: CC BY 4.0 (datos ERA5 de Copernicus)",
        source_url="https://open-meteo.com/en/docs/historical-weather-api"),
    "nasa_power": ExternalSourceSpec(
        name="nasa_power", schema_version="1", licence="NASA POWER: datos públicos NASA",
        source_url="https://power.larc.nasa.gov/docs/services/api/temporal/hourly/"),
}


def urls(start: str, end: str) -> dict[str, str]:
    compact_start, compact_end = start.replace("-", ""), end.replace("-", "")
    return {
        "bcrp_pbi": bcrp_series_url("PN01773AM", start[:7], end[:7]),
        "open_meteo_archive": archive_url(ICA_LAT, ICA_LON, start, end),
        "nasa_power": (
            "https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M&community=RE"
            f"&longitude={ICA_LON}&latitude={ICA_LAT}&start={compact_start}&end={compact_end}&format=JSON"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fecha", default=date.today().isoformat(), help="fecha_descarga (AAAA-MM-DD)")
    parser.add_argument("--desde", default="2023-01-01")
    parser.add_argument("--hasta", default="2026-09-11")
    args = parser.parse_args()
    for name, url in urls(args.desde, args.hasta).items():
        snap = store_snapshot(SPECS[name], default_fetcher, url, RAW, fecha_descarga=args.fecha,
                              timeout=TIMEOUT_S, attempts_max=3, backoff_seconds=2.0)
        print(f"{name}: {snap.size} bytes sha256={snap.sha256} -> {snap.path}")


if __name__ == "__main__":
    main()
