import hashlib
import json

import pandas as pd
import pytest

from prodma.exogenas.futuras import fetch_forecast_payload, save_forecast_snapshot


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_requests_lima_timezone_hourly_temperature_and_a_timeout():
    seen = {}

    def opener(url, timeout):
        seen["url"] = url
        seen["timeout"] = timeout
        return FakeResponse(json.dumps({"timezone": "America/Lima"}).encode())

    payload, raw = fetch_forecast_payload(-14.07, -75.73, opener=opener, timeout=20)
    assert "latitude=-14.07" in seen["url"]
    assert "timezone=America%2FLima" in seen["url"]
    assert "hourly=temperature_2m" in seen["url"]
    assert seen["timeout"] == 20
    assert payload["timezone"] == "America/Lima"
    assert raw == json.dumps({"timezone": "America/Lima"}).encode()


def test_api_error_payload_is_rejected():
    def opener(url, timeout):
        return FakeResponse(json.dumps({"error": True, "reason": "limite excedido"}).encode(), status=429)

    with pytest.raises(ValueError, match="limite excedido"):
        fetch_forecast_payload(-14.07, -75.73, opener=opener, timeout=20)


def test_snapshot_keeps_raw_bytes_with_sha256_and_attribution(tmp_path):
    raw = b'{"timezone": "America/Lima", "hourly": {}}'
    issue = pd.Timestamp("2026-10-16T08:00:00")
    path = save_forecast_snapshot(raw, tmp_path, issue)
    assert path.read_bytes() == raw
    meta = json.loads(path.with_name(path.stem + ".meta.json").read_text(encoding="utf-8"))
    assert meta["sha256"] == hashlib.sha256(raw).hexdigest()
    assert meta["licencia"] == "CC-BY 4.0"
    assert meta["uso"] == "no_comercial"
    assert "open-meteo" in meta["fuente"].lower()
