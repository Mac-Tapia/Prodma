from prodma.config import load_config
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_default_config_declares_the_coverage_tolerance_explicitly():
    config = load_config(ROOT / "config/default.yaml")
    assert config.raw["evaluation"]["coverage_tolerance"] == 0.05