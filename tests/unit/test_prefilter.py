import numpy as np
import pandas as pd

from prodma.modeling.prefilter import PrefilterPolicy, run_prefilter
from prodma.modeling.registry import ModelSpec


class Good:
    def fit(self, x, y): return self
    def predict(self, x): return np.ones(len(x))


class Bad(Good):
    def predict(self, x): return np.full(len(x), np.nan)


def test_non_finite_candidate_is_rejected_without_stopping_others():
    specs = {"bad": ModelSpec("bad", lambda seed: Bad(), "15min", False, "cpu"),
             "good": ModelSpec("good", lambda seed: Good(), "15min", False, "cpu")}
    x = pd.DataFrame({"x": range(20)})
    y = pd.Series(np.ones(20))
    table = run_prefilter(specs, x, y, train_positions=np.arange(15), validation_positions=np.arange(15, 20),
                          policy=PrefilterPolicy(max_seconds=5), seed=42)
    states = table.set_index("modelo")["estado"]
    assert states["bad"] == "RECHAZADO"
    assert states["good"] == "APROBADO"
