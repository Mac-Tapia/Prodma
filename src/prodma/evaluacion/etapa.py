"""Evaluacion por serie: perdidas alineadas por bloque, pruebas pareadas y ranking con todos los candidatos."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prodma.evaluacion.inferencia import diebold_mariano, friedman_holm
from prodma.evaluacion.mcs import model_confidence_set
from prodma.evaluacion.ranking import rank_candidates

BLOCK = ["horizonte", "fold", "timestamp"]


def _estado_entrenamiento(estado: str) -> str:
    return "TRAINED" if estado == "ENTRENADO" else "FAILED"


def _complexity(parameters: str) -> int:
    try:
        return len(json.loads(parameters))
    except (TypeError, ValueError):
        return 0


def evaluate_series(predictions: pd.DataFrame, trained: pd.DataFrame, times: pd.DataFrame,
                    prefilter: pd.DataFrame, coverage_tolerance: float) -> dict:
    predictions = predictions.assign(perdida=(predictions["y"] - predictions["prediccion"]).abs())
    wide = predictions.pivot_table(index=BLOCK, columns="modelo", values="perdida", aggfunc="first")
    aligned = wide.dropna(how="any")
    not_aligned = int(len(wide) - len(aligned))

    fold_mae = predictions.groupby(["modelo", "horizonte", "fold"])["perdida"].mean().groupby(level=0)
    rows = []
    for row in trained.itertuples(index=False):
        series = predictions.loc[predictions["modelo"] == row.modelo, "perdida"]
        rows.append({
            "modelo": row.modelo,
            "estado": _estado_entrenamiento(row.estado),
            "mae": float(series.mean()) if len(series) else np.nan,
            "estabilidad": float(fold_mae.std().get(row.modelo, np.nan)),
            "cobertura_error": np.nan,
            "complejidad": _complexity(row.parameters),
            "segundos": float(times.loc[times["modelo"] == row.modelo, "segundos"].sum()),
        })
    known = set(trained["modelo"])
    for row in prefilter.itertuples(index=False):
        if row.modelo in known:
            continue
        estado = "PREFILTER_REJECTED" if row.estado != "APROBADO" else "FAILED"
        rows.append({"modelo": row.modelo, "estado": estado, "mae": np.nan,
                     "estabilidad": np.nan, "cobertura_error": np.nan, "complejidad": 0, "segundos": 0.0})
    candidates = pd.DataFrame(rows)
    ranking = rank_candidates(candidates, coverage_tolerance=coverage_tolerance, require_calibration=False)

    inference: dict = {"bloques": int(len(aligned)), "bloques_no_alineados": not_aligned,
                       "mejor_modelo": None, "mcs": [], "pairwise": pd.DataFrame(columns=["modelo", "p_valor", "p_holm", "p_dm"])}
    if aligned.shape[1] >= 2 and aligned.shape[0] >= 30:
        mcs = model_confidence_set(aligned, alpha=0.1, block_length=max(1, min(24, len(aligned) // 10)))
        inference["mcs"] = mcs["mcs"]
        inference["mcs_p_valores"] = mcs["p_valores"]
    if aligned.shape[1] >= 3 and aligned.shape[0] >= 3:
        result = friedman_holm(aligned)
        inference["mejor_modelo"] = result["mejor_modelo"]
        inference["p_friedman"] = result["p_friedman"]
        pairwise = result["pairwise"]
        horizon = int(predictions["horizonte"].max())
        best = result["mejor_modelo"]
        inference["pairwise"] = pairwise.assign(p_dm=[
            diebold_mariano(aligned[name], aligned[best], horizon)["p_valor"] for name in pairwise["modelo"]
        ])
    elif aligned.shape[1] == 2 and aligned.shape[0] >= 2:
        best = aligned.mean().idxmin()
        inference["mejor_modelo"] = best
        inference["motivo"] = "friedman_requiere_tres_modelos"
    else:
        inference["motivo"] = "bloques_alineados_insuficientes"
    return {"ranking": ranking, "inferencia": inference}
