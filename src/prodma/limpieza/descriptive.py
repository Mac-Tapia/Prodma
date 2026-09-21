"""Estadistica descriptiva auditable de variables limpias."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def describe_variable(frame: pd.DataFrame) -> dict[str, float | int]:
    observed = frame["observado"].dropna()
    q25 = observed.quantile(0.25) if len(observed) else np.nan
    q75 = observed.quantile(0.75) if len(observed) else np.nan
    mean = observed.mean() if len(observed) else np.nan
    std = observed.std() if len(observed) > 1 else np.nan
    return {
        "observados": int(frame["observado"].notna().sum()),
        "imputados": int((frame["observado"].isna() & frame["contexto"].notna()).sum()),
        "faltantes_contexto": int(frame["contexto"].isna().sum()),
        "porcentaje_faltante_original": float(frame["observado"].isna().mean() * 100),
        "media": float(mean),
        "mediana": float(observed.median()) if len(observed) else np.nan,
        "desviacion": float(std),
        "coeficiente_variacion": float(std / abs(mean)) if len(observed) > 1 and mean != 0 else np.nan,
        "mad": float((observed - observed.median()).abs().median()) if len(observed) else np.nan,
        "q01": float(observed.quantile(0.01)) if len(observed) else np.nan,
        "q05": float(observed.quantile(0.05)) if len(observed) else np.nan,
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(observed.quantile(0.95)) if len(observed) else np.nan,
        "q99": float(observed.quantile(0.99)) if len(observed) else np.nan,
        "iqr": float(q75 - q25) if len(observed) else np.nan,
        "asimetria": float(observed.skew()) if len(observed) > 2 else np.nan,
        "curtosis_exceso": float(observed.kurt()) if len(observed) > 3 else np.nan,
        "minimo": float(observed.min()) if len(observed) else np.nan,
        "maximo": float(observed.max()) if len(observed) else np.nan,
        "ceros": int((observed == 0).sum()),
        "negativos": int((observed < 0).sum()),
        "atipicos": int(frame["es_atipico"].sum()),
        "recortados": int(frame["es_recortado"].sum()),
    }


def normality_diagnostics(series: pd.Series, alpha: float = 0.05, max_n: int = 5000) -> dict[str, float | int | str]:
    """Diagnostica normalidad con muestra determinista y criterios convergentes.

    La salida `parametrica_candidata` no certifica independencia ni homocedasticidad.
    """
    values = np.asarray(pd.Series(series).dropna(), dtype=float)
    values = values[np.isfinite(values)]
    n_total = int(values.size)
    if n_total > max_n:
        positions = np.linspace(0, n_total - 1, max_n, dtype=int)
        values = values[positions]
    n_used = int(values.size)
    base: dict[str, float | int | str] = {"n_total": n_total, "n_prueba": n_used, "alpha": alpha}
    if n_used < 20 or np.ptp(values) == 0:
        reason = "n_menor_20" if n_used < 20 else "serie_constante"
        return {
            **base,
            "dagostino_estadistico": np.nan, "dagostino_p": np.nan, "dagostino_resultado": "no_aplica",
            "jarque_bera_estadistico": np.nan, "jarque_bera_p": np.nan, "jarque_bera_resultado": "no_aplica",
            "anderson_estadistico": np.nan, "anderson_p": np.nan, "anderson_resultado": "no_aplica",
            "asimetria_prueba": np.nan, "curtosis_exceso_prueba": np.nan,
            "decision_inferencial": "muestra_insuficiente", "motivo": reason,
        }
    dagostino_stat, dagostino_p = stats.normaltest(values)
    jb = stats.jarque_bera(values)
    ad = stats.anderson(values, dist="norm", method="interpolate")
    ad_p = float(ad.pvalue)
    skewness = float(stats.skew(values, bias=False))
    excess_kurtosis = float(stats.kurtosis(values, fisher=True, bias=False))
    checks = (
        float(dagostino_p) >= alpha,
        float(jb.pvalue) >= alpha,
        ad_p >= alpha,
        abs(skewness) <= 0.5,
        abs(excess_kurtosis) <= 1.0,
    )
    decision = "parametrica_candidata" if all(checks) else "no_parametrica"
    return {
        **base,
        "dagostino_estadistico": float(dagostino_stat),
        "dagostino_p": float(dagostino_p),
        "dagostino_resultado": "no_rechaza_normalidad" if dagostino_p >= alpha else "rechaza_normalidad",
        "jarque_bera_estadistico": float(jb.statistic),
        "jarque_bera_p": float(jb.pvalue),
        "jarque_bera_resultado": "no_rechaza_normalidad" if jb.pvalue >= alpha else "rechaza_normalidad",
        "anderson_estadistico": float(ad.statistic),
        "anderson_p": ad_p,
        "anderson_resultado": "no_rechaza_normalidad" if ad_p >= alpha else "rechaza_normalidad",
        "asimetria_prueba": skewness,
        "curtosis_exceso_prueba": excess_kurtosis,
        "decision_inferencial": decision,
        "motivo": "criterios_convergentes" if decision == "parametrica_candidata" else "normalidad_rechazada_o_forma_incompatible",
    }


def plot_distribution_diagnostics(series: pd.Series, output, title: str) -> None:
    """Guarda histograma y grafico Q-Q de una serie transformada."""
    import matplotlib.pyplot as plt

    values = pd.Series(series).dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) > 5000:
        values = values[np.linspace(0, len(values) - 1, 5000, dtype=int)]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(values, bins="fd", color="tab:blue", alpha=0.8)
    axes[0].set_title("Histograma")
    axes[0].set_xlabel("Diferencia estacional")
    axes[0].set_ylabel("Frecuencia")
    if len(values) >= 2:
        stats.probplot(values, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q normal")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    plt.close(figure)
