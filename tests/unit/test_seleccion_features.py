import numpy as np
import pandas as pd
import pytest

from prodma.seleccion_features import seleccionar_features_ga


def _cv_simple(n, n_splits=3, tam_val=20):
    splits = []
    for k in range(n_splits, 0, -1):
        fin_va = n - (k - 1) * tam_val
        ini_va = fin_va - tam_val
        splits.append((np.arange(0, ini_va), np.arange(ini_va, fin_va)))
    return splits


def test_selecciona_las_columnas_informativas_y_descarta_el_ruido():
    rng = np.random.RandomState(0)
    n = 400
    util1 = rng.normal(size=n)
    util2 = rng.normal(size=n)
    ruido = pd.DataFrame({f'ruido_{i}': rng.normal(size=n) for i in range(6)})
    X = pd.concat([pd.DataFrame({'util1': util1, 'util2': util2}), ruido], axis=1)
    y = pd.Series(3 * util1 - 2 * util2 + rng.normal(scale=0.05, size=n))

    cv = _cv_simple(n)
    columnas, fitness = seleccionar_features_ga(X, y, cv, poblacion=16, generaciones=12, semilla=1)

    assert 'util1' in columnas
    assert 'util2' in columnas
    assert np.isfinite(fitness)


def test_nunca_devuelve_una_lista_vacia():
    rng = np.random.RandomState(2)
    X = pd.DataFrame({f'c{i}': rng.normal(size=100) for i in range(5)})
    y = pd.Series(rng.normal(size=100))
    cv = _cv_simple(100, n_splits=2, tam_val=15)
    columnas, _ = seleccionar_features_ga(X, y, cv, poblacion=8, generaciones=5, semilla=3)
    assert len(columnas) >= 1


def test_es_determinista_con_la_misma_semilla():
    rng = np.random.RandomState(4)
    X = pd.DataFrame({f'c{i}': rng.normal(size=150) for i in range(8)})
    y = pd.Series(rng.normal(size=150))
    cv = _cv_simple(150, n_splits=2, tam_val=20)
    c1, f1 = seleccionar_features_ga(X, y, cv, poblacion=10, generaciones=8, semilla=7)
    c2, f2 = seleccionar_features_ga(X, y, cv, poblacion=10, generaciones=8, semilla=7)
    assert c1 == c2
    assert f1 == f2


def test_rechaza_dataframe_sin_columnas():
    with pytest.raises(ValueError):
        seleccionar_features_ga(pd.DataFrame(index=range(10)), pd.Series(range(10)), [(np.arange(5), np.arange(5, 10))])
