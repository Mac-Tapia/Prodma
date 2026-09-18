from pathlib import Path

import nbformat
import pytest

from prodma.pipeline import NOTEBOOK_DEFAULT, indice_hasta_fase, listar_fases

NOTEBOOK = nbformat.read(NOTEBOOK_DEFAULT, as_version=4)


def test_notebook_default_path_exists():
    assert Path(NOTEBOOK_DEFAULT).is_file()


def test_listar_fases_encuentra_las_26_secciones_en_orden():
    fases = listar_fases(NOTEBOOK)
    numeros = [n for n, _, _ in fases]
    assert numeros == list(range(1, 27))


def test_listar_fases_indices_de_celda_son_crecientes():
    fases = listar_fases(NOTEBOOK)
    indices = [idx for _, _, idx in fases]
    assert indices == sorted(indices)


def test_indice_hasta_fase_sin_limite_devuelve_la_ultima_celda():
    fases = listar_fases(NOTEBOOK)
    assert indice_hasta_fase(NOTEBOOK, None, fases) == len(NOTEBOOK.cells) - 1


def test_indice_hasta_fase_es_creciente_con_el_numero_de_fase():
    fases = listar_fases(NOTEBOOK)
    idx_4 = indice_hasta_fase(NOTEBOOK, 4, fases)
    idx_14 = indice_hasta_fase(NOTEBOOK, 14, fases)
    idx_26 = indice_hasta_fase(NOTEBOOK, 26, fases)
    assert idx_4 < idx_14 < idx_26 == len(NOTEBOOK.cells) - 1


def test_indice_hasta_fase_incluye_la_celda_de_configuracion_desde_fase_2():
    fases = listar_fases(NOTEBOOK)
    idx_config = next(
        i for i, c in enumerate(NOTEBOOK.cells)
        if c.cell_type == 'code' and 'CFG = dict(' in c.source
    )
    assert indice_hasta_fase(NOTEBOOK, 2, fases) >= idx_config


def test_hasta_fase_inexistente_lanza_error_claro():
    fases = listar_fases(NOTEBOOK)
    with pytest.raises(SystemExit):
        indice_hasta_fase(NOTEBOOK, 999, fases)
