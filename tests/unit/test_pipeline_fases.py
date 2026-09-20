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


# --- reanudación sin reentrenar (--reanudar / --set) -------------------------------------------
import copy  # noqa: E402

from prodma.pipeline import (  # noqa: E402
    construir_celda_overrides,
    filtrar_celdas_reanudacion,
    parsear_overrides,
)


def _con_etiqueta(nb, etiqueta):
    return [i for i, c in enumerate(nb.cells) if c.metadata.get('prodma', {}).get(etiqueta)]


def test_las_celdas_de_entrenamiento_se_omiten_al_reanudar_y_la_restauracion_solo_al_reanudar():
    a_omitir = _con_etiqueta(NOTEBOOK, 'omitir_al_reanudar')
    solo_reanudar = _con_etiqueta(NOTEBOOK, 'solo_al_reanudar')
    assert len(a_omitir) >= 4 and len(solo_reanudar) == 2  # bucle de entrenamiento, secciones 15/16 y su selección
    n = len(NOTEBOOK.cells)

    reanudando = filtrar_celdas_reanudacion(copy.deepcopy(NOTEBOOK), True)
    assert len(reanudando.cells) == n - len(a_omitir)
    assert not any('seleccionar_features_ga(Xt_todo' in c.source for c in reanudando.cells if c.cell_type == 'code')
    assert any('restaurar_estado(' in c.source for c in reanudando.cells)

    normal = filtrar_celdas_reanudacion(copy.deepcopy(NOTEBOOK), False)
    assert len(normal.cells) == n - len(solo_reanudar)
    assert any('seleccionar_features_ga(Xt_todo' in c.source for c in normal.cells if c.cell_type == 'code')
    assert not any('restaurar_estado(' in c.source for c in normal.cells)


def test_las_26_fases_siguen_completas_al_filtrar_en_ambos_modos():
    for reanudando in (True, False):
        nb = filtrar_celdas_reanudacion(copy.deepcopy(NOTEBOOK), reanudando)
        assert [n for n, _, _ in listar_fases(nb)] == list(range(1, 27))


def test_el_entrenamiento_se_niega_a_correr_si_hay_reanudacion_activa():
    celda = next(c for c in NOTEBOOK.cells if c.cell_type == 'code' and 'seleccionar_features_ga(Xt_todo' in c.source)
    assert "assert not CFG.get('reanudar_run')" in celda.source


def test_overrides_de_reanudacion_y_set():
    assert parsear_overrides(['workers_pronostico=8', 'backtest_horizontes=(96,)', 'nombre=abc']) == {
        'workers_pronostico': 8, 'backtest_horizontes': (96,), 'nombre': 'abc'}
    codigo = construir_celda_overrides('completo', None, 'C:/x/corrida', {'workers_pronostico': 8})
    assert "CFG['modo']='completo'" in codigo
    assert "CFG['reanudar_run']='C:/x/corrida'" in codigo
    assert "CFG['workers_pronostico']=8" in codigo
    compile(codigo, '<overrides>', 'exec')
    with pytest.raises(SystemExit):
        parsear_overrides(['sin_igual'])
