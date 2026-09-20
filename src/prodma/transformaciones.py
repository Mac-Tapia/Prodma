"""Transformación del target usada por los modelos ML (signo * log1p(|v|)).

Vive en el paquete y no en el notebook porque los modelos entrenados guardan una
referencia a estas funciones al serializarse con joblib: si estuvieran definidas en el
notebook (``__main__``), los ``.joblib`` solo se podrían cargar desde ese mismo kernel.
Los nombres con guion bajo se conservan porque así quedaron referenciados en los modelos
ya entrenados (ver ``registrar_en_main``).
"""
from __future__ import annotations

import numpy as np


def _transformar_y(v):
    # signo*log1p(|v|): estabiliza varianza (el error absoluto de P/Q eléctrico crece con
    # la magnitud) preservando el signo real (Q puede ser negativo).
    v = np.asarray(v, dtype=float)
    return np.sign(v) * np.log1p(np.abs(v))


def _destransformar_y(v):
    v = np.asarray(v, dtype=float)
    return np.sign(v) * np.expm1(np.abs(v))


transformar_y = _transformar_y
destransformar_y = _destransformar_y


def registrar_en_main() -> None:
    """Expone las funciones en ``__main__`` para poder cargar los ``.joblib`` viejos.

    Los modelos entrenados antes de mover estas funciones al paquete las referencian como
    ``__main__._transformar_y``. En un kernel de Jupyter basta con importarlas al espacio
    del notebook; en un proceso trabajador (joblib/loky) hay que registrarlas a mano.
    """
    import __main__

    if not hasattr(__main__, "_transformar_y"):
        __main__._transformar_y = _transformar_y
    if not hasattr(__main__, "_destransformar_y"):
        __main__._destransformar_y = _destransformar_y
