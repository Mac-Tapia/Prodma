"""CLI para ejecutar notebook/proyecciondemanda.ipynb por fases, desde terminal.

No reimplementa la lógica del pipeline (eso viviría en módulos separados que todavía
no existen para la mayor parte del proyecto, ver docs/superpowers/plans/): envuelve el
notebook real y lo ejecuta con nbclient, celda por celda, hasta la fase pedida, e
inyecta CFG['modo']/CFG['backend'] según los flags. El notebook fuente nunca se
modifica -- se ejecuta una copia en memoria y el resultado (con outputs) se guarda
aparte.

Uso:
    python -m prodma.pipeline --listar-fases
    python -m prodma.pipeline --modo rapido --backend auto
    python -m prodma.pipeline --modo completo --backend gpu --hasta-fase 14
    python -m prodma.pipeline --reanudar resultados/C-AUT_20260918_105110 --timeout 43200

`--reanudar` continúa en la carpeta de una corrida previa SIN reentrenar: omite las secciones
14-16 (ya persistidas), restaura el estado desde los modelos guardados y sigue desde la 17.

Ver docs/MANUAL_EJECUCION.md para el flujo completo.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
NOTEBOOK_DEFAULT = RAIZ_PROYECTO / 'notebook' / 'proyecciondemanda.ipynb'
SECCION_RE = re.compile(r'^#{2}\s*(\d+)\s*[·.\-]')
RUN_RE = re.compile(r'Resultados:\s*(\S.*)')


def listar_fases(nb) -> list[tuple[int, str, int]]:
    """[(numero_seccion, titulo, indice_celda_markdown), ...] en el orden del notebook.

    Las fases son las secciones numeradas ("## NN · Título") que el propio notebook ya
    usa para organizarse -- no son un esquema nuevo, solo se leen del notebook para no
    duplicar (y desincronizar) una lista de fases a mano en este archivo.
    """
    fases = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'markdown':
            continue
        primera = cell.source.strip().split('\n')[0] if cell.source.strip() else ''
        m = SECCION_RE.match(primera)
        if m:
            fases.append((int(m.group(1)), primera.lstrip('#').strip(), i))
    return fases


def filtrar_celdas_reanudacion(nb, reanudando: bool):
    """Quita del notebook las celdas que no corresponden al modo de ejecución.

    Las celdas etiquetadas ``prodma.omitir_al_reanudar`` (entrenamiento y diagnósticos ya
    persistidos) solo corren en una ejecución normal; las etiquetadas
    ``prodma.solo_al_reanudar`` (restauración del estado) solo al reanudar. Se hace sobre la
    copia en memoria: el notebook fuente no cambia.
    """
    etiqueta_omitir = 'omitir_al_reanudar' if reanudando else 'solo_al_reanudar'
    nb.cells = [c for c in nb.cells if not c.metadata.get('prodma', {}).get(etiqueta_omitir)]
    return nb


def indice_hasta_fase(nb, hasta_fase: int | None, fases: list[tuple[int, str, int]]) -> int:
    """Índice de la última celda a ejecutar (inclusive) para llegar hasta `hasta_fase`."""
    if not fases:
        raise RuntimeError('No se encontraron secciones numeradas ("## NN ·") en el notebook')
    if hasta_fase is None:
        return len(nb.cells) - 1
    numeros = sorted(n for n, _, _ in fases)
    if hasta_fase not in numeros:
        raise SystemExit(f'--hasta-fase {hasta_fase} no existe. Fases disponibles: {numeros}')
    ordenadas = sorted(fases, key=lambda x: x[2])
    for pos, (numero, _titulo, idx_md) in enumerate(ordenadas):
        if numero == hasta_fase:
            siguiente = ordenadas[pos + 1][2] if pos + 1 < len(ordenadas) else len(nb.cells)
            return siguiente - 1
    raise AssertionError('inalcanzable')  # pragma: no cover


def asegurar_kernel(nombre: str = 'prodma_venv') -> str:
    """Registra (o reutiliza) un kernel de Jupyter que apunte al Python ACTUAL.

    Evita el problema real detectado en este proyecto: un kernel 'python3' registrado
    globalmente puede apuntar a un Python distinto al de este `.venv`, y entonces el
    notebook falla por "dependencias faltantes" que en realidad sí están instaladas,
    solo que en otro intérprete.
    """
    from ipykernel.kernelspec import install
    from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel

    ksm = KernelSpecManager()
    try:
        spec = ksm.get_kernel_spec(nombre)
        if Path(spec.argv[0]).resolve() == Path(sys.executable).resolve():
            return nombre
    except NoSuchKernel:
        pass
    install(user=True, kernel_name=nombre, display_name='Python (Prodma .venv, auto)')
    return nombre


def parsear_overrides(pares: list[str] | None) -> dict:
    """['clave=valor', ...] -> {clave: valor}; el valor se interpreta como literal de Python."""
    out = {}
    for par in pares or []:
        if '=' not in par:
            raise SystemExit(f"--set espera CLAVE=VALOR, recibió {par!r}")
        clave, valor = par.split('=', 1)
        try:
            out[clave.strip()] = ast.literal_eval(valor)
        except (ValueError, SyntaxError):
            out[clave.strip()] = valor
    return out


def construir_celda_overrides(modo: str | None, backend: str | None, reanudar: str | None = None,
                              extra: dict | None = None) -> str:
    lineas = ["# --- Overrides inyectados por 'python -m prodma.pipeline' (no persistidos en el notebook fuente) ---"]
    if modo is not None:
        lineas.append(f"CFG['modo']={modo!r}")
    if backend is not None:
        lineas.append(f"CFG['backend']={backend!r}")
    if reanudar is not None:
        lineas.append(f"CFG['reanudar_run']={reanudar!r}")
    for clave, valor in (extra or {}).items():
        lineas.append(f"CFG[{clave!r}]={valor!r}")
    claves = ['modo', 'backend'] + (['reanudar_run'] if reanudar is not None else []) + list(extra or {})
    lineas.append(f"print('CFG overrides aplicados por CLI:', {{k: CFG[k] for k in {claves!r} if k in CFG}})")
    return '\n'.join(lineas)


def _primera_linea(texto: str, n: int = 100) -> str:
    t = texto.strip()
    return t.split('\n')[0][:n] if t else '(celda vacía)'


def _on_cell_start(cell, cell_index, **_kwargs) -> None:
    if cell.cell_type == 'code' and cell.source.strip():
        print(f'\n>>> [celda {cell_index}] {_primera_linea(cell.source)}', flush=True)


class NotebookClientEnVivo(NotebookClient):
    """Igual que NotebookClient, pero vuelca cada mensaje de salida a stdout EN EL
    INSTANTE en que llega del kernel, no cuando termina la celda completa.

    Una sola celda de este notebook puede tardar varios minutos (p. ej. la que arma la
    matriz histórica recorriendo cada medidor, o la que entrena los 6 modelos): con
    on_cell_executed (que dispara solo al terminar la celda entera) no se ve nada
    avanzar hasta que esa celda completa termina. Sobrescribir `output()` -- el método
    de nbclient que procesa cada mensaje individual del kernel -- es lo único que da
    progreso realmente en vivo.
    """

    def output(self, outs, msg, display_id, cell_index):
        out = super().output(outs, msg, display_id, cell_index)
        if out is not None:
            tipo = out.get('output_type')
            if tipo == 'stream':
                sys.stdout.write(out.get('text', ''))
                sys.stdout.flush()
            elif tipo in ('execute_result', 'display_data'):
                texto = out.get('data', {}).get('text/plain')
                if texto:
                    print(texto, flush=True)
            elif tipo == 'error':
                print('!!! ERROR:', out.get('ename'), '-', out.get('evalue'), flush=True)
        return out


def extraer_run_dir(nb) -> str | None:
    """Lee la carpeta resultados/ejecucion_<...>/ real desde el output ya capturado de
    la celda '03 · Rutas locales', en vez de volver a consultar el kernel (que puede ya
    haber muerto si una celda posterior falló)."""
    for cell in nb.cells:
        for out in cell.get('outputs', []):
            texto = out.get('text', '')
            m = RUN_RE.search(texto)
            if m:
                return m.group(1).strip()
    return None


def ejecutar(nb, args, fases: list[tuple[int, str, int]]) -> dict:
    idx_ultimo = indice_hasta_fase(nb, args.hasta_fase, fases)

    idx_config = next(
        (i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and 'CFG = dict(' in c.source),
        None,
    )
    overrides_aplicables = idx_config is not None and idx_config <= idx_ultimo
    if overrides_aplicables:
        celda = nbformat.v4.new_code_cell(construir_celda_overrides(
            args.modo, args.backend, str(args.reanudar) if args.reanudar else None, args.overrides))
        nb.cells.insert(idx_config + 1, celda)
        idx_ultimo += 1
    elif args.modo is not None or args.backend is not None or args.reanudar or args.overrides:
        print(f"AVISO: --hasta-fase {args.hasta_fase} termina antes de la celda de "
              "configuración; --modo/--backend no se aplican en esta corrida.")

    nb.cells = nb.cells[: idx_ultimo + 1]

    nombre_kernel = asegurar_kernel()
    inicio = time.time()
    cliente = NotebookClientEnVivo(
        nb, timeout=args.timeout, kernel_name=nombre_kernel,
        resources={'metadata': {'path': str(args.notebook.parent)}},
        on_cell_start=_on_cell_start,
    )
    estado, error_detalle = 'SUCCESS', None
    try:
        cliente.execute()
    except CellExecutionError as exc:
        estado, error_detalle = 'FAILED', str(exc)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.output)

    manifest = dict(
        notebook_fuente=str(args.notebook), notebook_ejecutado=str(args.output),
        hasta_fase=args.hasta_fase, modo_solicitado=args.modo, backend_solicitado=args.backend,
        overrides_aplicados=overrides_aplicables, kernel=nombre_kernel,
        inicio=time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(inicio)),
        reanudar=str(args.reanudar) if args.reanudar else None, overrides_extra=args.overrides,
        fin=time.strftime('%Y-%m-%dT%H:%M:%S'), segundos=round(time.time() - inicio, 1),
        celdas_ejecutadas=len(nb.cells), estado=estado, error=error_detalle,
        resultados_run_dir=extraer_run_dir(nb),
    )
    manifest_path = args.output.with_name(args.output.stem + '.manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    manifest['manifest_path'] = str(manifest_path)

    if manifest['resultados_run_dir']:
        # Puntero a la corrida mas reciente: con muchas carpetas resultados/<fecha>/
        # acumuladas (nunca se sobreescriben, para poder comparar entre ejecuciones),
        # es facil abrir por error una carpeta vieja y confundir una figura ya corregida
        # con una desactualizada. Este archivo siempre apunta a la ULTIMA corrida real.
        puntero = RAIZ_PROYECTO / 'resultados' / 'ULTIMA_EJECUCION.txt'
        puntero.write_text(
            f"{manifest['resultados_run_dir']}\n"
            f"estado: {estado}\n"
            f"fin: {manifest['fin']}\n",
            encoding='utf-8',
        )

    return manifest


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m prodma.pipeline',
        description='Ejecuta notebook/proyecciondemanda.ipynb por fases, desde terminal.',
    )
    parser.add_argument('--modo', choices=['rapido', 'completo'], default=None,
                         help="Sobrescribe CFG['modo'] (por defecto: el que ya tiene el notebook).")
    parser.add_argument('--backend', choices=['auto', 'cpu', 'gpu'], default=None,
                         help="Sobrescribe CFG['backend'] usado por el preflight GPU (sección 13).")
    parser.add_argument('--reanudar', type=Path, default=None, metavar='CARPETA',
                         help='Continúa en la carpeta de una corrida previa (resultados/<corrida>) SIN reentrenar: '
                              'omite las secciones 14-16, restaura el estado desde los modelos guardados y sigue '
                              'desde la 17. Si no se indica --modo, se toma el de esa corrida.')
    parser.add_argument('--set', dest='sobrescribir', action='append', metavar='CLAVE=VALOR',
                         help="Sobrescribe una clave de CFG (valor como literal de Python), p. ej. "
                              "--set workers_pronostico=8 --set max_medidores_adicionales=5. Repetible.")
    parser.add_argument('--hasta-fase', type=int, default=None, metavar='N',
                         help='Ejecuta solo hasta la sección numerada N (ver --listar-fases). '
                              'Por defecto corre el notebook completo.')
    parser.add_argument('--listar-fases', action='store_true',
                         help='Lista las fases/secciones del notebook y termina, sin ejecutar nada.')
    parser.add_argument('--notebook', type=Path, default=NOTEBOOK_DEFAULT,
                         help='Ruta al notebook fuente (nunca se modifica).')
    parser.add_argument('--output', type=Path, default=None,
                         help='Dónde escribir el notebook ejecutado (default: '
                              'resultados/cli_<timestamp>/notebook_ejecutado.ipynb).')
    parser.add_argument('--timeout', type=int, default=7200,
                         help='Timeout por celda, en segundos (default 7200; la primera '
                              'lectura de datahistorica/ sin caché puede tardar).')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    args.overrides = parsear_overrides(args.sobrescribir)
    if args.reanudar is not None:
        if not args.reanudar.is_absolute():
            args.reanudar = (RAIZ_PROYECTO / args.reanudar).resolve()
        config_origen = args.reanudar / 'config.json'
        if not (args.reanudar / 'modelos').is_dir() or not config_origen.exists():
            raise SystemExit(f'--reanudar: {args.reanudar} no parece una corrida previa (falta modelos/ o config.json)')
        if args.modo is None:  # la configuración de la reanudación debe coincidir con la que entrenó los modelos
            args.modo = json.loads(config_origen.read_text(encoding='utf-8'))['modo']
    nb = nbformat.read(args.notebook, as_version=4)
    filtrar_celdas_reanudacion(nb, reanudando=args.reanudar is not None)
    fases = listar_fases(nb)

    if args.listar_fases:
        for numero, titulo, _ in fases:
            print(f'{numero:>2}  {titulo}')
        return 0

    if args.output is None:
        marca = time.strftime('%Y%m%d_%H%M%S')
        args.output = RAIZ_PROYECTO / 'resultados' / f'cli_{marca}' / 'notebook_ejecutado.ipynb'

    manifest = ejecutar(nb, args, fases)
    print(f"Estado: {manifest['estado']}")
    print(f"Notebook ejecutado: {manifest['notebook_ejecutado']}")
    print(f"Manifiesto: {manifest['manifest_path']}")
    if manifest['resultados_run_dir']:
        print(f"Resultados del pipeline: {manifest['resultados_run_dir']}")
    if manifest['estado'] == 'FAILED':
        print('ERROR (detalle completo en el manifiesto):')
        print((manifest['error'] or '')[:2000])
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
