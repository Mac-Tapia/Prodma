"""Apply the local VS Code execution profile to the demand notebook."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebook" / "proyecciondemanda.ipynb"


def lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

nb["cells"][0]["source"] = lines("""# Demanda eléctrica — VS Code (ejecución local)

Versión 1.3. Lectura local por hojas, seis modelos y escenarios.

**Uso:** abra la carpeta raíz `Prodma` en VS Code, seleccione el kernel `.venv` y ejecute las celdas en orden. Los cuatro históricos se leen desde `datahistorica/`, las altas desde `nuevosclientes/` y cada ejecución escribe una carpeta independiente dentro de `resultados/`. Los archivos de entrada nunca se modifican.

Las fechas originales se conservan como fin de intervalo. No se afirma que `kW tot` sea una media: la energía derivada requiere confirmación. La proyección larga es un escenario de perfil y crecimiento, no una predicción ML validada a diez años. Las nuevas cargas pendientes no se suman silenciosamente.

## Actualización 1.3 — Migración de Colab a VS Code

Se eliminaron el montaje de Google Drive, la autenticación interactiva y la descarga mediante Drive API. La raíz del proyecto se descubre desde el directorio de trabajo (raíz o carpeta `notebook`) o mediante la variable opcional `PRODMA_ROOT`. El inventario valida exactamente cuatro archivos históricos locales y un archivo de nuevos clientes antes de iniciar el procesamiento.
""")

nb["cells"][2]["source"] = lines("""## 01 · Dependencias

Antes de abrir el notebook, cree/active `.venv` e instale una vez `requirements.txt`. La celda siguiente solo valida el entorno; no modifica paquetes durante una ejecución reproducible.
""")

nb["cells"][3]["source"] = lines("""import sys, subprocess, importlib.util

MODULOS_REQUERIDOS = {
    'numpy':'numpy', 'pandas':'pandas', 'scipy':'scipy', 'xlrd':'xlrd',
    'openpyxl':'openpyxl', 'pyarrow':'pyarrow', 'xlsxwriter':'xlsxwriter',
    'sklearn':'scikit-learn', 'statsmodels':'statsmodels', 'xgboost':'xgboost',
    'lightgbm':'lightgbm', 'catboost':'catboost', 'holidays':'holidays',
    'matplotlib':'matplotlib', 'seaborn':'seaborn', 'psutil':'psutil',
    'joblib':'joblib'
}
faltantes = [paquete for modulo,paquete in MODULOS_REQUERIDOS.items()
             if importlib.util.find_spec(modulo) is None]
if faltantes:
    raise RuntimeError(
        'Faltan dependencias en el kernel actual: '+', '.join(faltantes)+
        '. En la terminal de VS Code ejecute: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'
    )
print('Entorno local validado:', sys.executable)
""")

cell5 = "".join(nb["cells"][5]["source"])
nb["cells"][5]["source"] = lines(cell5.replace(
    "CFG = dict(folder_id='15iClJLiknvJOhJ9VcOxmP00K3VsH3Hmg', modo='rapido', seed=42,",
    "CFG = dict(modo='rapido', seed=42,",
))

nb["cells"][6]["source"] = lines("""## 03 · Rutas locales y resultados

Descubre la raíz del proyecto de forma portable, valida su estructura y crea una carpeta de resultados independiente. Puede definir `PRODMA_ROOT` si el kernel se inicia fuera del proyecto.
""")

nb["cells"][7]["source"] = lines("""def encontrar_raiz_proyecto(inicio=None):
    override=os.environ.get('PRODMA_ROOT')
    if override:
        candidatos=[Path(override).expanduser().resolve()]
    else:
        actual=Path(inicio or Path.cwd()).resolve()
        candidatos=[actual,*actual.parents]
    for candidato in candidatos:
        if (candidato/'datahistorica').is_dir() and (candidato/'nuevosclientes').is_dir():
            return candidato
    raise FileNotFoundError(
        'No se encontró la raíz de Prodma. Abra la carpeta del proyecto en VS Code '
        'o defina la variable de entorno PRODMA_ROOT.'
    )

RAIZ_PROYECTO=encontrar_raiz_proyecto()
DIRECTORIO_HISTORICO=RAIZ_PROYECTO/'datahistorica'
DIRECTORIO_CLIENTES=RAIZ_PROYECTO/'nuevosclientes'
ROOT=RAIZ_PROYECTO/'resultados'; ROOT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'cache_v1'; CACHE.mkdir(parents=True,exist_ok=True)
RUN=ROOT/('ejecucion_'+pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'))
for p in ('tablas','figuras','modelos','pronosticos','logs'):
    (RUN/p).mkdir(parents=True,exist_ok=True)
(RUN/'config.json').write_text(json.dumps(CFG,indent=2,ensure_ascii=False),encoding='utf-8')
(RUN/'entorno.txt').write_text(
    subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
LOG=[]
def log(stage,item,status,detail=''):
    LOG.append(dict(etapa=stage,item=str(item),estado=status,detalle=str(detail)))
    pd.DataFrame(LOG).to_csv(RUN/'logs/eventos.csv',index=False)
print('Proyecto:',RAIZ_PROYECTO)
print('Resultados:',RUN)
""")

nb["cells"][8]["source"] = lines("""## 04 · Inventario local

Exige cuatro históricos (`demanda2023` a `demanda2026`) y un archivo de nuevos clientes. Acepta `.xls` o `.xlsx`, detecta duplicados y nunca elige un archivo ambiguo silenciosamente.
""")

nb["cells"][9]["source"] = lines(r"""# Cuatro históricos anuales (nombre base exacto, extensión .xls o .xlsx)
# y un archivo de nuevos clientes detectado por patrón dentro de su carpeta dedicada.
HIST_BASES=[f'demanda{year}' for year in range(2023,2027)]
PATRON_CLIENTES=re.compile(r'^(?=.*nuevos)(?=.*client).*\.xlsx?$|^(?=.*demanda)(?=.*solicitad).*\.xlsx?$',re.IGNORECASE)

archivos_historicos=[p for p in DIRECTORIO_HISTORICO.rglob('*')
                     if p.is_file() and p.suffix.lower() in ('.xls','.xlsx')]
archivos_clientes=[p for p in DIRECTORIO_CLIENTES.rglob('*')
                   if p.is_file() and p.suffix.lower() in ('.xls','.xlsx')]
todos_los_archivos=archivos_historicos+archivos_clientes
files=[dict(name=p.name,path=str(p.resolve()),size=p.stat().st_size,
            modifiedTime=pd.Timestamp(p.stat().st_mtime,unit='s').isoformat())
       for p in todos_los_archivos]
inventory=pd.DataFrame(files); display(inventory)
inventory.to_csv(RUN/'tablas/inventario_archivos.csv',index=False)

assert PATRON_CLIENTES.match('nuevosclientes.xlsx') and PATRON_CLIENTES.match('Nuevos_Clientes_2026.xlsx')
assert PATRON_CLIENTES.match('Demanda_Solicitada.xlsx') and not PATRON_CLIENTES.match('demanda2025.xls')

selected={}; ESTADO_ARCHIVOS=[]
for base in HIST_BASES:
    patron_hist=re.compile(re.escape(base)+r'\.xlsx?$',re.IGNORECASE)
    matches=[f for f in files if Path(f['path']).parent==DIRECTORIO_HISTORICO.resolve()
             and patron_hist.fullmatch(f['name'])]
    if len(matches)==1:
        nombre_normalizado=base+Path(matches[0]['name']).suffix.lower()
        selected[nombre_normalizado]=matches[0]; estado='encontrado'
    elif len(matches)==0:
        estado='faltante'; nombre_normalizado=base+'.xls/.xlsx'
    else:
        raise ValueError(f'{base}: se esperaba 1 archivo, encontrados {len(matches)}: {[m["path"] for m in matches]}')
    ESTADO_ARCHIVOS.append(dict(archivo=nombre_normalizado,estado=estado,tipo='historico_anual'))

client_matches=[f for f in files if Path(f['path']).is_relative_to(DIRECTORIO_CLIENTES.resolve())
                and PATRON_CLIENTES.fullmatch(f['name'])]
if len(client_matches)==1:
    CLIENTE_ARCHIVO_KEY='nuevosclientes'+Path(client_matches[0]['name']).suffix.lower()
    selected[CLIENTE_ARCHIVO_KEY]=client_matches[0]; estado_clientes='encontrado'
    nombre_clientes=client_matches[0]['name']
elif len(client_matches)==0:
    estado_clientes='faltante'; nombre_clientes='(patrón *nuevos*clientes*.xls* o *demanda*solicitada*.xls*)'
    CLIENTE_ARCHIVO_KEY=None
else:
    raise ValueError('Archivo de nuevos clientes ambiguo: '+str([f['path'] for f in client_matches]))
ESTADO_ARCHIVOS.append(dict(archivo=nombre_clientes,estado=estado_clientes,tipo='nuevos_clientes'))

estado_archivos=pd.DataFrame(ESTADO_ARCHIVOS); display(estado_archivos)
estado_archivos.to_csv(RUN/'tablas/estado_archivos_requeridos.csv',index=False)
FALTANTES=[r['archivo'] for r in ESTADO_ARCHIVOS if r['estado']=='faltante']
HISTORICOS_FALTANTES=[r['archivo'] for r in ESTADO_ARCHIVOS
                      if r['estado']=='faltante' and r['tipo']=='historico_anual']
CLIENTES_DISPONIBLE=(estado_clientes=='encontrado')
for f in FALTANTES:
    log('inventario',f,'FALTANTE','No encontrado en las carpetas locales configuradas')
if not selected:
    raise ValueError('Ningún archivo requerido está disponible; no es posible el diagnóstico')
if FALTANTES:
    print('ADVERTENCIA — archivos locales ausentes (diagnóstico parcial):',*FALTANTES,sep='\n - ')
else:
    print('Los cinco archivos requeridos están presentes.')
print('Disponibles para procesar:',list(selected.keys()))
""")

nb["cells"][10]["source"] = lines("""## 05 · Validación de archivos locales

Resuelve las rutas absolutas, confirma que cada entrada sea un archivo regular no vacío y conserva el tamaño para auditoría. No se copia ni modifica ningún Excel de entrada.
""")

nb["cells"][11]["source"] = lines("""paths={}
for name,meta in selected.items():
    path=Path(meta['path']).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Archivo local no encontrado: {path}')
    if path.stat().st_size<=0:
        raise ValueError(f'Archivo local vacío: {path}')
    if path.stat().st_size!=int(meta['size']):
        raise ValueError(f'El archivo cambió durante el inventario: {path}')
    paths[name]=path
    print(name,'->',path,round(path.stat().st_size/1e6,1),'MB verificados')
""")

for idx in (37, 47, 49):
    source = "".join(nb["cells"][idx]["source"])
    source = source.replace('en Drive', 'en la carpeta local')
    nb["cells"][idx]["source"] = lines(source.replace('ausentes en Drive', 'ausentes en las carpetas locales'))

source50 = "".join(nb["cells"][50]["source"])
source50 = source50.replace(
    '**No ejecutado aquí:** autenticación en su Colab, entrenamiento integral',
    '**No ejecutado aquí:** entrenamiento integral',
).replace(
    '[descargas Drive API](https://developers.google.com/workspace/drive/api/guides/manage-downloads)',
    '[pathlib](https://docs.python.org/3/library/pathlib.html)',
).replace('demanda2023.xlsxxx', 'demanda2023.xlsx')
if 'Sustento académico de la actualización local' not in source50:
    source50 += """

### Sustento académico de la actualización local

- Brown, C. (2021), tesis de maestría, *Reproducible notebooks*: documenta que declarar el entorno y controlar rutas/dependencias son condiciones centrales para reproducir notebooks (DOI: [10.62791/20195](https://doi.org/10.62791/20195)).
- Pimentel et al. (2021), artículo indexado, *Understanding and improving the quality and reproducibility of Jupyter notebooks*: sustenta limpiar estado/salidas heredadas y explicitar dependencias ([DOI 10.1007/s10664-021-09961-9](https://doi.org/10.1007/s10664-021-09961-9)).
- Cerqueira, Torgo y Mozetič (2020), artículo científico, *Evaluating time series forecasting models*: respalda evaluación fuera de muestra que preserve el orden temporal en series no estacionarias ([arXiv 1905.11744](https://arxiv.org/abs/1905.11744)).
- Hong, T. (2010), tesis doctoral, *Short Term Electric Load Forecasting*: fundamento del tratamiento de pronóstico de carga eléctrica y comparación de modelos ([North Carolina State University](https://repository.lib.ncsu.edu/items/c8f6d98e-3422-4a0b-9d5e-91cfeaa0de87)).
- Salehimehr et al. (2022), revisión indexada sobre pronóstico de carga de corto plazo con IA ([DOI 10.1049/tje2.12183](https://doi.org/10.1049/tje2.12183)).
"""
nb["cells"][50]["source"] = lines(source50)

nb.setdefault('metadata', {}).setdefault('kernelspec', {})['display_name'] = 'Python 3.11 (.venv)'
nb['metadata'].pop('colab', None)
for cell in nb['cells']:
    cell.setdefault('metadata', {}).pop('colab', None)
    if cell.get('cell_type')=='code':
        cell['outputs']=[]
        cell['execution_count']=None
NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
print(f'Actualizado: {NOTEBOOK}')
