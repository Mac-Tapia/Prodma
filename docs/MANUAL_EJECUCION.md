# Manual de ejecución — Prodma (proyección de demanda eléctrica)

Guía para correr el proyecto **desde terminal (PowerShell)**, de cero hasta tener los
resultados listos para entregar, usando la CLI `python -m prodma.pipeline`
(`src/prodma/pipeline.py`), que ejecuta el notebook real por fases sin necesidad de
abrir VS Code.

Todos los comandos asumen que estás parado en la raíz del proyecto
(`D:\Proyectos\ELDICA\Prodma`) en PowerShell.

## Resumen rápido (para quien ya conoce el proyecto)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

.\.venv\Scripts\python.exe tools\generar_variables_externas.py

.\.venv\Scripts\python.exe -m prodma.pipeline --modo rapido --backend auto
```

El resto de este documento explica cada paso, qué esperar, cómo verificar que salió
bien y dónde quedan los resultados.

---

## Fase 0 · Preparar el entorno (una sola vez por máquina)

### 0.1 Requisitos previos

- Windows 11, Python 3.11.x instalado (`py -3.11 --version`).
- Los archivos de entrada ya colocados en su lugar (ver Fase 1).
- Opcional: GPU NVIDIA con driver CUDA si quieres entrenamiento acelerado (XGBoost/CatBoost; LightGBM usa OpenCL en Windows y puede no estar disponible, ver Troubleshooting).

### 0.2 Crear el entorno virtual e instalar dependencias

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si vas a entrenar con GPU, instala además el índice oficial de PyTorch con CUDA:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

Instala el propio proyecto en modo editable (habilita `python -m prodma.pipeline` y las
importaciones `from prodma...` en los tests, sin tocar `PYTHONPATH` a mano):

```powershell
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Verifica que el entorno quedó bien armado:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Debe terminar en `38 passed` (el número exacto crece con el proyecto; lo importante es que no haya fallos). Uno de esos tests (`tests/integration/test_real_forecasters.py`)
lee los Excel reales de `datahistorica/` y entrena modelos de verdad (incluida una LSTM en
GPU si hay CUDA disponible) — si este paso falla, no sigas a las fases siguientes sin
resolverlo primero.

### 0.3 El kernel de Jupyter del `.venv`

Jupyter suele traer registrado un kernel llamado `python3` que en esta máquina apunta a
un Python global, **no** al `.venv` del proyecto — si lo usas por error verás errores de
"faltan dependencias" aunque `requirements.txt` esté instalado. La CLI del paso 0.2
(`python -m prodma.pipeline`) **registra y usa automáticamente** un kernel propio
(`prodma_venv`) apuntando siempre al Python de este `.venv`, así que con la CLI no hace
falta ningún paso manual. Solo si vas a ejecutar el notebook **interactivamente en VS
Code** (en vez de por terminal) debes seleccionar el kernel tú mismo la primera vez:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name=prodma_venv --display-name "Python (Prodma .venv)"
```

y elegir **"Python (Prodma .venv)"** en el selector de kernel del notebook.

---

## Fase 1 · Verificar los datos de entrada

El proyecto exige exactamente estos archivos, ya colocados (nunca se descargan ni se
generan solos):

```
datahistorica/demanda2023.xlsx
datahistorica/demanda2024.xlsx
datahistorica/demanda2025.xlsx
datahistorica/demanda2026.xlsx
nuevosclientes/nuevosclientes.xlsx   (o un archivo que matchee *nuevos*clientes*.xls*)
```

Verificación rápida:

```powershell
Get-ChildItem datahistorica, nuevosclientes -Filter *.xls*
```

Si falta alguno, el notebook igual corre pero marca la ejecución como `PARCIAL` en
`RESUMEN.json` (no falla en seco) — revisa la Fase 6 para saber dónde ver eso.

---

## Fase 2 · Generar las variables externas (exógenas)

Genera calendario de feriados (offline) y temperatura horaria/15-min (Open-Meteo, requiere
internet) en `variables_externas/`:

```powershell
.\.venv\Scripts\python.exe tools\generar_variables_externas.py
```

Esto es necesario **antes** de correr el notebook si `CFG['usar_temperatura_climatologica']`
está en `True` (por defecto lo está): la celda 12 del notebook falla explícitamente si no
encuentra ningún archivo en `variables_externas/temperatura/`. Solo hace falta regenerarlo
si cambian los años de interés o quieres refrescar la temperatura descargada; no hace falta
correrlo antes de cada ejecución del notebook.

---

## Fase 3 · Configurar la ejecución

Dos de las opciones más importantes se controlan con flags de la CLI, sin editar el
notebook:

| Flag | Equivale a | Valores |
|---|---|---|
| `--modo` | `CFG['modo']` — `'rapido'` (2 medidores, ventana corta) o `'completo'` (todos los medidores elegibles, histórico completo) | `rapido` (default si se omite), `completo` |
| `--backend` | `CFG['backend']` — usado por el preflight GPU de la sección 13 | `auto` (default), `cpu`, `gpu` |

El resto de las opciones (escenarios de planificación, exportar a Excel, umbrales
estadísticos, etc.) todavía se editan directamente en la **celda 5** del notebook
(sección "02 · Configuración"), diccionario `CFG`:

| Clave | Qué controla | Valor por defecto |
|---|---|---|
| `CFG['max_medidores_rapido']` | cuántos medidores procesa el modo rápido | `2` |
| `CFG['fin_plan']` | fin del horizonte de pronóstico (fin de intervalo: `2028-01-01 00:00` = cierre de 2027) | `'2028-01-01 00:00'` |
| `CFG['escenarios']` | diccionario de tasas de crecimiento para la planificación hasta `fin_plan` (vacío = la sección 22 queda PENDIENTE) | `{}` |
| `CFG['usar_temperatura_climatologica']` | agrega temperatura climatológica como exógena (requiere Fase 2 ya corrida) | `True` |
| `CFG['exportar_excel']` | además de Parquet, exporta también `.xlsx` | `True` |
| `CFG['reanudar_run']` | ruta de una corrida previa para continuar sin reentrenar (lo fija `--reanudar`) | `None` |
| `CFG['max_dias_sin_datos']` | series sin dato en más de N días respecto al último global no se proyectan | `180` |
| `CFG['umbral_sanidad']` | desvío máximo de la media proyectada frente a los últimos 12 meses antes de marcar `REVISAR` | `0.25` |
| `CFG['workers_pronostico']` | procesos del pronóstico recursivo (cada uno carga su modelo: cuidar la RAM) | `6` |
| `CFG['backtest_horizontes']` | horizontes (pasos de 15 min) del backtest sin reentrenar; el mayor (2688 = 28 días) decide el método de cada serie | `(96, 672, 2688)` |
| `CFG['margen_ml']` | el ML solo se usa si su RMSE a 28 días es menor que `margen` × el del mejor perfil | `0.95` |
| `CFG['max_medidores_adicionales']` | tope de medidores para la sección 26 (`None` = todos los que tengan serie activa) | `None` |

Para una primera corrida de validación deja `--modo rapido`. Para la corrida que vas a
entregar como resultado final, usa `--modo completo` (tardará mucho más: entrena los 6
modelos + 3 benchmarks para **todos** los medidores elegibles, no solo 2).

---

## Fase 4 · Ejecutar el notebook, por fases, desde terminal

`python -m prodma.pipeline` ejecuta el notebook real con `nbclient` (misma lógica, no
la reimplementa), inyecta `--modo`/`--backend` en `CFG` y **nunca modifica el archivo
fuente** `notebook\proyecciondemanda.ipynb` — siempre escribe una copia ejecutada
aparte, en `resultados/cli_<fecha>_<hora>/notebook_ejecutado.ipynb`, junto a un
`*.manifest.json` con el estado, el error (si lo hubo) y la carpeta real de resultados
del pipeline (`resultados/<modo>-<backend>_<fecha>_<hora>/`, la que crea el propio notebook).

**Ver las fases disponibles** (son las 26 secciones numeradas del propio notebook, no un
esquema aparte):

```powershell
.\.venv\Scripts\python.exe -m prodma.pipeline --listar-fases
```

**Ejecutar todo, de punta a punta:**

```powershell
.\.venv\Scripts\python.exe -m prodma.pipeline --modo rapido --backend auto
```

**Ejecutar solo hasta una fase y parar ahí** (por ejemplo, hasta la fase 14 — selección y
entrenamiento de modelos — sin llegar todavía a diagnósticos, pronóstico ni exportación):

```powershell
.\.venv\Scripts\python.exe -m prodma.pipeline --modo rapido --hasta-fase 14
```

Cada corrida normal arranca desde la fase 1 (las fases son *acumulativas*: para llegar
a la 14 hay que pasar por la 1-13, porque el notebook depende de ese estado). Entre
corridas normales solo se reaprovechan las cachés de datos (`resultados/cache_v1/`,
`cache_matriz_v1/`, `cache_limpio_v1/`): **el entrenamiento se repite completo** y cada
corrida crea una carpeta nueva. Para NO repetirlo, usa la reanudación (ver abajo).

Otros flags útiles: `--timeout` (segundos por celda, default 7200 — la primera lectura
de los Excel de `datahistorica/`, ~550 MB en total, es la parte más lenta; la celda del
pronóstico recursivo puede tardar más, usa p. ej. `--timeout 43200`), `--notebook` (ruta
alternativa), `--output` (ruta alternativa para la copia ejecutada) y
`--set CLAVE=VALOR` (sobrescribe una clave de `CFG`, repetible; p. ej.
`--set workers_pronostico=8`).

### Reanudar una corrida sin reentrenar (`--reanudar`)

Si una corrida completa se interrumpió después del entrenamiento (las secciones 14-16
tardan decenas de horas), **no hace falta repetirlo**: los modelos (`modelos/*.joblib`) y
las tablas (`metricas.csv`, `comparacion_completa.csv`, `OLS_referencia.csv`) ya están en
disco. Continúa en la MISMA carpeta con:

```powershell
.\.venv\Scripts\python.exe -m prodma.pipeline --reanudar resultados\C-AUT_20260918_105110 --timeout 43200
```

Qué hace (el `--modo` se toma de la corrida original; no se entrena nada):

1. Ejecuta las secciones 1-13 con las cachés y **omite** el entrenamiento y los
   diagnósticos ya persistidos (celdas etiquetadas `prodma.omitir_al_reanudar`).
2. **Sección 14b — restauración**: vuelve a predecir el tramo de prueba con cada modelo
   guardado, sobre el mismo corte train/test, y **valida** que el RMSE recalculado coincida
   con `metricas.csv` (`tablas/validacion_restauracion.csv`). Si no coincide, aborta. El
   estado queda en `tablas/estado_restaurado.joblib`, así que las siguientes reanudaciones
   tardan segundos en este paso.
3. El ganador operativo de cada serie es el mejor modelo **ML** por `CV_RMSE`
   (los benchmarks `Estacional_*` no se usan para pronosticar: recursivamente son una línea
   plana). Cada serie se clasifica en `tablas/estado_series.csv`: `OK`,
   `SERIE_CONSTANTE` (se proyecta como constante declarada) o `SIN_DATOS_RECIENTES`
   (`CFG['max_dias_sin_datos']`, no se proyecta).
4. Sigue desde la sección 17 hasta el cierre. La sección 20 usa los modelos guardados
   **sin reentrenar** —entrenados con el 80 % cronológico del histórico—. Un modelo entrenado
   a 1 paso **pierde el nivel en pocos días** al pronosticar de forma recursiva (converge a un
   punto fijo; el R² a 1 paso no lo revela), así que el método de cada serie se **elige por
   backtest a 28 días** sobre datos que el modelo no vio: ML recursivo, perfil calendario
   (mediana de 12 meses) o perfil anclado (forma estacional × nivel reciente
   desestacionalizado, sin entrenamiento). El ML solo se usa donde mejora al mejor perfil en
   más de `1-CFG['margen_ml']` y pasa las guardas de sanidad. El resultado por serie queda en
   `tablas/seleccion_metodo.csv` y en la columna `fuente` de
   `pronosticos/serie_<medidor>_<var>.parquet` (checkpoint por serie: si se interrumpe, al
   reanudar solo se calculan las pendientes; borra esos archivos si cambias la lógica). El
   ML se proyecta hasta el cierre solo donde ganó, con una serie por proceso
   (`CFG['workers_pronostico']`).

Con `CFG['fin_plan']='2028-01-01 00:00'` el horizonte cierra el 31-dic-2027 (la marca
00:00 del 1-ene cierra el año, convención de fin de intervalo). Las guardas de sanidad
comparan cada proyección ML con los últimos 12 meses (`CFG['umbral_sanidad']`); si se
aparta, se aplana o sale de rango, se descarta y se usa el mejor perfil
(`tablas/sanidad_proyecciones.csv`, columna `fallback`). Cada archivo por serie guarda
`pred` (la proyección final) y las alternativas `pred_ml`, `pred_perfil_anclado` y
`pred_perfil_calendario`.

Si prefieres verlo correr celda a celda de forma interactiva (para revisar una figura
antes de seguir, o pausar con más control que el que dan las 26 fases), ábrelo en VS
Code con el kernel `Python (Prodma .venv)` (Fase 0.3) y ejecútalo ahí — para eso sigue
sirviendo el flujo manual.

---

## Fase 5 · Verificar que la ejecución salió bien

Cada corrida crea una carpeta independiente `resultados/<modo>-<backend>_<fecha>_<hora>/`.
Verifica en este orden:

1. **`RESUMEN.json`** (en la raíz de la carpeta de la ejecución) — campo
   `estado_ejecucion`: `COMPLETA_SEGUN_INSUMOS_DISPONIBLES` (bien) vs `PARCIAL`
   (faltó algún insumo; revisa el campo `limitaciones` del mismo JSON para saber cuál).
   También trae `criterios_estadisticos_modelos_ganadores` (cuántos modelos ganadores
   cumplieron Durbin-Watson/Ljung-Box/homocedasticidad/R²≥0.90) y
   `criterios_estadisticos_ols_referencia`.
2. **`tablas/logs/eventos.csv`** (o `logs/eventos.csv`) — filtra por `estado=ERROR` para
   ver si algo falló silenciosamente en un medidor puntual sin detener toda la
   ejecución (el notebook está diseñado para seguir con los demás medidores aunque uno
   falle).
3. **`tablas/comparacion_completa.csv`** — columna `cumple_minimos` por
   medidor/variable/modelo (ver Manual de criterios estadísticos, celdas 15-16 del
   notebook).

```powershell
Get-Content resultados\<modo>-<backend>_<fecha>_<hora>\RESUMEN.json
```

---

## Fase 6 · Dónde están los resultados para entregar

Dentro de `resultados/<modo>-<backend>_<fecha>_<hora>/`:

| Carpeta/archivo | Contenido |
|---|---|
| `tablas/Matriz_Consolidada.parquet` (`.xlsx`) | Histórico P/Q de todos los medidores, una sola matriz |
| `tablas/metricas.csv` | CV_RMSE y métricas de los 6 modelos + 3 benchmarks por medidor/variable |
| `tablas/comparacion_completa.csv` | Métricas + diagnósticos + columnas `cumple_*` (criterios estadísticos) |
| `tablas/OLS_referencia.csv` | Inferencia clásica (F, t-Student, R² ajustado) del modelo OLS de referencia |
| `tablas/calidad_estadistica.csv`, `recorte_atipicos_boxplot.csv` | Calidad de datos e imputación |
| `tablas/resumen_planificacion.csv` | Resumen de escenarios de planificación (si `CFG['escenarios']` no está vacío) |
| `tablas/Escenarios_DIgSILENT_MaxPromMin.xlsx` | Tabla lista para pegar en PowerFactory |
| `pronosticos/operativo_<medidor>_<var>_<año>.parquet/xlsx` | Pronóstico ML recursivo, un archivo por año calendario |
| `pronosticos/Matriz_Operativa.xlsx` | Pronóstico total (base + nuevos clientes), todos los medidores |
| `pronosticos/serie_<medidor>_<var>.parquet` | Checkpoint por serie: `pred` (final), `pred_ml`, `pred_perfil`, `fuente`, marcas de sanidad |
| `tablas/estado_series.csv`, `estado_proyeccion.csv` | Estado de cada serie (`OK`/`SERIE_CONSTANTE`/`SIN_DATOS_RECIENTES`) y si se proyectó |
| `tablas/sanidad_proyecciones.csv` | Guardas de sanidad: desvío de la media, aplanamiento, fuera de rango y fuente usada (ML o perfil calendario) |
| `tablas/verificacion_proyecciones.csv` | Cada proyección cubre hasta `fin_plan`, sin huecos, finita y no plana |
| `tablas/backtest_recursivo.csv`, `seleccion_metodo.csv` | Backtest sin reentrenar por serie y método (ML, perfiles, semanal) y método elegido por serie (`fuera_de_muestra` indica si la ventana no entró al entrenamiento) |
| `tablas/Escenarios_DIgSILENT_<año>_proyectado.xlsx` | Máx/prom/mín proyectadas por medidor para el último año del horizonte |
| `tablas/validacion_restauracion.csv`, `estado_restaurado.joblib` | Solo al reanudar: RMSE guardado vs. restaurado de cada modelo y checkpoint del estado |
| `pronosticos/plan_<medidor>_<escenario>_<año>.xlsx` | Planificación por perfiles/tasas (si se configuraron escenarios) |
| `figuras/*.png` | Todas las gráficas (curvas, boxplots, correlaciones, proyecciones, entrenamiento) |
| `modelos/*.joblib` | Modelos entrenados serializados |
| `RESUMEN.json` | Resumen ejecutivo de la corrida (ver Fase 5) |
| `entorno.txt` | `pip freeze` exacto de esa corrida, para reproducibilidad |

Para entregar resultados, comparte la carpeta `resultados/<modo>-<backend>_<fecha>_<hora>/`
completa (o el subconjunto de `tablas/`, `figuras/` y `pronosticos/` que corresponda) —
nunca se sobreescribe una corrida anterior, así que puedes comparar entre ejecuciones.

**Cuál carpeta es la más reciente:** con varias corridas acumuladas es fácil abrir por
error una carpeta vieja y confundir una figura ya corregida con una desactualizada.
`python -m prodma.pipeline` escribe siempre `resultados/ULTIMA_EJECUCION.txt` con la
ruta exacta (y el estado) de la corrida más reciente — revisa ese archivo antes de abrir
una figura o tabla si no estás seguro de cuál carpeta es la actual.

---

## Limitaciones que hay que tener presentes (no ocultarlas al entregar)

- **La CLI (`python -m prodma.pipeline`) envuelve el notebook, no lo reemplaza**: las
  "fases" son las 26 secciones del propio notebook y buena parte de la lógica de negocio
  sigue viviendo ahí. Ya están en `src/prodma/` la reanudación (`reanudar.py`), el
  pronóstico recursivo (`recursivo.py`), las guardas de sanidad (`sanidad.py`) y la
  transformación del target (`transformaciones.py`); no hay todavía un manifiesto con
  reanudación fina por etapa como el del plan de arquitectura completo. Lo que se
  reanuda de verdad es lo descrito en «Reanudar una corrida sin reentrenar» (Fase 4).
- **Pronóstico sin reentrenar con el 100 % del histórico.** Con `--reanudar`, los modelos
  se entrenaron con el 80 % cronológico más antiguo de cada serie y los árboles no
  extrapolan por encima de lo visto al entrenar. La recursión ML a 15 min **pierde el
  nivel en pocos días**: en el backtest de la corrida `C-AUT_20260918_105110` (130 series,
  124 fuera de muestra) el RMSE mediano relativo al desvío real fue, a 1 día / 7 días / 28
  días, de 0,90 / 1,23 / 1,32 para el ML recursivo frente a 0,69 / 0,78 / 0,96 del perfil
  anclado (1,0 = no mejorar a la media). Por eso la mayoría de las series se proyectan con
  perfiles (calendario o anclado) elegidos por backtest a 28 días. **No hay validación a 16 meses**
  y ninguna proyección incluye crecimiento más allá del nivel reciente (la sección 22, con
  tasas de crecimiento explícitas, sigue PENDIENTE hasta que se definan). Revisa
  `seleccion_metodo.csv` y `sanidad_proyecciones.csv` antes de entregar.
- **Series sin datos recientes o constantes.** No se proyectan (o se proyectan como
  constante declarada); ver `estado_proyeccion.csv`. Confirma con el operador si los
  medidores sin datos recientes fueron dados de baja.
- **LightGBM en GPU (OpenCL) puede no estar disponible en Windows** aunque XGBoost/CatBoost
  sí lo estén: la celda 13 lo prueba con un microentrenamiento real y cae a CPU
  automáticamente sin detener la ejecución; revisa `logs/backends_gpu.csv` de la corrida
  para ver qué backend se usó realmente en cada modelo.
- **Los criterios estadísticos mínimos (Durbin-Watson, Ljung-Box, homocedasticidad, R²≥0.90,
  F/t-Student)** se calculan y se registran como ADVERTENCIA si no se cumplen, pero **no**
  descartan un modelo ni detienen la ejecución — revisa `comparacion_completa.csv` antes
  de dar un pronóstico por bueno sin mirarlo.
- La temperatura usada como exógena es una **climatología** (mediana histórica por
  mes/hora), no un pronóstico meteorológico real, y usa una sola coordenada
  representativa de Ica para toda el área de concesión — ver
  `variables_externas/README.md`.
