# Proyección de demanda Electro Dunas con GPU local — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convertir el notebook existente en un pipeline local reproducible que lea los cinco Excel reales, entrene modelos CPU/GPU con validación temporal, simule los horizontes aprobados y produzca salidas auditables sin depender de Colab ni Google Drive.

**Architecture:** El notebook seguirá siendo la interfaz narrativa, pero la lógica se extraerá a módulos Python pequeños y comprobables. Un perfil de cómputo `auto|cpu|gpu` hará pruebas reales de entrenamiento antes de habilitar cada backend; los fallos GPU quedarán registrados y solo activarán CPU cuando la política lo permita. La predicción operacional ML se limitará a horizontes validados de 1 y 7 días; la planeación de 1–5 años será un escenario separado basado en perfiles, crecimiento sustentado y nuevas cargas, no una recursión ML presentada como pronóstico validado.

**Tech Stack:** Windows 11, Python 3.11.9, Jupyter/VS Code, pandas, NumPy, PyArrow, scikit-learn, statsmodels, XGBoost 3.2, LightGBM 4.7, CatBoost 1.2, CUDA/NVIDIA y OpenCL cuando esté disponible.

**Spec:** `E:/Plan_implementacion_demanda_Electro_Dunas.md` (documento fuente actualizado en paralelo con este plan).

## Global Constraints

- Entradas obligatorias: `datahistorica/demanda2023.xlsx` a `demanda2026.xlsx` y `nuevosclientes/nuevosclientes.xlsx`.
- Los Excel originales son de solo lectura y nunca se sobrescriben.
- Zona horaria de negocio: `America/Lima`; frecuencia histórica: 15 minutos; la marca se conserva como fin de intervalo.
- Kernel local: `.venv/Scripts/python.exe`, Python 3.11.x; no instalar paquetes desde una celda.
- Hardware comprobado el 16/09/2026: NVIDIA GeForce RTX 4060 Laptop, 8 GB VRAM, controlador 560.94, CUDA reportada 12.6.
- `compute.backend=auto` prueba GPU y registra el backend efectivo; `gpu` falla de forma explícita si ningún backend GPU supera su prueba; `cpu` nunca intenta GPU.
- XGBoost GPU: `tree_method="hist", device="cuda:0"`.
- CatBoost GPU: `task_type="GPU", devices="0"`; sus resultados GPU no se declararán bit-a-bit deterministas.
- LightGBM Windows: probar `device_type="gpu"` (OpenCL), no `device_type="cuda"`; si el paquete no soporta GPU, registrar `GPU_UNAVAILABLE` y aplicar la política configurada.
- Una falta de VRAM puede reducir lote/paralelismo una sola vez; nunca cambiar silenciosamente datos, horizonte o espacio de hiperparámetros.
- Toda partición, imputación, escalado y selección de variables respeta el orden temporal y se ajusta únicamente con entrenamiento.
- “Cero errores” significa: pruebas con salida 0, manifiesto sin etapas `ERROR`, cinco entradas verificadas y artefactos de aceptación presentes; no significa ausencia absoluta de defectos futuros.
- Variables externas no disponibles son opcionales y se marcan `PENDIENTE`; no se rellenan con cero ni se presentan como observadas.

---

## Estructura objetivo

```text
Prodma/
├─ config/default.yaml
├─ src/prodma/
│  ├─ config.py          # configuración tipada y rutas
│  ├─ inventory.py       # inventario e integridad de entradas
│  ├─ excel_reader.py    # lectura por hojas y caché
│  ├─ quality.py         # tiempo, duplicados, huecos y unidades
│  ├─ features.py        # rezagos y calendario causales
│  ├─ compute.py         # detección CPU/CUDA/OpenCL y pruebas GPU
│  ├─ models.py          # fábricas de estimadores y early stopping
│  ├─ evaluation.py      # cortes temporales, métricas y benchmarks
│  ├─ forecast.py        # 1 día, 7 días y escenarios separados
│  ├─ new_loads.py       # altas constantes por medidor
│  └─ pipeline.py        # orquestación, checkpoints y manifiesto
├─ notebook/proyecciondemanda.ipynb
├─ tests/unit/
├─ tests/integration/
└─ resultados/<run_id>/
```

## Task 1: Congelar contrato de configuración y ejecución

**Files:**
- Create: `config/default.yaml`
- Create: `src/prodma/__init__.py`
- Create: `src/prodma/config.py`
- Create: `tests/unit/test_config.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `load_config(path: Path | None) -> AppConfig`; `AppConfig.project_root`, `inputs`, `output_root`, `compute.backend`, `forecast.operational_days`.

- [ ] **Step 1: Escribir pruebas fallidas** que comprueben descubrimiento desde la raíz y desde `notebook/`, rechazo de backend desconocido y horizontes operacionales exactos `[1, 7]`.
- [ ] **Step 2: Ejecutar RED:** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_config.py -q`; debe fallar porque `prodma.config` aún no existe.
- [ ] **Step 3: Implementar configuración mínima** con `dataclasses`, `pathlib` y YAML; permitir `PRODMA_ROOT` y sobreescrituras por CLI, sin rutas absolutas persistidas.
- [ ] **Step 4: Ejecutar GREEN** y comprobar que la configuración serializada se guarda en cada ejecución.
- [ ] **Step 5: Commit:** `git add config src/prodma/config.py tests/unit/test_config.py requirements.txt && git commit -m "feat: add typed local pipeline configuration"`.

## Task 2: Inventario verificable de los cinco Excel

**Files:**
- Create: `src/prodma/inventory.py`
- Create: `tests/unit/test_inventory.py`
- Create: `tests/integration/test_real_inputs.py`

**Interfaces:**
- Consumes: `AppConfig`.
- Produces: `build_inventory(config) -> list[InputFile]`; cada elemento contiene `kind`, `year`, `path`, `size`, `mtime_ns`, `sha256`.

- [ ] **Step 1: Escribir pruebas fallidas** para cinco archivos válidos, histórico faltante, duplicado, archivo vacío y nombre ambiguo de nuevos clientes.
- [ ] **Step 2: Ejecutar RED** y confirmar que falla por ausencia de `build_inventory`.
- [ ] **Step 3: Implementar inventario de solo lectura**, SHA-256 por bloques, nombres anuales exactos y patrón controlado de nuevos clientes.
- [ ] **Step 4: Ejecutar pruebas unitarias y la integración real:** `.\.venv\Scripts\python.exe -m pytest tests/unit/test_inventory.py tests/integration/test_real_inputs.py -q`.
- [ ] **Step 5: Criterio real:** 2023–2026 y nuevos clientes encontrados; tamaños mayores que cero; ninguna entrada fuera de sus directorios.
- [ ] **Step 6: Commit:** `git commit -m "feat: validate demand input inventory"`.

## Task 3: Lector de Excel, caché y esquema

**Files:**
- Create: `src/prodma/excel_reader.py`
- Create: `tests/unit/test_excel_reader.py`
- Create: `tests/integration/test_excel_samples.py`

**Interfaces:**
- Produces: `read_meter_sheet(path, sheet, tolerance_seconds) -> MeterFrame`; `read_new_loads(path) -> DataFrame`.

- [ ] **Step 1: Probar primero** alias `Local Time`, `kW tot[ mean]`, `kVAR tot[ mean]`, hojas `|PRIVATE|`, extensiones `.xls/.xlsx`, fechas Excel, tolerancia y duplicados contradictorios.
- [ ] **Step 2: Ver RED.**
- [ ] **Step 3: Implementar lectura iterativa**, motor por extensión, caché Parquet con clave `sha256+hoja+versión_parser` y auditoría JSON.
- [ ] **Step 4: Integración real:** abrir una hoja útil de cada histórico y `demandanueva.`; comprobar columnas y fechas sin cargar todos los libros en RAM simultáneamente.
- [ ] **Step 5: Ver GREEN** y medir memoria máxima; el lector debe cerrar siempre el libro, incluso ante excepción.
- [ ] **Step 6: Commit:** `git commit -m "feat: add cached schema-aware Excel reader"`.

## Task 4: Calidad temporal y transformación causal

**Files:**
- Create: `src/prodma/quality.py`
- Create: `tests/unit/test_quality.py`

**Interfaces:**
- Produces: `normalize_meter(frame) -> CleanSeries`; `causal_fill(series) -> tuple[Series, AuditFrame]`.

- [ ] **Step 1: Probar huecos, cero observado, duplicado igual, duplicado conflictivo, valores no finitos, fin de intervalo y borde anual.**
- [ ] **Step 2: Ver RED.**
- [ ] **Step 3: Implementar sin interpolación futura:** relleno corto hacia adelante y rezago semanal solo cuando exista; conservar máscara `observed`.
- [ ] **Step 4: Comprobar balance:** si P es potencia media confirmada, `E_MWh=P_kW*0.25/1000`; si no está confirmada, producir `energia_estado=PENDIENTE`.
- [ ] **Step 5: Ver GREEN y commit.**

## Task 5: Preflight GPU local y política de fallback

**Files:**
- Create: `src/prodma/compute.py`
- Create: `tests/unit/test_compute_policy.py`
- Create: `tests/integration/test_gpu_smoke.py`

**Interfaces:**
- Produces: `probe_compute(config) -> ComputeReport`; `backend_for(model_name, report) -> BackendDecision`.

- [ ] **Step 1: Escribir pruebas fallidas** para políticas `cpu`, `gpu`, `auto`, CUDA ausente, OpenCL ausente, OOM y fallo de entrenamiento.
- [ ] **Step 2: Ver RED.**
- [ ] **Step 3: Implementar sondeo:** capturar `nvidia-smi --query-gpu`, versiones de paquetes y memoria; ejecutar un `fit/predict` real pequeño, no confiar solo en imports.
- [ ] **Step 4: XGBoost smoke:** `XGBRegressor(tree_method="hist", device="cuda:0", n_estimators=8)`; exigir predicciones finitas.
- [ ] **Step 5: CatBoost smoke:** `CatBoostRegressor(task_type="GPU", devices="0", iterations=8)`; registrar que GPU no es determinista bit a bit.
- [ ] **Step 6: LightGBM smoke en Windows:** `LGBMRegressor(device_type="gpu", gpu_device_id=0, n_estimators=8)`; si la rueda carece de OpenCL, estado `GPU_UNAVAILABLE`, nunca fingir CUDA.
- [ ] **Step 7: Ver GREEN:** con `backend=gpu`, XGBoost y CatBoost deben pasar; LightGBM queda habilitado solo si su prueba real pasa. Con `auto`, CPU es fallback explícito y auditado.
- [ ] **Step 8: Guardar `compute_report.json`** con GPU, driver, VRAM total/libre, backend solicitado/efectivo y razón de fallback.
- [ ] **Step 9: Commit:** `git commit -m "feat: add verified local GPU compute policy"`.

## Task 6: Variables causales y conjuntos temporales

**Files:**
- Create: `src/prodma/features.py`
- Create: `src/prodma/evaluation.py`
- Create: `tests/unit/test_features.py`
- Create: `tests/unit/test_temporal_splits.py`

**Interfaces:**
- Produces: `make_features(history, origin) -> DataFrame`; `rolling_origins(index, horizons, folds) -> list[Split]`.

- [ ] **Step 1: Pruebas de fuga:** modificar valores posteriores al origen no puede alterar ninguna variable anterior; escalador e imputador no ven evaluación.
- [ ] **Step 2: Ver RED.**
- [ ] **Step 3: Implementar rezagos 1, 96, 672; medias/desviaciones desplazadas; calendario; temperatura solo cuando su disponibilidad sea válida.**
- [ ] **Step 4: Implementar backtesting rolling-origin** para 1 y 7 días con separación compatible con el horizonte.
- [ ] **Step 5: Métricas:** RMSE, MAE, WAPE, MASE, sesgo energético, error de pico y hora del pico; MAPE solo con denominador válido.
- [ ] **Step 6: Ver GREEN y commit.**

## Task 7: Fábricas de modelos CPU/GPU

**Files:**
- Create: `src/prodma/models.py`
- Create: `tests/unit/test_models.py`
- Create: `tests/integration/test_models_real_slice.py`

**Interfaces:**
- Produces: `make_model(name, decision, seed) -> Regressor`; `fit_with_early_stopping(...) -> FittedModel`.

- [ ] **Step 1: Probar parámetros efectivos por backend** y que `RandomForest`, Ridge, statsmodels y MLP permanezcan en CPU.
- [ ] **Step 2: Ver RED.**
- [ ] **Step 3: Implementar XGBoost/CatBoost GPU y LightGBM OpenCL condicionado al preflight; `GridSearchCV(n_jobs=1)` para no competir por una sola GPU.**
- [ ] **Step 4: Implementar early stopping con un tramo temporal reservado fuera del ajuste.**
- [ ] **Step 5: OOM:** reducir una vez `max_bin`/profundidad o tamaño de lote conforme a configuración; si vuelve a fallar, registrar error y continuar con modelos restantes, sin reducir datos ocultamente.
- [ ] **Step 6: Ejecutar un slice real de 30 días** en P y Q para un medidor; exigir predicciones finitas y modelos serializables.
- [ ] **Step 7: Commit:** `git commit -m "feat: add CPU and GPU model factories"`.

## Task 8: Selección honesta y escenarios de pronóstico

Antes de esta tarea se ejecuta la ampliación de modelos exigida por el documento fuente:

### Task 7A: Modelos estadísticos, Prophet y LSTM

**Files:**
- Create: `src/prodma/forecast_models.py`
- Create: `tests/unit/test_forecast_models.py`
- Create: `requirements-gpu.txt`
- Modify: `requirements.txt`
- Modify: `notebook/proyecciondemanda.ipynb`

**Interfaces:**
- Produces: `available_model_names() -> tuple[str, ...]`; `create_forecaster(name, seasonal_periods, **kwargs) -> BaseForecaster`.

- [x] **Step 1:** probar el registro requerido: ingenuo estacional, ARIMA, SARIMA, Holt-Winters, SARIMAX, Prophet, Random Forest, XGBoost y LSTM.
- [x] **Step 2:** implementar un contrato uniforme `fit/predict` con predicciones finitas y horizonte exacto.
- [x] **Step 3:** exigir exógenas históricas y futuras alineadas para SARIMAX.
- [x] **Step 4:** implementar Prophet como dependencia explícita y nunca asumir que supera referencias.
- [x] **Step 5:** implementar LSTM PyTorch con `device=auto|cpu|gpu`, normalización ajustada solo con entrenamiento y selección CUDA comprobada mediante `torch.cuda.is_available()`.
- [x] **Step 6:** instalar PyTorch desde el índice oficial CUDA 12.6 descrito en `requirements-gpu.txt` y ejecutar una prueba real LSTM sobre la RTX 4060.
- [x] **Step 7:** comparar modelos estadísticos/Prophet/LSTM en la capa mensual; no ejecutar SARIMA estacional semanal sobre millones de filas de 15 minutos.
- [x] **Step 8:** exportar métricas, parámetros, disponibilidad y backend por modelo.

**Files:**
- Create: `src/prodma/forecast.py`
- Create: `tests/unit/test_forecast.py`
- Create: `tests/integration/test_operational_forecast.py`

**Interfaces:**
- Produces: `forecast_operational(model, origin, days) -> ForecastFrame`; `forecast_planning(profile, assumptions) -> ScenarioFrame`.

- [ ] **Step 1: Probar benchmark semanal, horizontes exactos de 96 y 672 pasos y prohibición de fechas solapadas con el histórico.**
- [ ] **Step 2: Ver RED.**
- [ ] **Step 3: Seleccionar ganador por medidor, variable y horizonte**, exigiendo mejora consistente o conservando la referencia estacional.
- [ ] **Step 4: Separar largo plazo:** generar 1–5 años con perfil calendario y supuestos versionados; prohibir rotular la recursión ML hasta 2037 como validada.
- [ ] **Step 5: Exportar intervalos solo si se valida su cobertura; las bandas residuales simples se etiquetan `exploratorias`.**
- [ ] **Step 6: Ver GREEN y commit.**

## Task 9: Nuevos clientes y reconciliación eléctrica

**Files:**
- Create: `src/prodma/new_loads.py`
- Create: `tests/unit/test_new_loads.py`
- Create: `tests/integration/test_real_new_loads.py`

**Interfaces:**
- Produces: `apply_new_loads(base, stages) -> TotalForecast`.

- [ ] **Step 1: Probar que 1,600 kW constantes añaden 0.4 MWh por intervalo y 38.4 MWh/día, únicamente al medidor y desde la fecha indicada.**
- [ ] **Step 2: Probar etapas incrementales/acumuladas, factor de potencia inválido, fecha no confirmada, medidor inexistente, transferencia firmada y alta ya incluida.**
- [ ] **Step 3: Ver RED.**
- [ ] **Step 4: Implementar P/Q/S/FP y auditoría; una etapa pendiente no se suma.**
- [ ] **Step 5: Ejecutar integración con `nuevosclientes.xlsx`; producir lista explícita de altas confirmadas, pendientes y no coincidentes.**
- [ ] **Step 6: Ver GREEN y commit.**

## Task 10: Orquestador reanudable y manifiesto de ejecución

**Files:**
- Create: `src/prodma/pipeline.py`
- Create: `tests/unit/test_pipeline_resume.py`
- Create: `tests/integration/test_pipeline_smoke.py`

**Interfaces:**
- Produces: CLI `python -m prodma.pipeline --config config/default.yaml --mode smoke|full --backend auto|cpu|gpu`.

- [ ] **Step 1: Probar estado de etapas, reanudación desde caché válida, invalidación por SHA-256 y bloqueo de dos ejecuciones sobre el mismo `run_id`.**
- [ ] **Step 2: Ver RED.**
- [ ] **Step 3: Implementar etapas idempotentes** `inventory→parse→quality→features→train→evaluate→forecast→loads→report`.
- [ ] **Step 4: Escribir `manifest.json` atómicamente** con inicio/fin, configuración, hashes, hardware, backends, semillas, modelos, métricas, errores, advertencias y artefactos.
- [ ] **Step 5: El modo `smoke`** usa dos medidores, 30 días, los modelos principales y 1 día; el modo `full` usa todos los medidores elegibles, histórico completo y horizontes 1/7 días más escenarios.
- [ ] **Step 6: Ver GREEN y commit.**

## Task 11: Adaptar el notebook como interfaz fina

**Files:**
- Modify: `notebook/proyecciondemanda.ipynb`
- Create: `tests/integration/test_notebook_contract.py`

**Interfaces:**
- Consumes: CLI/módulos del pipeline; no duplica funciones de producción.

- [ ] **Step 1: Probar que no contiene Colab/Drive API, `pip install`, rutas `/content`, definiciones duplicadas del pipeline ni salidas antiguas.**
- [ ] **Step 2: Ver RED.**
- [ ] **Step 3: Sustituir celdas monolíticas** por configuración, invocación, lectura de manifiesto y visualización de métricas/salidas.
- [ ] **Step 4: Añadir selector documentado `backend`, `mode`, horizonte y escenarios; mostrar backend efectivo y razón de fallback.**
- [ ] **Step 5: Ejecutar el notebook con `nbconvert --execute` en modo smoke y exigir salida 0.**
- [ ] **Step 6: Commit:** `git commit -m "refactor: make notebook a thin local pipeline interface"`.

## Task 12: Simulación integral y cierre

**Files:**
- Create: `tests/acceptance/test_run_artifacts.py`
- Create: `docs/operacion-gpu-local.md`
- Outputs: `resultados/<run_id>/...`

**Interfaces:**
- Consumes: pipeline completo.
- Produces: ejecución aceptada y reproducible.

- [ ] **Step 1: Ejecutar todas las pruebas:** `.\.venv\Scripts\python.exe -m pytest -q`; resultado exigido: 0 fallos.
- [ ] **Step 2: Ejecutar smoke GPU:** `.\.venv\Scripts\python.exe -m prodma.pipeline --mode smoke --backend gpu`; XGBoost y CatBoost deben usar GPU real.
- [ ] **Step 3: Ejecutar CPU de referencia:** mismo smoke con `--backend cpu`; comparar forma, finitud y métricas, no igualdad bit a bit.
- [ ] **Step 4: Ejecutar full:** `.\.venv\Scripts\python.exe -m prodma.pipeline --mode full --backend auto`; conservar logs y checkpoints.
- [ ] **Step 5: Validar artefactos:** cinco hashes de entrada, cobertura por medidor, métricas por horizonte, modelos, pronósticos 1/7 días, escenarios 1–5 años, matriz base/total, altas pendientes y reporte DIgSILENT.
- [ ] **Step 6: Criterio de aceptación:** `manifest.status=SUCCESS`, ninguna etapa `ERROR`, todos los tests en verde, backends declarados, predicciones finitas, índices únicos y balances de energía dentro de tolerancia.
- [ ] **Step 7: Si el full falla:** conservar `FAILED` con traceback y checkpoint; corregir mediante prueba de regresión y reanudar. No editar el manifiesto manualmente.
- [ ] **Step 8: Documentar tiempos CPU/GPU, VRAM máxima y cualquier fallback; no afirmar aceleración sin benchmark medido.**
- [ ] **Step 9: Commit final:** `git commit -m "test: verify full demand forecasting simulation"`.

## Matriz de aceptación

| Área | Evidencia exigida |
|---|---|
| Entradas | 5 archivos, SHA-256, tamaño, fecha y apertura correcta |
| Esquema | P/Q/fecha válidos por hoja; descartes con causa |
| Tiempo | orden, unicidad, cobertura, huecos y convención de intervalo |
| Fuga | pruebas causales y rolling-origin aprobadas |
| GPU | microentrenamiento real, backend efectivo y VRAM registrada |
| Modelos | benchmark y candidatos evaluados con los mismos cortes |
| Operación | 96 y 672 pasos por serie elegible |
| Planeamiento | escenarios 1–5 años separados del pronóstico ML |
| Nuevas cargas | medidor/fecha exactos y balance kW–MWh |
| Repetibilidad | configuración, versiones, semillas, hashes y manifiesto |
| Cierre | pytest=0, pipeline=0 y `manifest.status=SUCCESS` |

## Sustento técnico y científico

- La validación temporal y fuera de muestra sigue a Cerqueira, Torgo y Mozetič, *Evaluating time series forecasting models* ([arXiv 1905.11744](https://arxiv.org/abs/1905.11744)).
- La separación entre notebook narrativo, dependencias y ejecución reproducible se apoya en Pimentel et al., *Understanding and improving the quality and reproducibility of Jupyter notebooks* ([DOI 10.1007/s10664-021-09961-9](https://doi.org/10.1007/s10664-021-09961-9)) y en la tesis de maestría de Brown ([DOI 10.62791/20195](https://doi.org/10.62791/20195)).
- El tratamiento del pronóstico de carga se apoya en la tesis doctoral de Hong, *Short Term Electric Load Forecasting* ([NCSU](https://repository.lib.ncsu.edu/items/c8f6d98e-3422-4a0b-9d5e-91cfeaa0de87)).
- XGBoost 3.2 habilita GPU con `device="cuda"` y `tree_method="hist"` ([documentación oficial](https://xgboost.readthedocs.io/en/release_3.2.0/gpu/)).
- CatBoost habilita GPU mediante `task_type="GPU"` y advierte que el entrenamiento GPU no es determinista bit a bit ([documentación oficial](https://catboost.ai/docs/en/features/training-on-gpu)).
- En Windows, LightGBM usa el backend OpenCL `device_type="gpu"`; su backend CUDA no está soportado en Windows ([guía oficial](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)).
