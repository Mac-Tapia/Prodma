# Cierre integral, exógenas y modelos de Prodma Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cerrar todas las brechas diagnosticadas de Prodma y entregar un flujo local, auditable, robusto y escalable que descubra y limpie todos los datos, integre clientes y variables externas con causalidad, entrene y evalúe todas las series y modelos elegibles, seleccione pronósticos reproducibles y los opere mediante API y React.

**Architecture:** Monolito modular con contratos de artefactos inmutables entre custodia, ingesta, limpieza, entrenamiento, evaluación y predicción. Los conectores descargan instantáneas crudas de fuentes oficiales; la capa de limpieza produce matrices históricas `as-of`, futuras conocidas y escenarios separados. El entrenamiento recorre todas las series autorizadas, prueba familias locales/globales y variantes con/sin exógenas mediante validación temporal anidada; la evaluación selecciona por gates deterministas y la predicción reconcilia resultados antes de publicar. FastAPI, un worker exclusivo y React consumen el mismo núcleo, sin duplicar lógica científica.

**Tech Stack:** Python 3.11, pandas, PyArrow, SciPy, statsmodels, scikit-learn, Optuna, XGBoost, LightGBM, CatBoost, PyTorch/NeuralForecast cuando sea aprobado, FastAPI, SQLite, React, TypeScript, Vite, Vitest, Testing Library y Playwright.

**Spec:** `.specify/memory/constitution.md`; `specs/000-reestructuracion-integral/roadmap.md`; `specs/001-custodia-inventario/spec.md` a `specs/010-migracion-convergencia/spec.md`. Este plan amplía `docs/superpowers/plans/2026-10-10-reestructuracion-integral-prodma.md`; no reemplaza los contratos ya aprobados.

## Global Constraints

- Tratar `datahistorica/`, `nuevosclientes/`, `variables_externas/` y `medidoralimentador.xlsx` como entradas de solo lectura; registrar SHA-256 antes y después.
- No usar listas fijas de años, hojas, medidores, modelos ni variables. Todo elemento descubierto debe tener un estado terminal y una causa trazable.
- Mantener frecuencia contractual de 15 minutos, zona `America/Lima`, unidades explícitas y fin de horizonte exclusivo.
- Separar valor observado, marca de calidad, contexto causal e imputación. Nunca usar un valor imputado como objetivo observado.
- Ninguna unión temporal puede ver datos publicados después del origen del pronóstico. Cada exógena debe portar `periodo_referencia`, `fecha_disponible`, `fecha_descarga`, fuente, versión y política futura.
- “Considerar todas las exógenas” significa extraerlas, validarlas, limpiarlas, concatenarlas en la matriz candidata y decidir su uso por prueba de ablación. No obliga a conservar una variable que degrade el pronóstico, sea redundante, carezca de cobertura o produzca fuga; el descarte debe quedar demostrado.
- Los clientes nuevos confirmados son escenario de planeamiento futuro. No se suman al objetivo histórico ni entran como pasado ficticio. Solo una serie histórica fechada de altas/clientes, si existe y supera custodia y causalidad, puede ser covariable de entrenamiento.
- Ejecutar todos los modelos registrados que sean técnicamente elegibles para cada serie; una dependencia ausente, timeout o incompatibilidad genera estado explícito, no omisión silenciosa.
- No declarar GPU por configuración: verificar `fit` y `predict` finitos, registrar backend solicitado/efectivo y permitir degradación documentada a CPU.
- Los modelos modernos adicionales son candidatos, no ganadores anticipados. Deben superar baselines simples, presupuesto de cómputo, estabilidad, calibración y reproducibilidad.
- Aplicar pruebas paramétricas solo cuando sus supuestos sean defendibles; si no, usar alternativas no paramétricas/permutación. Corregir comparaciones múltiples y reportar tamaño de efecto e intervalos, no solo valores p.
- Generar solo figuras que respondan una decisión o gate. Toda figura debe derivarse de un artefacto tabular reabrible; ninguna figura sustituye a los datos.
- Cada tarea sigue RED-GREEN-REFACTOR, termina con pruebas focalizadas y commit independiente. Ningún gate se aprueba con pruebas omitidas o artefactos no reabiertos.

## Review Focus

- Cobertura total: archivos, hojas, series, exógenas, clientes y candidatos deben reconciliarse por conteos y conjuntos.
- Causalidad: revisar particularmente PBI, población, producción regional, clima futuro y fechas de alta de clientes.
- Escala: no cargar simultáneamente todas las combinaciones serie-modelo; usar lotes, límites de recursos, checkpoints y reanudación idempotente.
- Selección: comparar todos los modelos sobre exactamente los mismos bloques; separar tuning, test final y evaluación de planeamiento extrapolativo.
- Evidencia: distinguir éxito de comando, persistencia física, reapertura efectiva y aceptación científica/operativa.

---

## Diagnóstico que este plan debe cerrar

| Hallazgo verificado | Riesgo | Cierre obligatorio |
|---|---|---|
| `config/default.yaml` fija años 2023–2026. | Archivos futuros quedan fuera. | Descubrimiento recursivo y rangos inferidos por contenido. |
| Calendario tiene feriados sin norma verificada. | Etiquetas no auditables. | Conector oficial, norma/alcance y validación de días no laborables. |
| Temperatura usa un único punto/reanálisis y climatología. | Baja representatividad espacial; no es pronóstico meteorológico futuro. | Estaciones/puntos por zona, procedencia explícita y políticas separadas de histórico, operación y planeamiento. |
| PBI queda 100 % nulo en la matriz histórica por usar la fecha de descarga 2026 como disponibilidad. | La variable se declara pero nunca entrena. | Calendario de publicación/vintages o lag conservador validado y test `as-of`. |
| PBI es nacional y producción regional está deshabilitada/ausente. | Señal económica incompleta para Ica. | BCRP/INEI regionales con granularidad, unidad y rezago documentados. |
| Población tiene pocos puntos; parte es interpolada/extrapolada. | Precisión aparente sin respaldo oficial. | Serie/proyección oficial hasta horizonte y etiqueta de estimado/proyectado. |
| `clientes_confirmados` y `produccion_regional` están deshabilitados. | Entradas sueltas y planeamiento incompleto. | Ingesta con estados explícitos; clientes como escenario, producción como covariable causal. |
| `exogenas/matrix.py` y `validation.py` conocen solo cuatro nombres. | Nuevas fuentes fallan o se omiten. | Registro genérico de conectores, esquemas y transformadores. |
| `stages/training.py` ejecuta `break` después del primer objetivo elegible. | Solo se entrena una serie. | Planificador de todas las series con aislamiento y reconciliación. |
| El registro contiene 13 modelos, pero el último run entrenó solo SeasonalNaive y Ridge. | Comparación incompleta. | Matriz serie × modelo, estados de prefiltro y presupuesto por candidato. |
| Solo Ridge y RandomForest reciben tuning real. | Comparación sesgada. | Espacios de búsqueda por familia y presupuesto comparable. |
| `modeling/ablation.py` existe, pero no está integrado al stage. | No se demuestra el aporte de exógenas. | Ablación obligatoria base vs. exógenas por horizonte/serie. |
| `04_evaluacion/` y `05_prediccion/` no están implementados. | No hay selección ni pronóstico aceptable. | Specs 005–006 completas, artefactos y gates. |
| No existen API/worker/React y el notebook conserva lógica. | Operación frágil y duplicada. | Specs 007–010 completas sobre un único orquestador. |
| `docs/evidencia_cientifica.yaml` deja ingesta, evaluación y predicción pendientes. | Decisiones sin sustento requerido. | Registro científico completo por decisión, prueba y artefacto. |

## Flujo objetivo y contrato de almacenamiento

| Capa | Ubicación | Contenido autorizado | Consumidor |
|---|---|---|---|
| Entradas originales | carpetas actuales | Excel/CSV/PDF/API sin modificar | Custodia |
| Instantáneas externas | `00_custodia/raw_external/<fuente>/<fecha_descarga>/` | Respuesta cruda, headers, URL, licencia y SHA-256 | Ingesta externa |
| Ingesta canónica | `01_ingesta/` | Filas/hojas normalizadas sin alterar valor, con linaje | Limpieza |
| Datos limpios | `02_limpieza/datos/` | Demanda limpia, calidad, exógenas históricas/futuras y escenarios | Entrenamiento/predicción |
| Estadística | `02_limpieza/estadistica_descriptiva/` | Tablas y figuras mínimas | Gates/calidad |
| Modelos | `03_entrenamiento/` | checkpoints por corrida/serie/modelo/feature-set | Evaluación |
| Evaluación | `04_evaluacion/` | predicciones OOS, métricas, inferencia, ranking, ganadores | Predicción |
| Predicción | `05_prediccion/` | operacional, planeamiento, escenarios, intervalos y entregables | API/React/PowerFactory |
| Control | SQLite local | jobs, eventos, locks y punteros; nunca resultados científicos | API/worker |

Todo catálogo de exógenas debe asignar uno de estos estados: `USED`, `REJECTED_WITH_EVIDENCE`, `SCENARIO_ONLY`, `BLOCKED_MISSING_SOURCE`, `BLOCKED_INVALID`, `SUPERSEDED`. No existe el estado implícito “deshabilitado y olvidado”.

## Figuras estrictamente necesarias

1. **Cobertura y faltantes:** mapa compacto fuente/variable × tiempo con porcentaje y disponibilidad `as-of`.
2. **Serie y calidad:** una plantilla por serie o grupo, con observados, faltantes, imputados, outliers y cambios de nivel; se pagina, no se generan variantes redundantes.
3. **Dependencia temporal:** ACF/PACF o periodograma solo cuando determine lags/estacionalidad; no ambos por rutina.
4. **Aprendizaje/generalización:** error de entrenamiento/validación por tamaño o época solo para modelos que lo produzcan; incluye brecha de sobreajuste.
5. **Comparación OOS:** distribución de error pareado por horizonte y diagrama de rangos/critical difference cuando haya suficientes series.
6. **Calibración probabilística:** cobertura nominal vs. empírica y ancho de intervalo por horizonte.
7. **Pronóstico final:** historia reciente, punto/intervalo y escenarios base/clientes/exógenas; agregados coherentes.

No se incluyen histogramas, cajas, correlaciones o dashboards duplicados si la tabla correspondiente ya resuelve la decisión.

## Fundamento científico y metodológico

| Decisión | Evidencia que debe registrarse en `docs/evidencia_cientifica.yaml` |
|---|---|
| Pipeline sistemático de carga eléctrica y variables meteorológicas/calendario | Hong, tesis doctoral (2010), *Short Term Electric Load Forecasting*, NCSU: https://repository.lib.ncsu.edu/items/c8f6d98e-3422-4a0b-9d5e-91cfeaa0de87 |
| Contexto peruano multinodal y preprocesamiento | Jesusi Huamani, tesis UNI (2025), *Análisis y formulación de un modelo de machine learning...*: https://cybertesis.uni.edu.pe/handle/20.500.14076/28764 |
| Variables socioeconómicas | Gastaldello, tesis doctoral USP (2017), demanda residencial con aspectos socioeconómicos: https://doi.org/10.11606/T.3.2017.tde-04082017-151342 |
| Validación temporal fuera de muestra | Tashman (2000): https://doi.org/10.1016/S0169-2070(00)00065-0; Cerqueira et al. (2020): https://doi.org/10.1007/s10994-020-05867-9 |
| Pronóstico probabilístico | Hong y Fan (2016): https://doi.org/10.1016/j.ijforecast.2015.11.011; Gneiting y Raftery (2007): https://doi.org/10.1198/016214506000001437 |
| Comparación estadística de múltiples modelos | Demšar (2006): https://www.jmlr.org/papers/v7/demsar06a.html; Hansen et al., Model Confidence Set: https://doi.org/10.3982/ECTA5771 |
| Modelos globales/probabilísticos y covariables | DeepAR: https://doi.org/10.1016/j.ijforecast.2019.07.001; TFT: https://doi.org/10.1016/j.ijforecast.2021.03.012 |
| Largo horizonte eficiente | N-HiTS: https://doi.org/10.1609/aaai.v37i6.25854 |
| Coherencia alimentador/medidor | MinT: https://doi.org/10.1080/01621459.2018.1448825 |
| Fuentes abiertas oficiales | BCRP API: https://estadisticas.bcrp.gob.pe/estadisticas/series/ayuda/api; INEI población: https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib1723/libro.pdf; NASA POWER: https://power.larc.nasa.gov/docs/services/api/temporal/hourly/ |

La literatura justifica qué métodos ensayar; no autoriza declarar que un modelo es mejor sin evidencia OOS local.

---

### Task 1: Congelar alcance, ampliar la Spec transversal y automatizar cobertura

**Files:**
- Create: `specs/011-exogenas-clientes-planeamiento/spec.md`
- Create: `specs/011-exogenas-clientes-planeamiento/plan.md`
- Create: `specs/011-exogenas-clientes-planeamiento/tasks.md`
- Modify: `specs/000-reestructuracion-integral/roadmap.md`
- Modify: `docs/evidencia_cientifica.yaml`
- Test: `tests/unit/test_spec_coverage.py`

- [ ] **Step 1: Write failing tests** that discover requirement IDs dynamically, reject duplicates, require every diagnostic row to map to a requirement/task/test/evidence, and require article + master's/doctoral thesis evidence for every scientific module.
- [ ] **Step 2: Run** `\.venv\Scripts\python.exe -m pytest tests/unit/test_spec_coverage.py -q`; expect failure because Spec 011 and dynamic coverage do not exist.
- [ ] **Step 3: Implement** S011 requirements for source lifecycle, terminal statuses, client scenarios, `as-of`, ablation, model-feature compatibility, probabilistic outputs and mandatory reconciliation. Remove the hard-coded “82” assumption from convergence documentation.
- [ ] **Step 4: Run** the focused test and validate every ID is unique and mapped.
- [ ] **Step 5: Commit** `git add specs docs/evidencia_cientifica.yaml tests/unit/test_spec_coverage.py && git commit -m "docs: specify complete exogenous planning lifecycle"`.

### Task 2: Implementar custodia e inventario completos (Specs 001 y 002)

**Files:**
- Create: `src/prodma/custody/{contracts,inventory,authorized,stage}.py`
- Create: `src/prodma/ingestion/{contracts,schema,excel,stage}.py`
- Modify: `src/prodma/stages/ingestion.py`
- Test: `tests/unit/test_input_inventory.py`
- Test: `tests/unit/test_authorized_meter_catalog.py`
- Test: `tests/integration/test_ingestion_all_sheets.py`

- [ ] **Step 1: Write failing tests** from plans 001/002 plus discovery of later years, root workbook, nested external files, unknown extensions, changed bytes, every sheet, blank/malformed/contradictory records and exact source coordinates.
- [ ] **Step 2: Run** the three focused test files; expect missing v2 modules/contracts.
- [ ] **Step 3: Implement** streaming SHA-256, read-only discovery and canonical rows without fixed years. Parse `medidoralimentador.xlsx` sheet `alimentador_medidor`, header `Medidor`, preserving exact identity and duplicate provenance.
- [ ] **Step 4: Publish/reopen** `00_custodia/inventario.jsonl`, `catalogo_medidores.parquet`, `01_ingesta/**/*.parquet` and reconciliation reports; verify source hashes/mtimes unchanged.
- [ ] **Step 5: Commit** `git add src/prodma/custody src/prodma/ingestion src/prodma/stages/ingestion.py tests && git commit -m "feat: inventory and ingest every declared input"`.

### Task 3: Crear el registro genérico y conectores reproducibles de fuentes externas

**Files:**
- Create: `src/prodma/external/{contracts,registry,http,snapshots,stage}.py`
- Modify: `src/prodma/exogenas/catalog.py`
- Modify: `src/prodma/exogenas/validation.py`
- Modify: `config/default.yaml`
- Test: `tests/unit/test_external_registry.py`
- Test: `tests/integration/test_external_snapshots.py`

- [ ] **Step 1: Write failing tests** for plugin registration, schema/version/licence metadata, retry/backoff, timeout, HTTP cache validators, offline replay, content hash, partial download rejection and unsupported enabled source.
- [ ] **Step 2: Run** focused tests; expect failure because validation is name-hardcoded.
- [ ] **Step 3: Implement** `ExternalSourceSpec`, `RawSnapshot`, `SourceConnector.fetch(request)->snapshot` and `SourceTransformer.transform(snapshot)->CanonicalExternalFrame`. Network retrieval writes first to a temporary file and atomically promotes only verified responses.
- [ ] **Step 4: Make** every configured source reach a terminal status; retain current `prodma.exogenas` APIs as adapters until parity tests pass.
- [ ] **Step 5: Commit** `git add src/prodma/external src/prodma/exogenas config tests && git commit -m "feat: add auditable external source registry"`.

### Task 4: Integrar calendario peruano verificable

**Files:**
- Create: `src/prodma/external/connectors/calendar_pe.py`
- Create: `tests/fixtures/external/calendar_pe/`
- Modify: `config/default.yaml`
- Test: `tests/unit/test_calendar_connector.py`

- [ ] **Step 1: Write failing tests** for official holiday date, name, legal basis, scope, bridge/non-working day, weekend, leap day, duplicate date and horizon extension.
- [ ] **Step 2: Run** focused tests; expect missing connector.
- [ ] **Step 3: Implement** extraction with source URL and legal metadata; generate only deterministic derived features (`dow`, `month`, cyclical terms, holiday distance) after preserving the official raw event.
- [ ] **Step 4: Verify** 100% calendar coverage for historical and requested future horizon; fail closed on an unverified holiday rather than filling `norma` silently.
- [ ] **Step 5: Commit** `git add src/prodma/external/connectors/calendar_pe.py config tests && git commit -m "feat: verify Peruvian calendar features"`.

### Task 5: Integrar clima histórico, operativo y de planeamiento

**Files:**
- Create: `src/prodma/external/connectors/{senamhi,nasa_power,open_meteo}.py`
- Create: `src/prodma/external/weather_locations.py`
- Modify: `src/prodma/exogenas/weather.py`
- Test: `tests/unit/test_weather_source_policy.py`
- Test: `tests/integration/test_weather_multi_location.py`

- [ ] **Step 1: Write failing tests** for UTC/local conversion, DST-free Lima alignment, missing sentinel values, units, multiple locations, source priority, issue time, forecast horizon, climatology cutoff and no future leakage.
- [ ] **Step 2: Run** tests; expect failure because only one Ica-center file exists.
- [ ] **Step 3: Implement** priority `SENAMHI observed/forecast if legally and technically accessible -> NASA POWER/Open-Meteo historical fallback`; map locations to alimentadores/medidores with explicit distance and elevation.
- [ ] **Step 4: Keep** operational forecasts, reanalysis and planning climatology in different columns/statuses. Never label interpolated hourly reanalysis as direct measurement or future forecast.
- [ ] **Step 5: Commit** `git add src/prodma/external src/prodma/exogenas/weather.py tests && git commit -m "feat: integrate causal multi-location weather"`.

### Task 6: Integrar economía BCRP con vintages y rezagos reales

**Files:**
- Create: `src/prodma/external/connectors/bcrp.py`
- Create: `src/prodma/external/vintages.py`
- Modify: `src/prodma/exogenas/asof.py`
- Test: `tests/unit/test_bcrp_connector.py`
- Test: `tests/unit/test_economic_asof.py`

- [ ] **Step 1: Write failing tests** for API metadata, monthly period parsing, units/base year, missing value, revision, publication calendar, conservative lag fallback and the historical PBI coverage regression.
- [ ] **Step 2: Run** focused tests; reproduce that assigning 2026 download dates makes 2023–2025 PBI unavailable.
- [ ] **Step 3: Implement** separate `periodo_referencia`, `fecha_publicacion` and `fecha_descarga`. Prefer true releases/vintages; where unavailable, declare and test a conservative lag policy, never backdate the download.
- [ ] **Step 4: Add** national and available regional economic series to the candidate catalog with collinearity and revision flags; require nonzero `as-of` coverage or terminal `BLOCKED_*` status.
- [ ] **Step 5: Commit** `git add src/prodma/external src/prodma/exogenas/asof.py tests && git commit -m "fix: make economic features historically available as of origin"`.

### Task 7: Integrar población y actividad productiva regional INEI

**Files:**
- Create: `src/prodma/external/connectors/inei.py`
- Create: `src/prodma/external/low_frequency.py`
- Modify: `config/default.yaml`
- Test: `tests/unit/test_inei_connector.py`
- Test: `tests/unit/test_low_frequency_features.py`

- [ ] **Step 1: Write failing tests** for Ica scope, table identity, annual/monthly frequency, unit/base year, official/provisional/projected labels, release dates, revisions and horizon coverage.
- [ ] **Step 2: Run** tests; expect absent regional production and sparse population failures.
- [ ] **Step 3: Implement** official population projections and departmental production indicators. Preserve native low frequency; generate step, growth and lagged variants only inside folds and never pretend 15-minute measurement precision.
- [ ] **Step 4: Reject** ad-hoc extrapolation when an official projection exists. If the official horizon ends, expose `scenario_assumption` with sensitivity bounds, not an observed feature.
- [ ] **Step 5: Commit** `git add src/prodma/external config tests && git commit -m "feat: add official Ica socioeconomic features"`.

### Task 8: Integrar clientes históricos y nuevos clientes sin contaminar el objetivo

**Files:**
- Modify: `src/prodma/exogenas/new_customers.py`
- Create: `src/prodma/customers/{contracts,ingestion,scenarios}.py`
- Modify: `config/default.yaml`
- Test: `tests/unit/test_customer_contracts.py`
- Test: `tests/integration/test_new_customers_scenario.py`

- [ ] **Step 1: Write failing tests** for Excel file discovery, all sheets, feeder/meter mapping, connection date, requested kW vs kVA, power factor, coincidence/diversity, ramp, duplicate customer, unknown feeder and missing date.
- [ ] **Step 2: Run** focused tests; expect current scenario parser to lack the full contract.
- [ ] **Step 3: Implement** two separate datasets: `clientes_historicos.parquet` for timestamped realized connections and `escenarios_clientes.parquet` for future confirmed/plausible additions. Store original row/cell and assumptions.
- [ ] **Step 4: Enforce**: realized history can enter training only after its availability date; future clients enter prediction after base-model inference and before aggregation/reconciliation. Publish base, low and high scenarios separately.
- [ ] **Step 5: Commit** `git add src/prodma/customers src/prodma/exogenas/new_customers.py config tests && git commit -m "feat: separate historical and future customer effects"`.

### Task 9: Construir matrices exógenas genéricas, limpias y reconciliadas

**Files:**
- Modify: `src/prodma/exogenas/matrix.py`
- Modify: `src/prodma/exogenas/contracts.py`
- Create: `src/prodma/external/quality.py`
- Test: `tests/unit/test_exogenous_matrix_v2.py`
- Test: `tests/integration/test_exogenous_reconciliation.py`

- [ ] **Step 1: Write failing tests** for dynamic sources, aligned index, duplicates, missing blocks, nonfinite values, units, resampling policy, vintage join, future policy, scenario isolation and catalog-to-column reconciliation.
- [ ] **Step 2: Run** tests; expect hardcoded-source failures.
- [ ] **Step 3: Implement** `matriz_exogenas_historicas.parquet`, `matriz_exogenas_futuras.parquet`, `matriz_escenarios.parquet`, `calidad_exogenas.parquet` and `catalogo_exogenas.json`. Every source/column records raw/clean hashes and transformation chain.
- [ ] **Step 4: Reopen and assert** every catalog variable is present or has a terminal exclusion; no all-null column is silently dropped.
- [ ] **Step 5: Commit** `git add src/prodma/exogenas src/prodma/external tests && git commit -m "feat: reconcile all exogenous matrices"`.

### Task 10: Completar limpieza masiva, descriptivos e inferencia previa (Spec 003)

**Files:**
- Create: `src/prodma/cleaning/{contracts,temporal,quality,descriptive,stage}.py`
- Modify: `src/prodma/stages/cleaning.py`
- Modify: `src/prodma/limpieza/descriptive.py`
- Test: `tests/unit/test_clean_variable_contract_v2.py`
- Test: `tests/unit/test_cleaning_statistics_v2.py`
- Test: `tests/integration/test_mass_cleaning_stage.py`

- [ ] **Step 1: Write failing tests** from plan 003 plus mean/median/quantiles/IQR/MAD/CV/skew/kurtosis, missing runs, robust outliers, changepoints, stationarity, seasonal differences and future-perturbation invariance.
- [ ] **Step 2: Run** tests; expect missing long-form v2 contract and stage reconciliation.
- [ ] **Step 3: Implement** causal cleaning and diagnostics. Use Shapiro only for small/appropriate samples, complement with Q-Q/robust summaries; use ADF/KPSS as complementary diagnostics; PELT only with penalty/sensitivity evidence. Diagnostics inform transformation, never delete data automatically.
- [ ] **Step 4: Generate only** coverage/quality timeline and conditional dependence figures listed above. Publish tabular statistics for all series, including all-null/insufficient cases.
- [ ] **Step 5: Reopen/hash/reconcile** `matriz_limpia.parquet`, quality, descriptives and exogenous matrices against all ingested series.
- [ ] **Step 6: Commit** `git add src/prodma/cleaning src/prodma/stages/cleaning.py src/prodma/limpieza tests && git commit -m "feat: clean and diagnose every series causally"`.

### Task 11: Corregir selección de objetivos y entrenar todas las series autorizadas

**Files:**
- Create: `src/prodma/training/{contracts,selection,scheduler,stage}.py`
- Modify: `src/prodma/stages/training.py`
- Test: `tests/unit/test_training_selection_v2.py`
- Test: `tests/integration/test_training_all_targets.py`

- [ ] **Step 1: Write failing regression test** with at least three eligible targets and assert all appear in the training manifest; it must fail due to the current `break`.
- [ ] **Step 2: Implement** intersection `clean targets ∩ authorized meters ∩ sufficient history ∩ finite quality`, with exact exclusion reasons. Remove the one-target break.
- [ ] **Step 3: Add** bounded scheduler, per-series isolation, deterministic ordering, memory budget, timeout and resume keys. Checkpoints must be nested by run/series/model/feature-set to avoid name collisions.
- [ ] **Step 4: Run** all selection/training integration tests and reconcile expected targets = trained + rejected + failed + cancelled.
- [ ] **Step 5: Commit** `git add src/prodma/training src/prodma/stages/training.py tests && git commit -m "fix: train every eligible authorized series"`.

### Task 12: Concatenar exógenas y demostrar su aporte por ablación

**Files:**
- Modify: `src/prodma/modeling/features.py`
- Modify: `src/prodma/modeling/ablation.py`
- Modify: `src/prodma/training/stage.py`
- Test: `tests/unit/test_exogenous_feature_sets.py`
- Test: `tests/integration/test_training_exogenous_ablation.py`

- [ ] **Step 1: Write failing tests** for feature sets `endogenous_only`, `calendar`, `weather`, `socioeconomic`, `all_eligible`, fold-local encoding/scaling, collinearity, missing future covariates and models without exogenous support.
- [ ] **Step 2: Run** focused tests; expect the existing unused ablation path to fail integration.
- [ ] **Step 3: Concatenate** every eligible historical exogenous column to the candidate design matrix. Route only compatible feature sets to each model; record explicit `INCOMPATIBLE_EXOG` where needed.
- [ ] **Step 4: Execute paired ablation** on identical inner folds/horizons. Retain a feature group only if it improves the declared primary score beyond tolerance or materially improves calibration without violating stability; otherwise preserve its rejection evidence.
- [ ] **Step 5: Commit** `git add src/prodma/modeling src/prodma/training tests && git commit -m "feat: prove exogenous value with temporal ablation"`.

### Task 13: Entrenar profesionalmente los 13 modelos actuales

**Files:**
- Modify: `src/prodma/modeling/registry.py`
- Modify: `src/prodma/modeling/train.py`
- Modify: `src/prodma/modeling/adapters.py`
- Modify: `src/prodma/modeling/prefilter.py`
- Test: `tests/unit/test_model_search_spaces.py`
- Test: `tests/integration/test_all_registered_models.py`

- [ ] **Step 1: Write failing parameterized tests** over Ridge, RandomForest, XGBoost, LightGBM, CatBoost, MLP, SeasonalNaive, ARIMA, SARIMA, SARIMAX, HoltWinters, Prophet and LSTM. Assert factory, resolution, exogenous capability, search space, budget and terminal status.
- [ ] **Step 2: Run** tests; expect failure because only Ridge/RandomForest receive multiple trials and past runs did not cover the registry.
- [ ] **Step 3: Define fair per-family searches** and successive-halving/Optuna budgets. Baselines remain cheap; statistical seasonal orders use bounded search; boosters tune depth/rate/regularization; MLP/LSTM tune architecture, dropout, learning rate and early stopping.
- [ ] **Step 4: Validate effective backend** with real `fit/predict`; capture seed, library, device, wall time, peak memory, convergence/early-stop trace and failure traceback.
- [ ] **Step 5: Require** all 13 candidates to finish as `TRAINED`, `PREFILTER_REJECTED`, `TIMEOUT`, `UNAVAILABLE`, `INCOMPATIBLE`, `FAILED` or `CANCELLED` for every eligible series. No absent row is allowed.
- [ ] **Step 6: Commit** `git add src/prodma/modeling tests && git commit -m "feat: fairly train all registered forecasting families"`.

### Task 14: Añadir modelos modernos globales solo bajo gate científico y operativo

**Files:**
- Create: `src/prodma/modeling/global_models.py`
- Modify: `src/prodma/modeling/registry.py`
- Modify: `requirements-gpu.txt`
- Test: `tests/unit/test_global_model_registry.py`
- Test: `tests/integration/test_global_models_smoke.py`

- [ ] **Step 1: Write failing tests** for N-HiTS, DeepAR and TFT capabilities: multi-series training, known-future vs observed-past covariates, quantiles/distributions, checkpoint reload, deterministic seed and CPU fallback.
- [ ] **Step 2: Run** tests; expect missing optional adapters. Do not install or activate dependencies implicitly.
- [ ] **Step 3: Implement optional adapters** behind extras. Use global models only when a minimum count/history gate is met; otherwise emit `INSUFFICIENT_GLOBAL_DATA`.
- [ ] **Step 4: Compare** against SeasonalNaive, Ridge and the best current local model under the same origins. Reject any modern model that cannot beat baselines, calibrate intervals, fit resource limits or replay from checkpoint.
- [ ] **Step 5: Commit** `git add src/prodma/modeling requirements-gpu.txt tests && git commit -m "feat: gate modern global forecasting candidates"`.

### Task 15: Completar backtesting, métricas e inferencia (Spec 005)

**Files:**
- Create: `src/prodma/evaluation/{contracts,backtesting,metrics,diagnostics,inference,ranking,stage}.py`
- Modify: `src/prodma/evaluacion/backtesting.py`
- Test: `tests/unit/test_evaluation_metrics.py`
- Test: `tests/unit/test_evaluation_inference.py`
- Test: `tests/integration/test_evaluation_stage.py`

- [ ] **Step 1: Write failing tests** for rolling origin at 96/672/2688 steps (or declared horizons), common blocks, MAE/RMSE/MASE/RMSSE/sMAPE, energy bias, peak magnitude/timing, pinball loss, interval score, coverage and zero denominators.
- [ ] **Step 2: Implement residual diagnostics** for bias, autocorrelation (Ljung–Box), heteroscedasticity and calibration. Treat normality as descriptive; forecast errors rarely justify selecting solely by Gaussian assumptions.
- [ ] **Step 3: Implement routed inference**: paired t/ANOVA only when assumptions and dependence treatment pass; Wilcoxon/permutation for two candidates; Friedman + Holm/Nemenyi for multiple models across series; Diebold–Mariano only on aligned loss differentials with horizon-aware variance. Report effect size and corrected confidence.
- [ ] **Step 4: Implement deterministic gates/ranking** using accuracy, stability, calibration, resources and simplicity. Add Model Confidence Set where sample conditions are met; otherwise mark `INSUFFICIENT_INFERENCE`.
- [ ] **Step 5: Publish/reopen** OOS predictions, fold metrics, residual tests, pairwise tests, ranking, exclusions and winners. Generate only OOS error/rank and calibration figures.
- [ ] **Step 6: Commit** `git add src/prodma/evaluation src/prodma/evaluacion tests && git commit -m "feat: evaluate and select forecasts out of sample"`.

### Task 16: Completar predicción, escenarios, reconciliación y entregables (Spec 006)

**Files:**
- Create: `src/prodma/prediction/{contracts,eligibility,operational,planning,reconciliation,stage}.py`
- Create: `src/prodma/delivery/{parquet,csv_excel,powerfactory}.py`
- Test: `tests/unit/test_prediction_engines.py`
- Test: `tests/unit/test_prediction_scenarios.py`
- Test: `tests/integration/test_prediction_delivery.py`

- [ ] **Step 1: Write failing tests** for four-way eligibility, default all, exact start, exclusive end, continuous index, feature availability, base/low/high clients, macro/climate sensitivities, finite/nonnegative policy and unauthorized medidor.
- [ ] **Step 2: Implement** operational forecast only within validated horizons and planning projection with an explicit extrapolation warning beyond them. Future covariates must come from `known_future`, issued forecasts or named scenarios.
- [ ] **Step 3: Apply** new-client effects with connection/ramp/coincidence assumptions after base prediction. Reconcile medidor→alimentador totals via bottom-up and benchmark MinT; retain base and reconciled values.
- [ ] **Step 4: Calibrate** conformal/empirical intervals using untouched residuals; measure conditional coverage by horizon/season and widen or block when calibration is insufficient.
- [ ] **Step 5: Export/reopen** Parquet, CSV, Excel and PowerFactory tables with units, hashes, selected meters, model/version, feature scenario and assumptions. Produce only the final forecast/interval/scenario figure.
- [ ] **Step 6: Commit** `git add src/prodma/prediction src/prodma/delivery tests && git commit -m "feat: publish coherent planning forecasts and scenarios"`.

### Task 17: Completar API, worker y control transaccional (Spec 007)

**Files:**
- Create: `src/prodma/api/{app,schemas,routes,events,security}.py`
- Create: `src/prodma/worker/{runner,locks,recovery}.py`
- Create: `src/prodma/control/{models,repository}.py`
- Test: `tests/integration/test_api_jobs.py`
- Test: `tests/integration/test_worker_recovery.py`

- [ ] **Step 1: Write failing tests** for versioned loopback API, config allowlist, traversal rejection, one-heavy-job lock, state transitions, cancel/resume, abandoned job recovery, SSE reconnection and safe downloads.
- [ ] **Step 2: Implement** SQLite transactions only for control. The worker calls the same application orchestrator as CLI and publishes `SUCCESS` only after artifact reopen/hash gates.
- [ ] **Step 3: Expose** inventory, source status, exogenous coverage, customer scenarios, eligible series, model matrix, metrics, winners, predictions and evidence—not merely progress percentages.
- [ ] **Step 4: Run** API/worker tests plus a restart smoke.
- [ ] **Step 5: Commit** `git add src/prodma/api src/prodma/worker src/prodma/control tests && git commit -m "feat: operate pipeline with local API and worker"`.

### Task 18: Completar React con decisiones visibles y figuras mínimas (Spec 008)

**Files:**
- Create: `web/` per `specs/008-frontend-react/plan.md`
- Test: `web/src/**/*.test.tsx`
- Test: `tests/e2e/test_full_workflow.py`

- [ ] **Step 1: Write failing UI tests** for typed OpenAPI client, jobs/SSE, source lifecycle, data quality, all eligible meters selected by default, deselection only, horizon, scenarios, model matrix and artifact links.
- [ ] **Step 2: Implement** module views for Specs 001–006. Every rejection/error must be visible with cause; no UI-only success state.
- [ ] **Step 3: Render** only the seven decision-oriented figure families from this plan, backed by verified table/artifact endpoints and accessible text summaries.
- [ ] **Step 4: Run** Vitest and Playwright through the real API/worker on fixtures; verify cancellation and resume.
- [ ] **Step 5: Commit** `git add web tests/e2e && git commit -m "feat: expose auditable forecasting workflow in React"`.

### Task 19: Completar contexto, memoria, migración y retiro de duplicación (Specs 009 y 010)

**Files:**
- Create/Modify: `AGENTS.md`
- Create: `.agents/skills/prodma-context/`
- Create: `.agents/skills/prodma-memory/`
- Create: `src/prodma/application/orchestrator.py`
- Modify: `src/prodma/pipeline.py`
- Modify: `notebook/proyecciondemanda.ipynb`
- Test: `tests/unit/test_agent_contracts.py`
- Test: `tests/integration/test_cli_orchestrator.py`
- Test: `tests/test_notebook_vscode.py`

- [ ] **Step 1: Write failing tests** for progressive context, memory provenance/obsolescence, one orchestrator path, legacy adapters, config migration, notebook-as-client and removal criteria.
- [ ] **Step 2: Implement** the Specs 009/010 plans. The notebook must call public APIs and contain no productive cleaning/training logic.
- [ ] **Step 3: Characterize** legacy outputs before replacement; remove an old path only after parity, import search and full tests pass. Preserve prior result directories and user files.
- [ ] **Step 4: Run** context/link checks, CLI/API/notebook parity and full import scan for duplicate productive implementations.
- [ ] **Step 5: Commit** `git add AGENTS.md .agents src/prodma/application src/prodma/pipeline.py notebook tests && git commit -m "refactor: converge every entry point on one orchestrator"`.

### Task 20: Automatizar observabilidad, deriva y reentrenamiento seguro

**Files:**
- Create: `src/prodma/monitoring/{data_quality,drift,forecast_performance,policy}.py`
- Create: `config/monitoring.yaml`
- Test: `tests/unit/test_monitoring_policy.py`
- Test: `tests/integration/test_monitoring_artifacts.py`

- [ ] **Step 1: Write failing tests** for delayed/missing source, schema drift, distribution drift, residual drift, degradation versus naive, coverage decay, alert deduplication and no automatic promotion.
- [ ] **Step 2: Implement** PSI/KS or robust distance as monitored signals, change detection on residual/loss series and rolling champion-vs-challenger performance. Thresholds require backtested calibration.
- [ ] **Step 3: Trigger** a candidate retraining job when policy fires, but require the same gates and explicit promotion record. Never overwrite the current champion in place.
- [ ] **Step 4: Publish** monitoring tables/status; figures only when an alert needs a time-localized explanation.
- [ ] **Step 5: Commit** `git add src/prodma/monitoring config/monitoring.yaml tests && git commit -m "feat: detect data and forecast drift safely"`.

### Task 21: Ejecutar convergencia completa y aceptación sobre datos reales

**Files:**
- Create: `tools/verify.ps1`
- Create: `tests/acceptance/test_real_run_contract.py`
- Create: `docs/acceptance/README.md`
- Modify: requirement-evidence matrix defined by Spec 010

- [ ] **Step 1: Run static gates**: formatting/type/lint as configured, unit, integration, UI and E2E tests. Missing tools/dependencies are `BLOCKED`, never pass.
- [ ] **Step 2: Run fixture smoke** end-to-end and verify exact artifact schemas, hashes, model reload/prediction replay and API/React visibility.
- [ ] **Step 3: Run real controlled pipeline** on copies/read-only originals: discover all current files, process all sheets/series, retrieve or replay pinned external snapshots, train the complete eligible matrix, evaluate and forecast the selected authorized set.
- [ ] **Step 4: Reopen every deliverable** and reconcile: files, rows, sheets, medidores, series, external variables, clients, models, failures, winners and forecast timestamps. Verify original hashes/mtimes unchanged.
- [ ] **Step 5: Generate final matrix** `requirement -> task -> code -> test -> real artifact -> hash -> status`. A requirement may be `VERIFIED` or `BLOCKED` with evidence; nothing is omitted or marked verified by UI/copy evidence alone.
- [ ] **Step 6: Record limitations**: source/licence gaps, unavailable vintages, extrapolative horizons, insufficient histories, disabled optional backends and unresolved blocks.
- [ ] **Step 7: Commit** `git add tools/verify.ps1 tests/acceptance docs/acceptance && git commit -m "test: prove end to end Prodma convergence"`.

---

## Gates de liberación y orden de ejecución

- [ ] **G0 — Spec:** Specs 001–011 y matriz diagnóstica sin requisitos huérfanos.
- [ ] **G1 — Custodia:** todos los archivos/hojas/fuentes catalogados; originales intactos.
- [ ] **G2 — Datos:** demanda y exógenas limpias, causales, finitas, reabiertas y reconciliadas; clientes separados correctamente.
- [ ] **G3 — Entrenamiento:** todas las series y los 13 modelos actuales con estado terminal; ablación y backend real demostrados.
- [ ] **G4 — Modernización:** candidatos globales opcionales solo si superan suficiencia, baseline, recursos y replay.
- [ ] **G5 — Evaluación:** mismos bloques OOS, métricas/intervalos/inferencia reproducibles y ranking sin omisiones.
- [ ] **G6 — Planeamiento:** proyecciones, clientes, escenarios e intervalos coherentes; entregables reabiertos.
- [ ] **G7 — Operación:** API/worker/React usan el mismo núcleo; cancelación, reanudación y seguridad local verificadas.
- [ ] **G8 — Convergencia:** suite completa + smoke real + matriz requisito-evidencia; limitaciones explícitas.

No avanzar al gate siguiente si el anterior falla. Los módulos independientes pueden desarrollarse en paralelo, pero sus artefactos no se integran hasta aprobar su gate contractual.

## Definition of Done

- Todos los requisitos de Specs 001–011 tienen prueba y evidencia o bloqueo explícito.
- Todo archivo, hoja, serie, exógena, cliente y modelo descubierto aparece en una reconciliación.
- No quedan fuentes `enabled: false` sin decisión terminal documentada.
- El entrenamiento no se detiene en la primera serie y no omite modelos silenciosamente.
- Las exógenas se almacenan en capas raw/canónica/limpia, entran a matrices candidatas y su uso final queda sustentado por ablación causal.
- Los nuevos clientes se reflejan en escenarios de planeamiento sin contaminar historia ni test.
- La selección usa backtesting temporal común, métricas apropiadas, incertidumbre e inferencia con supuestos comprobados.
- Los entregables se pueden reabrir, reproducir y trazar hasta originales, código, configuración, fuentes web y modelos.
- Solo se publican las figuras necesarias y todas tienen tabla fuente verificable.
- Suite completa, E2E y corrida real aceptados; Git conserva únicamente cambios intencionales.

