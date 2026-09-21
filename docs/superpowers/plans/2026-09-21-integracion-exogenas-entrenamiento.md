# Integracion Exogena y Entrenamiento Robusto - Plan de Implementacion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrar variables externas con semantica temporal auditable desde la limpieza y construir un modulo 3 que prefiltro, entrene y guarde solamente modelos reproducibles y libres de fuga.

**Architecture:** La etapa 2 publicara dos contratos separados: demanda limpia/calidad y exogenas historicas/futuras con procedencia, fecha de disponibilidad y escenario. La etapa 3 consumira hashes de ambos contratos, adaptara cada familia de modelo a una interfaz comun, reservara el test final, realizara seleccion interna rolling-origin y escribira checkpoints atomicos. La etapa 4 sera la unica autorizada para abrir el test final.

**Tech Stack:** Python 3.11, pandas, NumPy, PyArrow, scikit-learn, statsmodels, XGBoost, LightGBM CPU, CatBoost, PyTorch, Prophet, Optuna, joblib, psutil, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-pipeline-modular-demanda-design.md`

## Global Constraints

- Los Excel originales y CSV fuente son de solo lectura; cada entrada se identifica mediante SHA-256.
- Frecuencia canonica: 15 minutos; zona horaria: `America/Lima`; convención de timestamp: fin de intervalo.
- Ningun ajuste, imputacion, escalado, seleccion de variables o hiperparametros puede ver validacion externa ni test final.
- Valores imputados pueden entrar como contexto, nunca como objetivo observado ni como verdad para metricas.
- Medidores recientes permanecen aislados y no reciben historia anterior a su primera medicion.
- Atipicos reales se conservan en la capa observada; cualquier vista recortada debe quedar identificada.
- Toda exogena conserva `fuente`, `fecha_descarga`, `fecha_disponible`, `calidad`, `es_observada`, `es_proyectada` y `escenario`.
- LightGBM usa CPU mientras el binario Windows instalado no demuestre OpenCL mediante fit/predict real.
- Un backend GPU se registra como efectivo solo despues de un fit/predict real y evidencia del dispositivo.
- Cada decision aprobada se incorpora a `docs/evidencia_cientifica.yaml` con articulo indexado, tesis de maestria, tesis doctoral y al menos dos idiomas.

## Review Focus

- PBI publicado con retraso: una fila no puede usar una cifra cuya `fecha_disponible` sea posterior al origen; Task 3 lo prueba.
- Temperatura futura desconocida: entrenamiento y backtest deben usar la misma representacion disponible en operacion; Task 4 lo prueba.
- Calendario incompleto o feriado trasladado: debe fallar la validacion o quedar advertencia versionada, nunca convertirse silenciosamente en dia ordinario; Task 2 lo prueba.
- Medidor reciente o hueco largo: ninguna union exogena puede expandir su ventana operativa; Task 5 lo prueba.
- Modelos sklearn y forecasters tienen firmas incompatibles: el adaptador comun debe probar ambas familias; Task 7 lo prueba.

---

### Task 1: Catalogo tipado de variables externas

**Files:**
- Create: `src/prodma/exogenas/contracts.py`
- Create: `src/prodma/exogenas/catalog.py`
- Create: `tests/unit/test_exogenous_catalog.py`
- Modify: `config/default.yaml`

**Interfaces:**
- Consumes: archivos bajo `variables_externas/` y configuracion `exogenous.sources`.
- Produces: `ExogenousSource`, `ExogenousCatalog`, `build_exogenous_catalog(config) -> ExogenousCatalog`.

- [ ] **Step 1: Write the failing contract test**

```python
def test_every_exogenous_source_declares_availability_and_future_policy():
    catalog = build_exogenous_catalog(load_config("config/default.yaml"))
    assert {s.name for s in catalog.sources} >= {"calendario_pe", "temperatura_ica", "pbi_bcrp", "poblacion_ica"}
    assert all(s.availability in {"known_future", "publication_lag", "historical_only", "scenario"} for s in catalog.sources)
    assert all(s.future_policy in {"calendar", "climatology", "last_known", "official_projection", "scenario_only"} for s in catalog.sources)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/unit/test_exogenous_catalog.py -q`

Expected: FAIL importing `prodma.exogenas.contracts`.

- [ ] **Step 3: Implement immutable contracts**

```python
@dataclass(frozen=True)
class ExogenousSource:
    name: str
    path: Path
    frequency: str
    value_columns: tuple[str, ...]
    availability: str
    future_policy: str
    source_url: str
    sha256: str
```

Add explicit YAML records for calendar, temperature, PBI and population. Mark customer additions and regional production as `scenario`/disabled until their verified files exist.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest tests/unit/test_exogenous_catalog.py -q`

```powershell
git add config/default.yaml src/prodma/exogenas tests/unit/test_exogenous_catalog.py
git commit -m "feat: add typed exogenous source catalog"
```

### Task 2: Validate provenance, schema, coverage and calendar

**Files:**
- Create: `src/prodma/exogenas/validation.py`
- Create: `tests/unit/test_exogenous_validation.py`
- Modify: `src/prodma/stages/cleaning.py`

**Interfaces:**
- Consumes: `ExogenousCatalog`, canonical demand index.
- Produces: `validate_exogenous_sources(catalog, index) -> ExogenousValidationReport` and `02_limpieza/auditoria/exogenas_validacion.json`.

- [ ] **Step 1: Write failing tests for duplicates, timezone and coverage**

```python
def test_duplicate_exogenous_timestamp_is_rejected():
    frame = pd.DataFrame({"fecha_hora": ["2026-01-01 00:15"] * 2, "temperatura_c": [20, 21]})
    with pytest.raises(ValueError, match="timestamp duplicado"):
        validate_frame(frame, expected_frequency="15min", timezone="America/Lima")

def test_unverified_holiday_is_reported_not_silently_accepted():
    report = validate_calendar(pd.DataFrame({"fecha": ["2026-07-28"], "feriado": [True], "norma": [""]}))
    assert report.warnings == ("feriado_sin_norma",)
```

- [ ] **Step 2: Run RED, implement and persist report**

The validator must check required columns, finite values, units, monotonic unique timestamps, timezone, expected frequency, start/end coverage, SHA-256 and permitted quality labels. A failed required source makes Stage 2 `FAILED`; an optional source is excluded with an audited warning.

- [ ] **Step 3: Run GREEN and commit**

Run: `python -m pytest tests/unit/test_exogenous_validation.py tests/unit/test_clean_contract.py -q`

```powershell
git add src/prodma/exogenas/validation.py src/prodma/stages/cleaning.py tests/unit/test_exogenous_validation.py
git commit -m "feat: validate external data provenance and coverage"
```

### Task 3: Eliminate publication-lag leakage in PBI and other delayed sources

**Files:**
- Create: `src/prodma/exogenas/asof.py`
- Modify: `src/prodma/variables_externas.py`
- Modify: `tests/unit/test_variables_externas_economia.py`

**Interfaces:**
- Consumes: observations with `periodo`, `valor`, `fecha_disponible` and forecast origins.
- Produces: `asof_values(index, vintages, origin) -> pd.Series`.

- [ ] **Step 1: Pin the current leakage with a failing test**

```python
def test_pbi_never_backfills_the_past_with_a_future_release():
    vintages = pd.DataFrame({
        "periodo": pd.to_datetime(["2023-01-01", "2023-02-01"]),
        "valor": [100.0, 200.0],
        "fecha_disponible": pd.to_datetime(["2023-03-15", "2023-04-15"]),
    })
    result = asof_values(pd.DatetimeIndex(["2023-02-01"]), vintages, origin=pd.Timestamp("2023-02-28"))
    assert result.isna().all()
```

- [ ] **Step 2: Run RED and replace global fill**

Use `merge_asof` constrained by `fecha_disponible <= origin`. Never apply `fillna(ultimo_valor)` to dates before the first available vintage. For operational future rows, emit both `pbi_last_known` and `pbi_scenario`, with the scenario disabled unless explicitly configured.

- [ ] **Step 3: Run GREEN and commit**

Run: `python -m pytest tests/unit/test_variables_externas_economia.py -q`

```powershell
git add src/prodma/exogenas/asof.py src/prodma/variables_externas.py tests/unit/test_variables_externas_economia.py
git commit -m "fix: enforce as-of availability for delayed exogenous data"
```

### Task 4: Build leakage-safe weather representations

**Files:**
- Create: `src/prodma/exogenas/weather.py`
- Create: `tests/unit/test_weather_features.py`
- Modify: `src/prodma/variables_externas.py`

**Interfaces:**
- Consumes: historical hourly reanalysis, training index and forecast origin.
- Produces: `WeatherFeatures(observed, climatology, anomaly, available_for_forecast)` and `build_weather_features(...)`.

- [ ] **Step 1: Write failing origin-safety tests**

```python
def test_climatology_for_validation_is_fit_only_before_origin():
    weather = synthetic_weather_with_future_heatwave()
    a = build_weather_features(weather, INDEX, origin=ORIGIN)
    weather.loc[weather.fecha_hora > ORIGIN, "temperatura_c"] += 100
    b = build_weather_features(weather, INDEX, origin=ORIGIN)
    pd.testing.assert_series_equal(a.climatology.loc[:ORIGIN], b.climatology.loc[:ORIGIN])
```

- [ ] **Step 2: Implement two explicit modes**

`operational_climatology` uses climatology fit only on data available before each origin in both train and validation. `weather_scenario` accepts a separately hashed future series. Historical observed temperature remains available for explanatory analysis but cannot enter the primary operational backtest without archived weather forecasts.

- [ ] **Step 3: Run GREEN and commit**

Run: `python -m pytest tests/unit/test_weather_features.py -q`

```powershell
git add src/prodma/exogenas/weather.py src/prodma/variables_externas.py tests/unit/test_weather_features.py
git commit -m "feat: add origin-safe weather feature modes"
```

### Task 5: Publish the exogenous matrix from Stage 2

**Files:**
- Create: `src/prodma/exogenas/matrix.py`
- Create: `tests/integration/test_cleaning_exogenous_stage.py`
- Modify: `src/prodma/stages/cleaning.py`
- Modify: `src/prodma/artifacts.py`

**Interfaces:**
- Consumes: validated sources, established/recent demand indices.
- Produces: `matriz_exogenas_historicas.parquet`, `matriz_exogenas_futuras.parquet`, `calidad_exogenas.parquet`, `catalogo_exogenas.json`.

- [ ] **Step 1: Write the integration contract**

```python
def test_exogenous_join_preserves_demand_window_and_provenance(cleaning_run):
    demand = pd.read_parquet(cleaning_run.matrix)
    exog = pd.read_parquet(cleaning_run.exogenous_historical)
    assert exog.index.equals(demand.index)
    assert not exog.index.duplicated().any()
    assert {"feriado", "fin_semana", "temp_climatologica_c", "pbi_last_known", "poblacion_ica"} <= set(exog)
    assert cleaning_run.exogenous_sha256
```

- [ ] **Step 2: Implement strict left alignment**

Build against the demand index; never outer-join in a way that creates demand rows. Keep categorical provenance in `calidad_exogenas.parquet`, numeric model inputs in the historical/future matrices, and source hashes in the catalog.

- [ ] **Step 3: Add descriptive exogenous QA**

Write coverage, missingness, unique values, min/max, changes, correlations by season, and availability class to `02_limpieza/estadistica_descriptiva/exogenas_resumen.csv`. Generate missingness, time-series and demand-versus-weather figures without interpreting correlation as causation.

- [ ] **Step 4: Run GREEN, full suite and commit**

Run: `python -m pytest tests/integration/test_cleaning_exogenous_stage.py -q`

Run: `python -m pytest -q`

```powershell
git add src/prodma/exogenas/matrix.py src/prodma/stages/cleaning.py src/prodma/artifacts.py tests/integration/test_cleaning_exogenous_stage.py
git commit -m "feat: publish audited exogenous matrices from cleaning"
```

### Task 5A: Ingest confirmed new customers as a scenario, not historical demand

**Files:**
- Create: `src/prodma/exogenas/new_customers.py`
- Create: `tests/integration/test_new_customers_scenario.py`
- Modify: `src/prodma/stages/ingestion.py`
- Modify: `src/prodma/stages/cleaning.py`

**Interfaces:**
- Consumes: `nuevosclientes/nuevosclientes.xlsx`, sheet `demandanueva.` with customer, kW, power factor, feeder, meter and annual additions.
- Produces: `01_ingesta/auditoria/nuevos_clientes.json` and `02_limpieza/datos/escenario_clientes_confirmados.parquet`.

- [ ] **Step 1: Write schema and conservation tests**

```python
def test_new_customer_annual_energy_is_not_added_to_history(customer_workbook, layout):
    result = ingest_new_customers(customer_workbook, layout)
    assert {"cliente", "demanda_kw", "factor_potencia", "alimentador", "medidor", "anio", "incremento_kw"} <= set(result.table)
    assert result.table.groupby("cliente").incremento_kw.sum().le(result.table.groupby("cliente").demanda_kw.first() + 1e-9).all()
    assert result.table["fecha_entrada"].isna().all(), "el libro solo informa año; no se debe inventar mes o día"
```

- [ ] **Step 2: Implement header normalization and audit ambiguities**

Resolve `aliementador` as the source spelling but publish `alimentador`. Parse annual columns 2026-2029 as increments. Validate `0 < factor_potencia <= 1`, nonnegative kW and known meter mapping. Preserve unmatched meters as `SIN_MAPEO`, not as new columns in historical demand.

- [ ] **Step 3: Publish annual scenario bounds**

Do not inject customer load into historical training. Without an approved connection date and load profile, publish annual lower/central/upper scenario totals by meter. Once dates/profiles are supplied, expansion to 15 minutes must preserve annual peak/energy assumptions and label every point `scenario`, never `observed`.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest tests/integration/test_new_customers_scenario.py -q`

```powershell
git add src/prodma/exogenas/new_customers.py src/prodma/stages/ingestion.py src/prodma/stages/cleaning.py tests/integration/test_new_customers_scenario.py
git commit -m "feat: ingest confirmed customers as audited scenarios"
```

### Task 6: Reserve final test and define nested temporal splits

**Files:**
- Create: `src/prodma/modeling/splits.py`
- Create: `tests/unit/test_nested_temporal_splits.py`
- Modify: `src/prodma/evaluacion/backtesting.py`

**Interfaces:**
- Consumes: eligible observed target index and configured horizons.
- Produces: `TrainingPartition(inner_folds, calibration_index, final_test_index)`.

- [ ] **Step 1: Write isolation tests**

```python
def test_final_test_is_disjoint_from_prefilter_selection_and_tuning():
    partition = nested_temporal_partition(INDEX, horizons=(96, 672, 2688), folds=3, min_train=2688)
    used = set().union(*(set(f.train_index) | set(f.validation_index) for f in partition.inner_folds))
    assert used.isdisjoint(partition.final_test_index)
    assert partition.inner_folds[-1].validation_index[-1] < partition.final_test_index[0]
```

- [ ] **Step 2: Implement expanding rolling origins**

Reserve the most recent contiguous 2688 observed targets as final test. Create inner folds only in the earlier region. Reject a series when any horizon lacks the configured minimum history; record `HISTORIA_INSUFICIENTE` rather than shortening the test silently.

- [ ] **Step 3: Run GREEN and commit**

Run: `python -m pytest tests/unit/test_nested_temporal_splits.py tests/unit/test_backtesting.py -q`

```powershell
git add src/prodma/modeling/splits.py src/prodma/evaluacion/backtesting.py tests/unit/test_nested_temporal_splits.py
git commit -m "feat: reserve final test with nested rolling origins"
```

### Task 7: Unify sklearn and forecasting model interfaces

**Files:**
- Create: `src/prodma/modeling/adapters.py`
- Modify: `src/prodma/modeling/registry.py`
- Modify: `src/prodma/modeling/prefilter.py`
- Create: `tests/unit/test_model_adapters.py`
- Modify: `tests/unit/test_prefilter.py`

**Interfaces:**
- Produces: `ForecastAdapter.fit(train: TrainingSlice)`, `predict(request: ForecastRequest)`, `effective_backend`.

- [ ] **Step 1: Reproduce both families in tests**

```python
@pytest.mark.parametrize("name", ["Ridge", "SeasonalNaive"])
def test_prefilter_accepts_sklearn_and_series_forecasters(name, training_slice):
    result = run_prefilter({name: model_registry()[name]}, training_slice, POLICY, seed=42)
    assert result.iloc[0].estado == "APROBADO"
```

- [ ] **Step 2: Implement family-specific adapters**

Sklearn receives lagged `X/y`; statistical forecasters receive `y` and optional aligned exogenous frames; horizon-based `predict` is normalized to a timestamped Series. Resample 1h/1d models explicitly, then restore 15-minute predictions using a declared intraday profile fit only on training data.

- [ ] **Step 3: Enforce an actual timeout and resource record**

Run each candidate in an isolated process. Terminate after `prefilter_max_seconds`; record wall time, peak RSS, exception, library version, requested backend and effective backend. The final test indices are not accepted by the prefilter API.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest tests/unit/test_model_adapters.py tests/unit/test_prefilter.py -q`

```powershell
git add src/prodma/modeling/adapters.py src/prodma/modeling/registry.py src/prodma/modeling/prefilter.py tests/unit/test_model_adapters.py tests/unit/test_prefilter.py
git commit -m "fix: unify model families in temporal prefilter"
```

### Task 8: Compare endogenous-only and exogenous candidates

**Files:**
- Create: `src/prodma/modeling/ablation.py`
- Create: `tests/unit/test_exogenous_ablation.py`
- Modify: `src/prodma/modeling/features.py`

**Interfaces:**
- Produces: `compare_feature_sets(base, exogenous, folds) -> AblationResult`.

- [ ] **Step 1: Write the no-automatic-benefit test**

```python
def test_exogenous_set_is_rejected_when_it_does_not_improve_rolling_error(noise_exog_case):
    result = compare_feature_sets(noise_exog_case.base, noise_exog_case.exogenous, noise_exog_case.folds)
    assert result.selected == "endogenous_only"
    assert result.reason == "no_mejora_fuera_de_muestra"
```

- [ ] **Step 2: Implement fold-local transforms and ablation**

Fit imputers/scalers/selectors inside each inner fold. Compare `endogenous_only`, `calendar`, `calendar_weather`, and `all_available_asof` against the same origins. Require configurable practical improvement over the simpler set and stability of the sign across folds.

- [ ] **Step 3: Run GREEN and commit**

Run: `python -m pytest tests/unit/test_exogenous_ablation.py tests/unit/test_modeling_features.py -q`

```powershell
git add src/prodma/modeling/ablation.py src/prodma/modeling/features.py tests/unit/test_exogenous_ablation.py
git commit -m "feat: select exogenous features by temporal ablation"
```

### Task 9: Implement compatible atomic checkpoints

**Files:**
- Create: `src/prodma/modeling/checkpoints.py`
- Create: `tests/unit/test_checkpoints.py`

**Interfaces:**
- Produces: `CheckpointKey`, `store_checkpoint`, `load_checkpoint`.

- [ ] **Step 1: Write mismatch tests**

```python
def test_checkpoint_rejects_changed_exogenous_hash(tmp_path):
    saved = CheckpointKey("matrix-a", "exog-a", ("lag_1",), "Ridge", "1.0", "cpu", 42)
    store_checkpoint(tmp_path, saved, {"model": "sentinel"})
    with pytest.raises(CheckpointMismatch, match="exogenous_sha256"):
        load_checkpoint(tmp_path, dataclasses.replace(saved, exogenous_sha256="exog-b"))
```

- [ ] **Step 2: Implement full identity and atomic replace**

Include demand hash, exogenous hash, ordered columns, transformations, model/library version, hyperparameters, backend, seed, train end and source code version. Write to a sibling temporary file, fsync, then `replace`.

- [ ] **Step 3: Run GREEN and commit**

Run: `python -m pytest tests/unit/test_checkpoints.py -q`

```powershell
git add src/prodma/modeling/checkpoints.py tests/unit/test_checkpoints.py
git commit -m "feat: add exogenous-aware atomic checkpoints"
```

### Task 10: Train approved models with nested tuning

**Files:**
- Create: `src/prodma/modeling/train.py`
- Create: `tests/unit/test_training.py`

**Interfaces:**
- Consumes: approved candidates, `TrainingPartition`, demand/exogenous hashes.
- Produces: `TrainingResult(table, checkpoints, traces, timings)`.

- [ ] **Step 1: Write target and fold-safety tests**

```python
def test_imputed_rows_are_context_but_never_training_targets(training_case):
    result = build_training_slice(training_case.clean, training_case.features)
    assert result.y.index.equals(training_case.clean.index[training_case.clean.es_observado])
    assert not result.y.index.intersection(training_case.clean.index[training_case.clean.es_imputado]).size
```

- [ ] **Step 2: Implement tuning inside inner folds**

Optuna trials receive only inner folds. Early stopping uses the last contiguous block of each training fold. Persist trial parameters, fold errors, selected exogenous set, seeds, warnings and effective backend. Refit the selected configuration through the calibration end but never through final test.

- [ ] **Step 3: Run GREEN and commit**

Run: `python -m pytest tests/unit/test_training.py -q`

```powershell
git add src/prodma/modeling/train.py tests/unit/test_training.py
git commit -m "feat: train approved models with nested temporal tuning"
```

### Task 11: Build Stage 3 outputs and recovery

**Files:**
- Create: `src/prodma/stages/training.py`
- Create: `tests/integration/test_training_stage.py`
- Modify: `src/prodma/artifacts.py`
- Modify: `src/prodma/pipeline.py`
- Modify: `tests/unit/test_pipeline_fases.py`

**Interfaces:**
- Produces: `run_training(config, layout) -> StageResult` and all `03_entrenamiento` artifacts.

- [ ] **Step 1: Write the end-to-end smoke contract**

```python
def test_training_stage_writes_isolated_recoverable_artifacts(training_layout):
    result = run_training(training_layout.config, training_layout.layout, model_names=("SeasonalNaive", "Ridge"))
    assert result.status == "SUCCESS"
    assert (result.output / "csv/entrenamientos.csv").is_file()
    assert list((result.output / "checkpoints").glob("*.joblib"))
    assert list((result.output / "times").glob("*.csv"))
    assert list((result.output / "traces").glob("*.csv"))
```

- [ ] **Step 2: Implement isolated outputs and restart**

Write model binaries to `modelos/`, compatibility envelopes to `checkpoints/`, learning/importance plots to `figuras/`, tables to `csv/`, resources to `times/`, Optuna trials to `traces/`, and structured events to `logs/`. On restart, reuse only compatible successful checkpoints.

Expose `--run-id`, `--stage` and `--models` in `prodma.pipeline`; reject Stage 3 when Stage 2 is not `SUCCESS` or its recorded hashes differ from current artifacts.

- [ ] **Step 3: Add failure states**

The stage result distinguishes `SUCCESS`, `PARTIAL` and `FAILED`. A model failure does not stop other candidates; a contract/hash/final-test isolation failure stops the stage.

- [ ] **Step 4: Run GREEN, full suite and commit**

Run: `python -m pytest tests/integration/test_training_stage.py -q`

Run: `python -m pytest -q`

```powershell
git add src/prodma/stages/training.py src/prodma/artifacts.py src/prodma/pipeline.py tests/integration/test_training_stage.py tests/unit/test_pipeline_fases.py
git commit -m "feat: add recoverable modular training stage"
```

### Task 12: GPU smoke, scientific evidence and real-data acceptance

**Files:**
- Create: `tools/validar_backends.py`
- Create: `tests/integration/test_effective_backends.py`
- Modify: `docs/evidencia_cientifica.yaml`
- Modify: `docs/MANUAL_EJECUCION.md`

**Interfaces:**
- Produces: `03_entrenamiento/csv/backends_efectivos.csv` and an approved evidence record for module 3.

- [ ] **Step 1: Test effective backend recording**

```python
def test_backend_is_recorded_after_real_fit_predict(gpu_smoke_result):
    assert set(gpu_smoke_result.columns) >= {"modelo", "solicitado", "efectivo", "fit_ok", "predict_ok", "dispositivo"}
    assert gpu_smoke_result.loc[gpu_smoke_result.modelo == "LightGBM", "efectivo"].iloc[0] == "cpu"
```

- [ ] **Step 2: Run real backend smoke**

Execute small fit/predict jobs for XGBoost, CatBoost, LightGBM and LSTM. Record CUDA device, library versions and fallback reason. Imports and `nvidia-smi` alone do not pass.

- [ ] **Step 3: Run one real established meter through Stage 3**

Use the current Stage 2 matrix and quality masks, with `SeasonalNaive` and `Ridge` first. Verify hashes, disjoint test, nonempty predictions, finite metrics, checkpoint reload equivalence and every expected folder. Then expand the prefilter to all registered candidates.

- [ ] **Step 4: Complete scientific traceability**

Add at least these decisions to `docs/evidencia_cientifica.yaml`: rolling-origin/nested evaluation, exogenous availability, operational weather representation, model-family adapters, checkpoint identity and real backend validation. Keep module 3 pending until all linked tests and real artifacts pass.

- [ ] **Step 5: Run final gates and commit**

Run: `python -m pytest -q`

Run: `python tools/validar_backends.py --output resultados/ingesta-manual-20260920/03_entrenamiento/csv/backends_efectivos.csv`

Run: `python -m prodma.pipeline --run-id ingesta-manual-20260920 --stage 03_entrenamiento --models SeasonalNaive,Ridge`

```powershell
git add tools/validar_backends.py tests/integration/test_effective_backends.py docs/evidencia_cientifica.yaml docs/MANUAL_EJECUCION.md
git commit -m "test: validate training evidence and effective backends"
```

## Acceptance Gate Before Module 4

Module 4 may start only when all conditions are true:

1. Stage 2 publishes hashed historical/future exogenous matrices and quality/provenance artifacts.
2. PBI and every delayed source pass as-of availability tests.
3. Primary weather features are reproducible at every historical forecast origin.
4. New customers are audited as scenarios and have not contaminated historical training targets.
5. Final test timestamps are absent from prefilter, feature selection and tuning traces.
6. SeasonalNaive and Ridge complete a real end-to-end run and reload identically from checkpoint.
7. All requested candidates have explicit `APROBADO`, `RECHAZADO`, `NO_DISPONIBLE` or `TIMEOUT` status.
8. Effective CPU/GPU backend is evidenced by fit/predict.
9. `modelos`, `checkpoints`, `figuras`, `csv`, `times`, `traces` and `logs` contain current-run artifacts only.
10. Full pytest suite passes.
11. Module 3 scientific evidence status changes to `validado` only after the real-data gate succeeds.

## Scientific Basis to Verify During Execution

- Tashman (2000), out-of-sample and rolling-origin evaluation: <https://doi.org/10.1016/S0169-2070(00)00065-0>.
- Bergmeir, Hyndman and Koo (2018), validity of cross-validation for autoregressive time-series prediction: <https://doi.org/10.1016/j.csda.2017.11.003>.
- Cerqueira, Torgo and Mozetic (2020), empirical evaluation of time-series performance estimation: <https://doi.org/10.1007/s10994-020-05910-7>.
- Hong et al. (2015), weather-station selection for electric-load forecasting: <https://doi.org/10.1016/j.ijforecast.2014.07.001>.
- Hong (2010), doctoral thesis on short-term electric load forecasting: <https://repository.lib.ncsu.edu/items/c8f6d98e-3422-4a0b-9d5e-91cfeaa0de87>.
- Lotfipoor (2024), doctoral thesis on data-driven energy forecasting and missing data: <http://hdl.handle.net/10399/5167>.
- Hasanov (2022), master's thesis on smart-meter load forecasting: <https://www.researchgate.net/publication/363108666_Load_forecasting_using_machine_learning_for_smart_meter_data_processing_-_Master_Thesis_of_Magsud_Hasanov>.
- Wang, Tsai and Lin (2021), missing electric-power data imputation: <https://doi.org/10.1016/j.eswa.2021.114743>.
- Nunes et al. (2019), Portuguese-language treatment of anomalies and missing data: <https://doi.org/10.1590/1677-5449.190004>.

## Self-Review Record

- Spec coverage: stages 2 and 3, exogenous history/future, hashes, isolation, checkpoints, backends and handoff to Stage 4 are mapped to Tasks 1-12.
- Placeholder scan: no deferred implementation placeholders are used; unavailable business sources have an explicit disabled state and enabling contract.
- Type consistency: `ExogenousCatalog`, `TrainingPartition`, `ForecastAdapter`, `CheckpointKey` and `TrainingResult` are introduced before consumers.
- Review focus coverage: publication lag, future weather, calendar quality, recent-meter boundaries and model-family interfaces each have an explicit test.
- Scope boundary: inferential model comparison and final ranking remain in Module 4; this plan only preserves its untouched final test and inputs.
