# Pipeline modular de demanda electrica - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir un pipeline CLI reanudable de cinco etapas que produzca una matriz limpia unica, entrene y filtre modelos, seleccione ganadores con evaluacion temporal e inferencia, y publique la prediccion final con artefactos aislados.

**Architecture:** `prodma.pipeline` orquesta cinco funciones de etapa con contratos de entrada/salida respaldados por hashes y manifiestos atomicos. El notebook queda como cliente fino; cada etapa lee solamente artefactos contractuales de la anterior y escribe exclusivamente en `resultados/<run_id>/<etapa>/`.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, PyArrow, scikit-learn, statsmodels, Optuna, XGBoost, LightGBM, CatBoost, Prophet, PyTorch opcional, matplotlib, seaborn, PyYAML y pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-pipeline-modular-demanda-design.md`

## Global Constraints

- Los Excel originales de `datahistorica/` y `nuevosclientes/` son de solo lectura.
- La frecuencia canonica es `15min`; la zona de negocio es `America/Lima`.
- Ninguna limpieza o feature puede consultar valores posteriores al instante procesado.
- Los valores imputados pueden alimentar contexto, pero nunca objetivos ni metricas observadas.
- La etapa 2 publica la unica `matriz_limpia.parquet` admitida por las etapas 3 a 5.
- Cada etapa escribe exclusivamente bajo su carpeta del `run_id`.
- Toda seleccion de variables, transformacion y ajuste se realiza dentro del conjunto temporal permitido.
- No se publica como validado un horizonte distinto al backtest evaluado.
- No se borran resultados historicos ni cambios locales ajenos.
- Cada tarea se implementa con RED-GREEN-REFACTOR y termina con la suite completa en verde.

## Review Focus

- Dos procesos sobre el mismo `run_id`: Task 1 prueba bloqueo y escritura atomica sin corrupcion.
- Cambio de contenido Excel sin cambio de nombre: Task 2 prueba invalidacion por SHA-256.
- Hueco que cruza el corte train/test: Task 4 prueba que la imputacion de contexto no convierte el test en objetivo observado.
- Pico junto a un cambio de nivel: Task 4 prueba clasificacion causal sin `center=True` ni acceso futuro.
- Modelo/checkpoint con columnas o versiones incompatibles: Task 7 prueba rechazo explicito y reentrenamiento limpio.

---

### Task 1: Configuracion tipada, arbol de ejecucion y manifiestos atomicos

**Files:**
- Create: `config/default.yaml`
- Create: `src/prodma/config.py`
- Create: `src/prodma/artifacts.py`
- Create: `tests/unit/test_config_artifacts.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `PipelineConfig`, `load_config(path: Path) -> PipelineConfig`.
- Produces: `RunLayout.create(root: Path, run_id: str) -> RunLayout`.
- Produces: `ManifestStore.update_stage(stage: str, status: str, **data) -> None`.

- [ ] **Step 1: Add PyYAML and write failing configuration tests**

```python
def test_layout_creates_exclusive_stage_directories(tmp_path):
    layout = RunLayout.create(tmp_path, "run-001")
    assert layout.stage("03_entrenamiento", "checkpoints") == tmp_path / "run-001" / "03_entrenamiento" / "checkpoints"
    assert layout.stage("04_evaluacion", "ranking").is_dir()

def test_unknown_stage_is_rejected(tmp_path):
    layout = RunLayout.create(tmp_path, "run-001")
    with pytest.raises(ValueError, match="Etapa desconocida"):
        layout.stage("06_otro")
```

- [ ] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_config_artifacts.py -q`

Expected: FAIL importing `prodma.config` and `prodma.artifacts`.

- [ ] **Step 3: Implement the exact configuration and layout contracts**

```python
@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    output_root: Path
    frequency: str
    timezone: str
    seed: int
    horizons: tuple[int, ...]
    raw: dict[str, Any]

STAGES = ("01_ingesta", "02_limpieza", "03_entrenamiento", "04_evaluacion", "05_prediccion")

@dataclass(frozen=True)
class RunLayout:
    root: Path
    def stage(self, name: str, child: str | None = None) -> Path:
        if name not in STAGES:
            raise ValueError(f"Etapa desconocida: {name}")
        return self.root / name / child if child else self.root / name
```

Use a temporary sibling plus `Path.replace()` for every JSON manifest update. Acquire the run lock with exclusive file creation (`open("x")`) and store PID and UTC timestamp.

- [ ] **Step 4: Add atomicity and lock tests**

```python
def test_second_lock_for_same_run_fails(tmp_path):
    first = RunLock.acquire(tmp_path / "run-001")
    try:
        with pytest.raises(RuntimeError, match="ejecucion activa"):
            RunLock.acquire(tmp_path / "run-001")
    finally:
        first.release()
```

- [ ] **Step 5: Run GREEN and full regression suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_config_artifacts.py -q`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add config/default.yaml requirements.txt src/prodma/config.py src/prodma/artifacts.py tests/unit/test_config_artifacts.py
git commit -m "feat: add pipeline configuration and artifact contracts"
```

### Task 2: Etapa 1 de inventario e ingesta Excel

**Files:**
- Modify: `src/prodma/ingesta/excel.py`
- Create: `src/prodma/stages/ingestion.py`
- Create: `src/prodma/stages/__init__.py`
- Create: `tests/unit/test_ingestion_stage.py`
- Create: `tests/integration/test_ingestion_xlsx.py`

**Interfaces:**
- Consumes: `PipelineConfig`, `RunLayout`.
- Produces: `InputRecord(path, sha256, size, mtime_ns, kind, year)`.
- Produces: `run_ingestion(config, layout) -> StageResult` and `01_ingesta/catalogo.json`.

- [ ] **Step 1: Write failing inventory/hash tests**

```python
def test_changed_bytes_invalidate_same_named_input(tmp_path):
    p = tmp_path / "demanda2023.xlsx"
    p.write_bytes(b"primera-version")
    first = fingerprint_input(p)
    p.write_bytes(b"segunda-version")
    second = fingerprint_input(p)
    assert first.sha256 != second.sha256

def test_ambiguous_annual_file_is_rejected(tmp_path):
    (tmp_path / "demanda2023.xls").write_bytes(b"a")
    (tmp_path / "demanda2023.xlsx").write_bytes(b"b")
    with pytest.raises(ValueError, match="ambiguos"):
        discover_historical_files(tmp_path, years=(2023,))
```

- [ ] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_ingestion_stage.py -q`

Expected: FAIL because stage functions do not exist.

- [ ] **Step 3: Implement inventory and wire the existing parser**

```python
def fingerprint_input(path: Path) -> InputRecord:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return InputRecord(path, digest.hexdigest(), stat.st_size, stat.st_mtime_ns, "historico", extract_year(path))
```

Use `ingestar_libro()` as the only production Excel parser. Copy its outputs into `01_ingesta/parquet/` and audit JSON into `01_ingesta/auditoria/`; do not keep the notebook parser as a second path.

- [ ] **Step 4: Create synthetic XLSX integration tests**

```python
def test_stage_reads_sheet_and_preserves_raw_values(tmp_path, make_demand_xlsx):
    source = make_demand_xlsx(tmp_path / "datahistorica" / "demanda2023.xlsx")
    result = run_ingestion(config_for(tmp_path, years=(2023,)), RunLayout.create(tmp_path / "resultados", "r1"))
    frame = pd.read_parquet(result.artifacts["parquet"][0])
    assert frame.loc[pd.Timestamp("2023-01-01 00:15"), "P"] == 100.0
    assert result.status == "SUCCESS"
```

- [ ] **Step 5: Run stage tests and full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_ingestion_stage.py tests/integration/test_ingestion_xlsx.py -q`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

- [ ] **Step 6: Commit**

```powershell
git add src/prodma/ingesta src/prodma/stages tests/unit/test_ingestion_stage.py tests/integration/test_ingestion_xlsx.py
git commit -m "feat: add hashed Excel ingestion stage"
```

### Task 3: Modelo de datos limpio y consolidacion unica

**Files:**
- Create: `src/prodma/limpieza/__init__.py`
- Create: `src/prodma/limpieza/contracts.py`
- Create: `src/prodma/limpieza/temporal.py`
- Create: `src/prodma/stages/cleaning.py`
- Create: `tests/unit/test_clean_contract.py`

**Interfaces:**
- Consumes: `01_ingesta/catalogo.json` and ingested parquet files.
- Produces: `CleanVariableFrame` columns and `run_cleaning(config, layout) -> StageResult`.
- Produces: `02_limpieza/datos/matriz_limpia.parquet` plus SHA-256.

- [ ] **Step 1: Write failing observed/context contract tests**

```python
def test_imputed_context_never_becomes_observed_target():
    idx = pd.date_range("2025-01-01", periods=8, freq="15min")
    raw = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0], index=idx)
    clean = build_clean_variable(raw, forward_limit=1, weekly_lag=672)
    assert clean.loc[idx[1], "contexto"] == 1.0
    assert not clean.loc[idx[1], "es_observado"]
    assert clean.loc[idx[1], "metodo_imputacion"] == "ffill"
    assert np.isnan(clean.loc[idx[1], "objetivo"])
```

- [ ] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_clean_contract.py -q`

- [ ] **Step 3: Implement explicit clean columns**

```python
CLEAN_COLUMNS = ("observado", "contexto", "limpio", "objetivo", "es_observado",
                 "es_imputado", "metodo_imputacion", "es_atipico", "es_recortado")

def build_clean_variable(raw: pd.Series, forward_limit: int, weekly_lag: int) -> pd.DataFrame:
    observed = raw.replace([np.inf, -np.inf], np.nan).astype(float)
    forward = observed.ffill(limit=forward_limit)
    weekly = observed.shift(weekly_lag)
    context = forward.fillna(weekly)
    method = pd.Series("", index=raw.index, dtype="string")
    method.loc[observed.isna() & forward.notna()] = "ffill"
    method.loc[observed.isna() & forward.isna() & weekly.notna()] = "semana_observada"
    return assemble_clean_frame(observed, context, method)
```

- [ ] **Step 4: Prove only Stage 2 can publish the matrix**

```python
def test_matrix_path_belongs_to_cleaning_stage(tmp_path):
    layout = RunLayout.create(tmp_path, "r1")
    assert matrix_output_path(layout).parent == layout.stage("02_limpieza", "datos")
```

- [ ] **Step 5: Run GREEN and regression suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_clean_contract.py -q`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

- [ ] **Step 6: Commit**

```powershell
git add src/prodma/limpieza src/prodma/stages/cleaning.py tests/unit/test_clean_contract.py
git commit -m "feat: add clean matrix data contract"
```

### Task 4: Limpieza causal, controles electricos y descriptivos

**Files:**
- Create: `src/prodma/limpieza/outliers.py`
- Create: `src/prodma/limpieza/electrical.py`
- Create: `src/prodma/limpieza/descriptive.py`
- Modify: `src/prodma/stages/cleaning.py`
- Create: `tests/unit/test_cleaning_rules.py`
- Create: `tests/integration/test_cleaning_stage.py`

**Interfaces:**
- Produces: `detect_causal_outliers(series, window, threshold) -> OutlierResult`.
- Produces: `electrical_flags(frame, limits) -> DataFrame`.
- Produces: `descriptive_tables(matrix, quality) -> dict[str, DataFrame]`.

- [ ] **Step 1: Write the future-invariance regression test**

```python
def test_outlier_decision_does_not_change_when_future_changes():
    idx = pd.date_range("2025-01-01", periods=800, freq="15min")
    base = pd.Series(100.0, index=idx)
    base.iloc[700] = 500.0
    changed = base.copy()
    changed.iloc[701:] = 10_000.0
    a = detect_causal_outliers(base, window=672, threshold=6.0)
    b = detect_causal_outliers(changed, window=672, threshold=6.0)
    pd.testing.assert_series_equal(a.is_outlier.loc[:idx[700]], b.is_outlier.loc[:idx[700]])
```

- [ ] **Step 2: Add physical and frozen-sensor tests**

```python
def test_electrical_flags_detect_impossible_apparent_power():
    frame = pd.DataFrame({"P": [100.0], "Q": [80.0], "S_medida": [90.0]})
    flags = electrical_flags(frame, ElectricalLimits())
    assert flags.loc[0, "S_incoherente"]

def test_frozen_sensor_requires_configured_run_length():
    s = pd.Series([4.0] * 20 + [5.0])
    result = detect_frozen_sensor(s, min_run=16, tolerance=1e-9)
    assert result.iloc[:20].all() and not result.iloc[20]
```

- [ ] **Step 3: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_cleaning_rules.py -q`

- [ ] **Step 4: Implement causal Hampel-style detection without centered windows**

```python
def detect_causal_outliers(series: pd.Series, window: int = 672, threshold: float = 6.0) -> OutlierResult:
    history = series.shift(1).rolling(window, min_periods=max(96, window // 7))
    median = history.median()
    mad = (series.shift(1) - median).abs().rolling(window, min_periods=max(96, window // 7)).median()
    scale = 1.4826 * mad.replace(0, np.nan)
    candidate = (series - median).abs() > threshold * scale
    persistence = candidate.shift(1).rolling(17, min_periods=1).mean()
    isolated = candidate & persistence.lt(0.5)
    return OutlierResult(isolated.fillna(False), median, scale)
```

Classify level shifts separately from point anomalies; never use `rolling(..., center=True)`. Clip only observed values marked as correctable and retain raw values and limits in the audit table.

- [ ] **Step 5: Implement descriptive outputs and figures**

Create `resumen_por_medidor.csv`, `huecos.csv`, `calidad_electrica.csv`, `atipicos.csv`, `cambios_nivel.csv`, histograms, boxplots, raw-vs-clean curves, missingness maps and daily/weekly/monthly profiles. Tests assert exact filenames and that figures are non-empty PNGs.

- [ ] **Step 6: Run GREEN, integration and full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_cleaning_rules.py tests/integration/test_cleaning_stage.py -q`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

- [ ] **Step 7: Commit**

```powershell
git add src/prodma/limpieza src/prodma/stages/cleaning.py tests/unit/test_cleaning_rules.py tests/integration/test_cleaning_stage.py
git commit -m "feat: add causal cleaning and descriptive analysis"
```

### Task 5: Features y particiones temporales sin fuga

**Files:**
- Create: `src/prodma/modeling/__init__.py`
- Create: `src/prodma/modeling/features.py`
- Create: `src/prodma/evaluacion/__init__.py`
- Create: `src/prodma/evaluacion/backtesting.py`
- Create: `tests/unit/test_modeling_features.py`
- Create: `tests/unit/test_backtesting.py`

**Interfaces:**
- Consumes: clean variable columns from Task 3.
- Produces: `make_features(clean: DataFrame, calendar: FeatureCalendar) -> DataFrame`.
- Produces: `rolling_origins(index, horizons, folds, min_train) -> list[TemporalSplit]`.

- [ ] **Step 1: Write causal feature and gap-boundary tests**

```python
def test_future_target_change_does_not_change_past_features(clean_frame):
    altered = clean_frame.copy()
    altered.loc[altered.index[-100:], "contexto"] *= 100
    cutoff = clean_frame.index[-101]
    pd.testing.assert_frame_equal(
        make_features(clean_frame).loc[:cutoff],
        make_features(altered).loc[:cutoff],
    )

def test_imputed_row_is_never_in_validation_target(clean_frame):
    splits = rolling_origins(clean_frame.index, horizons=(96,), folds=2, min_train=672)
    for split in splits:
        assert clean_frame.loc[split.validation_index, "es_observado"].all()
```

- [ ] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_modeling_features.py tests/unit/test_backtesting.py -q`

- [ ] **Step 3: Extract notebook features into the module**

Implement calendar harmonics, weekday indicators, holidays, lags `(1, 4, 96, 672)`, shifted rolling statistics `(4, 24, 96, 672)`, four-week same-slot statistics and imputation proportion. Every dynamic variable must use `contexto.shift(...)`; target indices must use `es_observado` and finite `objetivo`.

- [ ] **Step 4: Implement horizon-matched rolling origins**

```python
@dataclass(frozen=True)
class TemporalSplit:
    train_index: pd.DatetimeIndex
    validation_index: pd.DatetimeIndex
    horizon: int

def rolling_origins(index, horizons, folds, min_train):
    index = pd.DatetimeIndex(index).sort_values().unique()
    splits = []
    for horizon in horizons:
        for fold in range(folds, 0, -1):
            stop = len(index) - horizon * (fold - 1)
            start = stop - horizon
            if start < min_train:
                raise ValueError("Historia insuficiente para rolling-origin")
            train = index[:start]
            valid = index[start:stop]
            if train[-1] >= valid[0] or len(valid) != horizon:
                raise ValueError("Particion temporal invalida")
            splits.append(TemporalSplit(train, valid, horizon))
    return splits
```

- [ ] **Step 5: Run GREEN and full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_modeling_features.py tests/unit/test_backtesting.py -q`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

- [ ] **Step 6: Commit**

```powershell
git add src/prodma/modeling src/prodma/evaluacion tests/unit/test_modeling_features.py tests/unit/test_backtesting.py
git commit -m "feat: add leakage-safe features and rolling origins"
```

### Task 6: Registro de modelos y prefiltro barato

**Files:**
- Create: `src/prodma/modeling/registry.py`
- Create: `src/prodma/modeling/prefilter.py`
- Create: `tests/unit/test_model_registry.py`
- Create: `tests/unit/test_prefilter.py`

**Interfaces:**
- Produces: `ModelSpec`, `model_registry() -> dict[str, ModelSpec]`.
- Produces: `run_prefilter(specs, dataset, splits, policy) -> pd.DataFrame`.

- [ ] **Step 1: Write registry completeness tests**

```python
def test_registry_contains_all_requested_candidates():
    expected = {"SeasonalNaive", "Ridge", "RandomForest", "XGBoost", "LightGBM",
                "CatBoost", "MLP", "ARIMA", "SARIMA", "SARIMAX", "HoltWinters",
                "Prophet", "LSTM"}
    assert expected <= set(model_registry())
```

- [ ] **Step 2: Write prefilter isolation and failure tests**

```python
def test_prefilter_never_receives_final_test_indices(dataset, prefilter_splits, final_test_index):
    table = run_prefilter(model_registry(), dataset, prefilter_splits, PrefilterPolicy(max_seconds=5))
    assert not set(table["evaluated_timestamp"]).intersection(final_test_index)

def test_non_finite_candidate_is_rejected_without_stopping_others(fake_specs, dataset):
    table = run_prefilter(fake_specs, dataset, one_split(), PrefilterPolicy(max_seconds=5))
    assert table.set_index("modelo").loc["bad", "estado"] == "RECHAZADO"
    assert table.set_index("modelo").loc["good", "estado"] == "APROBADO"
```

- [ ] **Step 3: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_model_registry.py tests/unit/test_prefilter.py -q`

- [ ] **Step 4: Implement specs and explicit prefilter states**

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[int], Any]
    resolution: str
    supports_exogenous: bool
    backend: str

PREFILTER_STATES = ("APROBADO", "RECHAZADO", "NO_DISPONIBLE")
```

The reduced backtest uses only the prefilter region, a fixed seed and small resource budgets. Record exception, RMSE versus SeasonalNaive, elapsed seconds, peak RSS and effective backend.

- [ ] **Step 5: Run GREEN and full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_model_registry.py tests/unit/test_prefilter.py -q`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

- [ ] **Step 6: Commit**

```powershell
git add src/prodma/modeling/registry.py src/prodma/modeling/prefilter.py tests/unit/test_model_registry.py tests/unit/test_prefilter.py
git commit -m "feat: add model registry and temporal prefilter"
```

### Task 7: Entrenamiento completo, checkpoints, tiempos y trazas

**Files:**
- Create: `src/prodma/modeling/checkpoints.py`
- Create: `src/prodma/modeling/train.py`
- Create: `src/prodma/stages/training.py`
- Create: `tests/unit/test_checkpoints.py`
- Create: `tests/integration/test_training_stage.py`

**Interfaces:**
- Consumes: clean matrix hash, approved `ModelSpec` rows and temporal splits.
- Produces: `CheckpointKey`, `train_approved_models(...) -> TrainingResult`.
- Produces: Task 3 artifact subfolders and `03_entrenamiento/csv/entrenamientos.csv`.

- [ ] **Step 1: Write incompatible-checkpoint tests**

```python
def test_checkpoint_rejected_when_columns_change(tmp_path):
    saved = CheckpointKey("hash-a", ("lag_1",), "Ridge", "1.0", "cpu", 42)
    store_checkpoint(tmp_path, saved, {"model": "sentinel"})
    requested = dataclasses.replace(saved, columns=("lag_1", "lag_96"))
    with pytest.raises(CheckpointMismatch, match="columns"):
        load_checkpoint(tmp_path, requested)

def test_training_records_matrix_hash(training_run):
    assert training_run.table.loc[0, "matrix_sha256"] == sha256_file(training_run.matrix_path)
```

- [ ] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoints.py tests/integration/test_training_stage.py -q`

- [ ] **Step 3: Implement checkpoint identity and isolated outputs**

```python
@dataclass(frozen=True)
class CheckpointKey:
    matrix_sha256: str
    columns: tuple[str, ...]
    model_name: str
    library_version: str
    backend: str
    seed: int
```

Write models with joblib, Optuna studies/trials to `traces/`, elapsed and resource samples to `times/`, model metrics/parameters to `csv/`, and convergence/importance plots to `figuras/`. Use a temporary file and atomic replace for checkpoints.

- [ ] **Step 4: Enforce nested temporal fitting**

Feature selection and hyperparameter tuning use inner rolling-origin splits. Early stopping uses the last contiguous block of the current training fold. The final test is unavailable to `training.py` and is read only by Stage 4.

- [ ] **Step 5: Run GREEN, training smoke and full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_checkpoints.py tests/integration/test_training_stage.py -q`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

- [ ] **Step 6: Commit**

```powershell
git add src/prodma/modeling/checkpoints.py src/prodma/modeling/train.py src/prodma/stages/training.py tests/unit/test_checkpoints.py tests/integration/test_training_stage.py
git commit -m "feat: add checkpointed approved-model training"
```

### Task 8: Metricas, diagnosticos e inferencia parametrica/no parametrica

**Files:**
- Create: `src/prodma/evaluacion/metrics.py`
- Create: `src/prodma/evaluacion/inference.py`
- Create: `src/prodma/evaluacion/ranking.py`
- Create: `tests/unit/test_evaluation_metrics.py`
- Create: `tests/unit/test_inference.py`

**Interfaces:**
- Produces: `forecast_metrics(actual, predicted, seasonal_reference) -> dict`.
- Produces: `compare_models(losses: DataFrame, alpha: float) -> InferenceResult`.
- Produces: `rank_models(metrics, inference, policy) -> pd.DataFrame`.

- [ ] **Step 1: Write metric edge-case tests**

```python
def test_metrics_do_not_report_mape_with_zero_denominator():
    result = forecast_metrics(np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    assert np.isnan(result["MAPE"])
    assert np.isfinite(result["MAE"])

def test_peak_time_error_is_reported_in_intervals():
    result = forecast_metrics(np.array([1, 4, 2]), np.array([1, 2, 5]), np.array([1, 1, 1]))
    assert result["error_hora_pico_intervalos"] == 1
```

- [ ] **Step 2: Write inferential routing tests**

```python
def test_two_models_use_wilcoxon_and_effect_size(paired_losses):
    result = compare_models(paired_losses[["A", "B"]], alpha=0.05)
    assert result.omnibus_test == "wilcoxon"
    assert "effect_size" in result.pairwise.columns

def test_three_models_use_friedman_and_adjusted_posthoc(losses_three_models):
    result = compare_models(losses_three_models, alpha=0.05)
    assert result.omnibus_test == "friedman"
    assert result.pairwise["p_adjusted"].between(0, 1).all()
```

- [ ] **Step 3: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_evaluation_metrics.py tests/unit/test_inference.py -q`

- [ ] **Step 4: Implement metrics and inference**

Use SciPy `wilcoxon` and `friedmanchisquare`, Holm correction, paired effect sizes, confidence intervals by chronological block bootstrap, and Diebold-Mariano only when the sample length exceeds the configured minimum for the horizon. Preserve test name, assumptions, statistic, raw/adjusted p-value, effect size and sample count.

- [ ] **Step 5: Implement deterministic ranking rules**

Reject non-finite, incomplete or irreproducible candidates first. Rank eligible candidates by normalized RMSE, MAE, absolute bias, peak error, stability and cost. Require configurable improvement over SeasonalNaive; ties inside tolerance choose the simpler model according to a versioned complexity order.

- [ ] **Step 6: Run GREEN and full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_evaluation_metrics.py tests/unit/test_inference.py -q`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

- [ ] **Step 7: Commit**

```powershell
git add src/prodma/evaluacion tests/unit/test_evaluation_metrics.py tests/unit/test_inference.py
git commit -m "feat: add forecast inference and model ranking"
```

### Task 9: Etapa 4 de evaluacion y seleccion por horizonte

**Files:**
- Create: `src/prodma/stages/evaluation.py`
- Create: `tests/integration/test_evaluation_stage.py`

**Interfaces:**
- Consumes: trained models, untouched final-test partitions and clean matrix hash.
- Produces: `04_evaluacion/ranking/ganadores.csv` and evaluation artifacts.

- [ ] **Step 1: Write horizon isolation tests**

```python
def test_winner_is_selected_per_meter_variable_and_horizon(evaluation_run):
    winners = evaluation_run.winners
    assert winners.set_index(["medidor", "variable", "horizonte"]).index.is_unique
    assert set(winners["horizonte"]) == {96, 672}

def test_evaluation_rejects_model_trained_from_different_matrix(evaluation_inputs):
    evaluation_inputs.model_metadata["matrix_sha256"] = "wrong"
    with pytest.raises(ValueError, match="matriz limpia"):
        run_evaluation(**evaluation_inputs.as_kwargs())
```

- [ ] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_evaluation_stage.py -q`

- [ ] **Step 3: Implement the evaluation artifact tree**

Write metric tables, residual diagnostics, parametric results, non-parametric results, ranking, exclusions and figures to their named Stage 4 directories. `ganadores.csv` includes medidor, variable, horizon, model path, clean matrix hash, metrics, comparison with baseline and selection reason.

- [ ] **Step 4: Run GREEN and full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_evaluation_stage.py -q`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

- [ ] **Step 5: Commit**

```powershell
git add src/prodma/stages/evaluation.py tests/integration/test_evaluation_stage.py
git commit -m "feat: add horizon-specific model evaluation stage"
```

### Task 10: Etapa 5 de prediccion final e intervalos calibrados

**Files:**
- Create: `src/prodma/prediccion/__init__.py`
- Create: `src/prodma/prediccion/operational.py`
- Create: `src/prodma/prediccion/planning.py`
- Create: `src/prodma/prediccion/exports.py`
- Create: `src/prodma/stages/prediction.py`
- Create: `tests/unit/test_final_prediction.py`
- Create: `tests/integration/test_prediction_stage.py`

**Interfaces:**
- Consumes: Stage 4 winners and the Stage 2 clean matrix.
- Produces: `run_prediction(config, layout) -> StageResult`.

- [ ] **Step 1: Write exact-origin and no-winner tests**

```python
def test_forecast_starts_one_interval_after_last_observation(history, winner):
    forecast = forecast_operational(history, winner, horizon=96, frequency="15min")
    assert forecast.index[0] == history.dropna().index[-1] + pd.Timedelta(minutes=15)
    assert len(forecast) == 96

def test_series_without_valid_winner_is_not_published(tmp_path):
    result = run_prediction(config, layout_with_empty_winners(tmp_path))
    assert result.status == "PARTIAL"
    assert not list((result.output / "pronosticos").glob("*.csv"))
```

- [ ] **Step 2: Write interval-label test**

```python
def test_uncalibrated_bands_are_labeled_exploratory():
    exported = export_forecast(frame_with_bands(), coverage=None)
    assert set(exported["tipo_intervalo"]) == {"exploratorio"}
```

- [ ] **Step 3: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_final_prediction.py tests/integration/test_prediction_stage.py -q`

- [ ] **Step 4: Implement operational versus planning outputs**

Operational forecasts use only validated winner/horizon pairs. Planning forecasts use calendar profiles and versioned assumptions in a separate `escenarios/` dataset; they must not be labeled as validated ML forecasts. Reuse existing supervised client-stage logic only after its inputs are confirmed.

- [ ] **Step 5: Run GREEN and full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_final_prediction.py tests/integration/test_prediction_stage.py -q`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

- [ ] **Step 6: Commit**

```powershell
git add src/prodma/prediccion src/prodma/stages/prediction.py tests/unit/test_final_prediction.py tests/integration/test_prediction_stage.py
git commit -m "feat: add validated final prediction stage"
```

### Task 11: CLI reanudable que conecta las cinco etapas

**Files:**
- Replace: `src/prodma/pipeline.py`
- Create: `tests/unit/test_pipeline_cli.py`
- Create: `tests/integration/test_pipeline_smoke.py`
- Modify: `docs/MANUAL_EJECUCION.md`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int` and `python -m prodma.pipeline run|stage|resume|status`.
- Calls: `run_ingestion`, `run_cleaning`, `run_training`, `run_evaluation`, `run_prediction`.

- [ ] **Step 1: Write CLI order and resume tests**

```python
def test_run_executes_stages_in_contract_order(fake_stages):
    exit_code = main(["run", "--config", "config.yml", "--run-id", "r1"])
    assert exit_code == 0
    assert fake_stages.calls == ["01_ingesta", "02_limpieza", "03_entrenamiento", "04_evaluacion", "05_prediccion"]

def test_resume_reuses_success_only_when_fingerprints_match(completed_run):
    exit_code = main(["resume", "--run-id", completed_run.run_id])
    assert exit_code == 0
    assert completed_run.stage_calls == []
```

- [ ] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/test_pipeline_cli.py -q`

- [ ] **Step 3: Implement argparse commands and dependency gates**

`stage 03_entrenamiento` first verifies successful Stage 2 and matrix hash; Stage 4 verifies Stage 3; Stage 5 verifies winners. `resume` recalculates fingerprints and starts at the first missing or invalid stage. Preserve compatibility flags only by mapping them explicitly into the new config.

- [ ] **Step 4: Run synthetic end-to-end smoke**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_pipeline_smoke.py -q`

Expected: five `SUCCESS` stages, matrix hash propagated through training/evaluation/prediction, and every artifact inside the correct stage folder.

- [ ] **Step 5: Run full suite and update manual with exact commands**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

- [ ] **Step 6: Commit**

```powershell
git add src/prodma/pipeline.py tests/unit/test_pipeline_cli.py tests/integration/test_pipeline_smoke.py docs/MANUAL_EJECUCION.md
git commit -m "feat: orchestrate the five-stage demand pipeline"
```

### Task 12: Convertir el notebook en cliente fino y cerrar la migracion

**Files:**
- Modify: `notebook/proyecciondemanda.ipynb`
- Modify: `README.md`
- Create: `tests/test_notebook_thin_client.py`
- Create: `tests/acceptance/test_run_artifacts.py`

**Interfaces:**
- Notebook consumes CLI/API and manifests; it defines no production ingestion, cleaning, features, training, evaluation or prediction functions.

- [ ] **Step 1: Write failing notebook contract tests**

```python
def test_notebook_does_not_duplicate_pipeline_functions(notebook_sources):
    forbidden = ("def causal_fill", "def features", "def models", "def load_meter", "OptunaSearchCV(")
    assert not any(token in notebook_sources for token in forbidden)

def test_notebook_calls_modular_pipeline(notebook_sources):
    assert "from prodma.pipeline import" in notebook_sources
    assert "manifest.json" in notebook_sources
```

- [ ] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_notebook_thin_client.py -q`

- [ ] **Step 3: Mechanically rewrite notebook cells**

Keep narrative, configuration, invocation, status tables and artifact visualization. Remove duplicated production definitions and stale outputs. Use the project maintenance script for notebook JSON changes; do not manually reorder cell IDs.

- [ ] **Step 4: Add acceptance assertions**

```python
def test_complete_run_has_required_artifact_groups(run_dir):
    required = {
        "01_ingesta": ("parquet", "auditoria", "logs"),
        "02_limpieza": ("datos", "csv", "figuras", "estadistica_descriptiva", "auditoria", "logs"),
        "03_entrenamiento": ("modelos", "checkpoints", "figuras", "csv", "times", "traces", "logs"),
        "04_evaluacion": ("metricas", "inferencia_parametrica", "inferencia_no_parametrica", "diagnosticos", "ranking", "figuras", "logs"),
        "05_prediccion": ("pronosticos", "intervalos", "escenarios", "figuras", "entregables", "logs"),
    }
    for stage, children in required.items():
        assert all((run_dir / stage / child).is_dir() for child in children)
```

- [ ] **Step 5: Execute notebook smoke and complete verification**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Run: `.\.venv\Scripts\python.exe -m prodma.pipeline run --config config/default.yaml --mode smoke --run-id acceptance-smoke`

Run: `.\.venv\Scripts\python.exe -m pytest tests/acceptance/test_run_artifacts.py -q --run-dir resultados/acceptance-smoke`

Run: `.\.venv\Scripts\python.exe -m prodma.pipeline status --run-id acceptance-smoke`

Expected: test exit code 0, smoke exit code 0, five successful stages, unique time indices, finite published forecasts and exact matrix hash lineage.

- [ ] **Step 6: Update README and commit**

```powershell
git add notebook/proyecciondemanda.ipynb README.md tests/test_notebook_thin_client.py tests/acceptance/test_run_artifacts.py
git commit -m "refactor: make notebook a thin pipeline client"
```

### Task 13: Evidencia cientifica, trazabilidad y cierre operativo

**Files:**
- Create: `docs/VALIDACION_PIPELINE.md`
- Modify: `docs/MANUAL_EJECUCION.md`
- Create: `tests/acceptance/test_scientific_traceability.py`

**Interfaces:**
- Consumes: final manifests, metric tables, inference results and scientific references from the approved spec.
- Produces: evidence matrix mapping each methodological claim to code, test, artifact and source.

- [ ] **Step 1: Write traceability test**

```python
def test_each_required_method_has_code_test_artifact_and_source(traceability_table):
    required = {"imputacion_causal", "MAD_causal", "rolling_origin", "baseline_estacional",
                "wilcoxon", "friedman", "holm", "seleccion_por_horizonte"}
    rows = traceability_table.set_index("metodo")
    assert required <= set(rows.index)
    assert rows.loc[list(required), ["codigo", "prueba", "artefacto", "fuente"]].notna().all().all()
```

- [ ] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/acceptance/test_scientific_traceability.py -q`

- [ ] **Step 3: Document verified scientific support and limitations**

Use the exact theses and indexed articles listed in the approved spec. State which choices are literature-backed, which thresholds are project configuration, and which claims still require empirical validation on the unavailable real workbooks. Include no unsupported compliance or accuracy claims.

- [ ] **Step 4: Run final verification from a clean process**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Run: `.\.venv\Scripts\python.exe -m prodma.pipeline status --run-id acceptance-smoke`

Run: `git diff --check`

Expected: all tests pass, smoke manifest reports five successful stages, and no whitespace errors.

- [ ] **Step 5: Commit documentation and evidence**

```powershell
git add docs/VALIDACION_PIPELINE.md docs/MANUAL_EJECUCION.md tests/acceptance/test_scientific_traceability.py
git commit -m "docs: add pipeline validation traceability"
```

## Completion Checklist

- [ ] Re-read the approved spec and map every acceptance criterion to Tasks 1-13.
- [ ] Confirm `matriz_limpia.parquet` is the sole training input and its hash reaches every downstream manifest.
- [ ] Confirm no imputed row enters objectives or reported observed metrics.
- [ ] Confirm future-invariance tests cover cleaning and features.
- [ ] Confirm training owns separate figures, checkpoints, CSV, times, traces, models and logs.
- [ ] Confirm parametric and non-parametric outputs include assumptions, sample size, effect size and adjusted p-values.
- [ ] Confirm winners are selected independently by medidor, variable and validated horizon.
- [ ] Confirm operational forecast and long-term planning are labeled and stored separately.
- [ ] Confirm the notebook has no duplicated production implementation.
- [ ] Run the full suite and end-to-end smoke immediately before claiming completion.
