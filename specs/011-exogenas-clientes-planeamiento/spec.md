# Spec 011: exógenas, clientes y planeamiento completo

**Entrada:** roadmap `000-reestructuracion-integral`; specs 001–010; plan
`docs/superpowers/plans/2026-10-12-cierre-integral-exogenas-modelos-prodma.md`.
**Resultado:** contrato verificable que cierra los hallazgos diagnósticos sobre
fuentes externas, clientes, causalidad, cobertura de modelos y trazabilidad.

## Historias

- Como operador, quiero que cada fuente declarada o descubierta tenga un estado
  terminal para no depender de omisiones silenciosas.
- Como analista, quiero que toda variable externa solo use información disponible
  en el origen del pronóstico.
- Como responsable de planeamiento, quiero que los clientes confirmados entren como
  escenario y no como historia ficticia.
- Como auditor, quiero reconciliar cada archivo, serie, variable, cliente y modelo
  con su evidencia.

## Requisitos funcionales

- **S011-FR-001:** toda fuente configurada, descubierta o declarada recibe exactamente
  uno de los estados `USED`, `REJECTED_WITH_EVIDENCE`, `SCENARIO_ONLY`,
  `BLOCKED_MISSING_SOURCE`, `BLOCKED_INVALID` o `SUPERSEDED`. No existe el estado
  implícito de deshabilitado sin decisión.
- **S011-FR-002:** el descubrimiento de entradas y fuentes externas es recursivo y
  no depende de listas fijas de años; los rangos temporales se infieren del contenido.
- **S011-FR-003:** cada respuesta externa se guarda cruda en
  `00_custodia/raw_external/<fuente>/<fecha_descarga>/` con URL, licencia, cabeceras
  y SHA-256; la promoción al almacenamiento final es atómica y solo tras verificación.
- **S011-FR-004:** cada variable exógena porta `periodo_referencia`,
  `fecha_disponible`, `fecha_descarga`, fuente, versión y política futura. Ninguna
  unión temporal usa datos publicados después del origen del pronóstico.
- **S011-FR-005:** el clima separa observación histórica, pronóstico operativo y
  climatología de planeamiento en columnas o estados distintos; un reanálisis
  interpolado nunca se etiqueta como medición directa ni como pronóstico futuro.
- **S011-FR-006:** las series económicas y de producción regional usan vintages o un
  rezago conservador declarado. Prohibido asignar a la fecha de publicación la fecha
  de descarga.
- **S011-FR-007:** población y proyecciones oficiales se etiquetan como `oficial`,
  `provisional` o `proyectada`. Una extrapolación más allá del horizonte oficial es
  un supuesto de escenario con cotas de sensibilidad, no una covariable observada.
- **S011-FR-008:** los clientes realizados con fecha de alta y los clientes
  confirmados futuros son datasets separados. Los realizados solo entran al
  entrenamiento tras su fecha de disponibilidad; los confirmados solo se aplican en
  planeamiento después de la inferencia base.
- **S011-FR-009:** la matriz candidata concatena todas las exógenas elegibles. Su uso
  final se decide por ablación pareada sobre los mismos folds y horizontes; cada
  descarte conserva su evidencia.
- **S011-FR-010:** la compatibilidad entre modelo y conjunto de exógenas es explícita.
  Una incompatibilidad produce `INCOMPATIBLE_EXOG`, nunca omisión.
- **S011-FR-011:** cada combinación serie × modelo termina en `TRAINED`,
  `PREFILTER_REJECTED`, `TIMEOUT`, `UNAVAILABLE`, `INCOMPATIBLE`, `FAILED` o
  `CANCELLED`. Ninguna fila esperada puede faltar.
- **S011-FR-012:** el backend efectivo (CPU o GPU) se declara solo después de un `fit`
  y un `predict` con valores finitos, registrando backend solicitado y efectivo.
- **S011-FR-013:** los intervalos probabilísticos se calibran con residuos no usados
  para ajuste, y su cobertura condicional se publica por horizonte.
- **S011-FR-014:** archivos, filas, hojas, medidores, series, exógenas, clientes,
  modelos, fallos y entregables se reconcilian por conjuntos y conteos antes de
  publicar cualquier resultado.
- **S011-FR-015:** solo se generan las siete familias de figuras definidas en el plan
  de cierre; cada figura se deriva de una tabla reabrible.
- **S011-FR-016:** años, medidores, modelos, variables y cantidad de requisitos no se
  fijan como constantes en configuración o documentación de convergencia; se obtienen
  por descubrimiento o registro.
- **S011-FR-017:** el calendario de días no laborables proviene de norma oficial con
  fecha, nombre, base legal y alcance. Un día sin norma verificada bloquea la
  etiqueta; no se rellena silenciosamente.
- **S011-FR-018:** cada decisión metodológica validada tiene registro en
  `docs/evidencia_cientifica.yaml` con artículo y tesis de maestría o doctoral,
  código, prueba y artefacto.

## Fallo cerrado

Una fuente con estado `BLOCKED_*` no puede alimentar entrenamiento ni predicción.
Un snapshot sin hash, una fecha de disponibilidad inferida desde la descarga, una
exógena sin estado terminal o un modelo sin fila de estado impiden publicar la
corrida. Una variable no usada debe tener evidencia de ablación, redundancia,
cobertura insuficiente o fuga.

## Aceptación

- Ninguna tabla de conteo contiene estados implícitos.
- Todo requisito `S011-FR-*` aparece en la trazabilidad diagnóstica y en `tasks.md`.
- Ningún documento de convergencia contiene la cifra fija de requisitos.
- La prueba de cobertura falla si cambia el texto de un hallazgo diagnóstico sin
  actualizar esta spec.

## Trazabilidad diagnóstica

Los hallazgos son los del plan de cierre; el texto de la columna `Hallazgo` debe
coincidir exactamente con la tabla del plan.

| Hallazgo | Requisitos | Tarea | Prueba | Evidencia |
|---|---|---|---|---|
| `config/default.yaml` fija años 2023–2026. | S011-FR-002, S011-FR-016 | Tarea 2 | tests/unit/test_input_inventory.py | 00_custodia/inventario.jsonl |
| Calendario tiene feriados sin norma verificada. | S011-FR-017 | Tarea 4 | tests/unit/test_calendar_connector.py | 00_custodia/raw_external/calendario_pe/ |
| Temperatura usa un único punto/reanálisis y climatología. | S011-FR-005 | Tarea 5 | tests/unit/test_weather_source_policy.py | 02_limpieza/datos/matriz_exogenas_historicas.parquet |
| PBI queda 100 % nulo en la matriz histórica por usar la fecha de descarga 2026 como disponibilidad. | S011-FR-004, S011-FR-006 | Tarea 6 | tests/unit/test_economic_asof.py | 02_limpieza/datos/calidad_exogenas.parquet |
| PBI es nacional y producción regional está deshabilitada/ausente. | S011-FR-006 | Tarea 6 | tests/unit/test_bcrp_connector.py | 02_limpieza/datos/catalogo_exogenas.json |
| Población tiene pocos puntos; parte es interpolada/extrapolada. | S011-FR-007 | Tarea 7 | tests/unit/test_inei_connector.py | 02_limpieza/datos/catalogo_exogenas.json |
| `clientes_confirmados` y `produccion_regional` están deshabilitados. | S011-FR-001, S011-FR-008 | Tarea 8 | tests/integration/test_new_customers_scenario.py | 05_prediccion/escenarios_clientes.parquet |
| `exogenas/matrix.py` y `validation.py` conocen solo cuatro nombres. | S011-FR-001, S011-FR-003 | Tarea 3 | tests/unit/test_external_registry.py | 00_custodia/raw_external/ |
| `stages/training.py` ejecuta `break` después del primer objetivo elegible. | S011-FR-011, S011-FR-014 | Tarea 11 | tests/integration/test_training_all_targets.py | 03_entrenamiento/manifest_entrenamiento.json |
| El registro contiene 13 modelos, pero el último run entrenó solo SeasonalNaive y Ridge. | S011-FR-010, S011-FR-011, S011-FR-012 | Tarea 13 | tests/integration/test_all_registered_models.py | 03_entrenamiento/csv/backends_efectivos.csv |
| Solo Ridge y RandomForest reciben tuning real. | S011-FR-011 | Tarea 13 | tests/unit/test_model_search_spaces.py | 03_entrenamiento/traces/optuna_trials.csv |
| `modeling/ablation.py` existe, pero no está integrado al stage. | S011-FR-009 | Tarea 12 | tests/integration/test_training_exogenous_ablation.py | 03_entrenamiento/ablacion/ablacion_exogenas.csv |
| `04_evaluacion/` y `05_prediccion/` no están implementados. | S011-FR-013, S011-FR-014 | Tarea 15 y 16 | tests/integration/test_evaluation_stage.py | 04_evaluacion/ y 05_prediccion/ |
| No existen API/worker/React y el notebook conserva lógica. | S011-FR-014, S011-FR-015, S011-FR-016 | Tarea 17 a 19 | tests/integration/test_api_jobs.py | docs/acceptance/README.md |
| `docs/evidencia_cientifica.yaml` deja ingesta, evaluación y predicción pendientes. | S011-FR-018 | Tarea 1 | tests/unit/test_scientific_evidence.py | docs/evidencia_cientifica.yaml |
