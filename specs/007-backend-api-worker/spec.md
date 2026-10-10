# Spec 007: backend local, API y worker

**Entrada:** APIs Python estables de Specs 001–006.
**Resultado:** control local recuperable sin ejecutar lógica científica en rutas HTTP.

## Requisitos funcionales

- **S007-FR-001:** FastAPI escuchará por defecto solo en loopback y expondrá una
  versión de contrato explícita.
- **S007-FR-002:** la API validará configuración, selección y horizonte antes de
  crear un trabajo; no aceptará rutas arbitrarias del cliente.
- **S007-FR-003:** un worker Python separado ejecutará operaciones pesadas para no
  bloquear consultas ni visualizaciones.
- **S007-FR-004:** solo una operación pesada podrá estar activa; el bloqueo incluirá
  PID, tiempo, run_id y recuperación segura de locks obsoletos.
- **S007-FR-005:** SQLite almacenará trabajos, eventos, progreso y referencias; los
  datos científicos permanecerán en artefactos versionados.
- **S007-FR-006:** soportar crear, consultar, cancelar cooperativamente y reanudar
  trabajos desde checkpoints compatibles.
- **S007-FR-007:** transmitir progreso mediante SSE o contrato equivalente
  reconectable, conservando eventos en disco/SQLite.
- **S007-FR-008:** endpoints de artefactos permitirán solo archivos registrados
  dentro del run y evitarán recorrido de rutas.
- **S007-FR-009:** un fallo o reinicio nunca convertirá trabajo incompleto en éxito.

## API mínima

`GET /health`, `GET /catalog`, `POST /runs`, `GET /runs`, `GET /runs/{id}`,
`POST /runs/{id}/cancel`, `POST /runs/{id}/resume`, `GET /runs/{id}/events` y
`GET /runs/{id}/artifacts`.

## Aceptación

- Contratos OpenAPI y pruebas de integración cubren éxito, conflicto y entradas
  inválidas.
- Un entrenamiento simulado largo no bloquea `/health` ni consultas de progreso.
- Reiniciar API/worker conserva el estado real y detecta trabajos interrumpidos.

## Trazabilidad prevista

Módulos objetivo: `prodma/application/`, `prodma/api/`, `prodma/worker/` y
`prodma/control/`. Pruebas API, concurrencia, reinicio y seguridad de rutas.
