# Spec 003: limpieza masiva, causal y completa

**Entrada:** datos estandarizados de Spec 002.
**Resultado:** única matriz limpia admisible por etapas 3–5.

## Historias

- Como científico de datos, quiero limpiar toda serie sin fuga temporal.
- Como auditor, quiero saber qué pasó con cada valor y con cada serie.

## Requisitos funcionales

- **S003-FR-001:** procesar todas las series inventariadas; una serie fallida se
  registra y no desaparece del informe de cobertura.
- **S003-FR-002:** conservar por variable valor original, observado válido, contexto
  causal, valor limpio y banderas de imputación, corrección, atípico, recorte,
  congelamiento, hueco, cero y rechazo.
- **S003-FR-003:** completar la malla temporal sin convertir filas creadas en
  observaciones.
- **S003-FR-004:** aplicar relleno hacia adelante limitado y rezago semanal solo
  como contexto, con método y fuente temporal registrados.
- **S003-FR-005:** detectar atípicos con estadísticas calculadas únicamente con
  pasado; quedan prohibidas ventanas centradas para decisiones productivas.
- **S003-FR-006:** aplicar reglas eléctricas configurables de finitud, P/Q/S,
  tensión, frecuencia, factor de potencia y sensor, sin inventar límites faltantes.
- **S003-FR-007:** integrar exógenas con disponibilidad `as-of`; una publicación
  futura no puede aparecer en una fila histórica anterior.
- **S003-FR-008:** publicar `matriz_limpia.parquet`, máscaras, descriptivos,
  auditoría fila a fila, cobertura y hash.
- **S003-FR-009:** reconciliar por archivo, hoja, medidor y variable los estados
  recibido, limpio, sin datos suficientes, cuarentena y fallo.

## Fallo cerrado

No se publica matriz si existe desalineación de índices, columnas no finitas sin
bandera, pérdida inexplicada de series o imposibilidad de reproducir el hash. Un
estado `PARTIAL` conserva resultados aislados, pero no habilita automáticamente el
entrenamiento de series afectadas.

## Aceptación

- Alterar valores futuros no cambia decisiones anteriores al punto alterado.
- Ninguna imputación aparece con `es_observado=True`.
- La matriz reabierta coincide en esquema, filas, hash y métricas de cobertura.
- Todas las entradas de Spec 002 tienen destino reconciliado.

## Trazabilidad prevista

Módulos objetivo: `prodma/cleaning/` y `prodma/exogenous/`, migrando
`limpieza/`, `exogenas/` y `stages/cleaning.py`. Evidencia en `02_limpieza/`.
