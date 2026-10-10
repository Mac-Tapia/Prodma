# Spec 006: predicción autorizada y entregables

**Entrada:** ganadores válidos de Spec 005.
**Resultado:** pronósticos y entregables sin medidores no autorizados.

## Requisitos funcionales

- **S006-FR-001:** calcular elegibles como intersección de catálogo autorizado,
  matriz limpia, historia suficiente y ganador válido.
- **S006-FR-002:** seleccionar todos los elegibles por defecto y aceptar desde React
  solo subconjuntos de esa lista.
- **S006-FR-003:** usar `2028-01-01 00:00` como fin exclusivo inicial y permitir un
  horizonte distinto validado por backend.
- **S006-FR-004:** separar pronóstico operacional validado de planeamiento de largo
  plazo basado en perfiles y supuestos versionados.
- **S006-FR-005:** comenzar después de la última observación elegible, sin
  solapamiento ni huecos fabricados.
- **S006-FR-006:** aplicar exógenas futuras solo si su política y cobertura están
  declaradas; escenarios no se mezclan con historia observada.
- **S006-FR-007:** llamar intervalo predictivo únicamente a una banda calibrada fuera
  de muestra; las demás se etiquetan como exploratorias.
- **S006-FR-008:** producir Parquet/CSV, figuras, escenarios y tablas para
  PowerFactory con unidad, medidor, modelo, origen, horizonte y escenario.
- **S006-FR-009:** verificar que ningún artefacto contenga un medidor fuera del
  subconjunto autorizado solicitado.

## Aceptación

- Intentar inyectar un medidor externo por API produce rechazo.
- Cambiar el horizonte en React llega a configuración resuelta y artefactos.
- Los entregables reabiertos son finitos, continuos y reconciliados con la selección.
- Una serie bloqueada no aparece como cero ni como pronóstico exitoso.

## Trazabilidad prevista

Módulos objetivo: `prodma/prediction/` y `prodma/delivery/`. Evidencia en
`05_prediccion/` y manifiesto raíz.
