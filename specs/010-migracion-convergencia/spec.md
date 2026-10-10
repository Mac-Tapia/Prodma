# Spec 010: migración, paridad y convergencia

**Entrada:** implementaciones verificadas de Specs 001–009.
**Resultado:** retiro seguro del notebook como motor productivo y cierre integral.

## Requisitos funcionales

- **S010-FR-001:** migrar por slices, conservando adaptadores temporales solo con
  propietario, consumidores y criterio explícito de retiro.
- **S010-FR-002:** impedir dos implementaciones productivas de una misma regla; el
  notebook será cliente ligero de la API Python o se archivará como ejemplo.
- **S010-FR-003:** ejecutar pruebas de caracterización antes de retirar lógica
  heredada y pruebas de paridad sobre entradas representativas.
- **S010-FR-004:** preservar originales, ejecuciones históricas y cambios del
  usuario; ninguna limpieza de resultados es parte implícita de la migración.
- **S010-FR-005:** proporcionar migración de configuración y estado sin interpretar
  un artefacto antiguo incompatible como checkpoint válido.
- **S010-FR-006:** construir matriz final `requisito -> código -> prueba -> evidencia`
  para todas las sub-specs.
- **S010-FR-007:** ejecutar suite completa, pruebas frontend/backend, smoke de las
  cinco etapas y una corrida real proporcional antes de declarar convergencia.
- **S010-FR-008:** documentar limitaciones, bloqueos y componentes no verificados;
  una convergencia parcial no se etiqueta como reestructuración completa.

## Orden de migración

Custodia -> ingesta -> limpieza -> entrenamiento -> evaluación -> predicción ->
API/worker -> React -> agentes/memoria -> retiro de rutas heredadas.

## Aceptación

- `prodma.pipeline` orquesta módulos y no celdas del notebook.
- No existe código productivo duplicado entre notebook y paquete.
- Los contratos de artefactos antiguos incompatibles fallan con diagnóstico.
- La matriz de convergencia no contiene requisitos sin código/prueba/evidencia o una
  excepción explícitamente aceptada por el usuario.
- El informe final distingue pruebas sintéticas, smoke real y aceptación operativa.

## Trazabilidad prevista

Pruebas de caracterización, migración, paridad y end-to-end; informe de convergencia
versionado junto con los manifiestos de la corrida de aceptación.
