# Spec 008: frontend React local

**Entrada:** OpenAPI y eventos de Spec 007.
**Resultado:** control y visualización sin editar fuentes originales.

## Requisitos funcionales

- **S008-FR-001:** React + TypeScript + Vite consumirá una capa cliente generada o
  validada contra OpenAPI.
- **S008-FR-002:** ofrecer vistas de resumen, fuentes/cobertura, limpieza,
  entrenamiento, evaluación, predicción, ejecuciones y artefactos.
- **S008-FR-003:** mostrar todos los archivos y series con estados explícitos; los
  rechazados o en cuarentena no desaparecen por filtros iniciales.
- **S008-FR-004:** mostrar medidores autorizados, disponibles, elegibles y motivo de
  exclusión. Seleccionar todos los elegibles por defecto y permitir desmarcarlos.
- **S008-FR-005:** editar fecha/hora final con cierre de 2027 precargado, zona
  `America/Lima` visible y validación del backend como autoridad.
- **S008-FR-006:** controlar etapas, cancelación cooperativa y reanudación sin
  habilitar acciones cuyos prerrequisitos falten.
- **S008-FR-007:** visualizar cobertura, calidad, métricas, ranking, pronósticos,
  incertidumbre y procedencia desde artefactos, no desde cálculos duplicados en JS.
- **S008-FR-008:** distinguir `PENDING`, `RUNNING`, `SUCCESS`, `PARTIAL`, `FAILED`,
  `CANCELLED` y `BLOCKED` con texto, no solo color.
- **S008-FR-009:** no permitir editar ni sobrescribir Excel/CSV de entrada.

## Aceptación

- Pruebas de componentes cubren selección, horizonte, estados y errores.
- Pruebas end-to-end demuestran UI -> API -> worker -> progreso -> artefacto.
- Refrescar o reconectar no pierde el run ni muestra éxito inexistente.
- Accesibilidad básica: teclado, foco, etiquetas y contraste verificables.

## Trazabilidad prevista

Proyecto objetivo `web/` organizado por features y contratos compartidos. Vitest,
Testing Library y Playwright verificarán comportamiento; las reglas de rendimiento
React se aplicarán después de preservar corrección.
