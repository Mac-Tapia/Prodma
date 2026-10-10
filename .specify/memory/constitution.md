# Constitución de Prodma

**Versión:** 1.0.0
**Ratificada:** 2026-10-10

## Propósito

Prodma transforma mediciones eléctricas históricas en pronósticos auditables por
medidor para operación, planeamiento y entregables de PowerFactory. La
reproducibilidad, la causalidad temporal y la procedencia de los datos prevalecen
sobre la velocidad o la cantidad de resultados publicados.

## Principios no negociables

### I. Cobertura total y custodia

- Se inventariarán todos los archivos actuales y futuros de las raíces de entrada
  declaradas. Ningún archivo, libro, hoja, medidor o variable desaparecerá sin un
  estado y una causa registrados.
- Los originales son de solo lectura. Cada entrada tendrá ruta relativa, tamaño,
  fecha, SHA-256, tipo detectado y resultado de procesamiento.
- Un formato desconocido se cataloga y queda en cuarentena; nunca se interpreta
  mediante heurísticas destructivas.

### II. Cinco etapas independientes

`01_ingesta`, `02_limpieza`, `03_entrenamiento`, `04_evaluacion` y
`05_prediccion` poseen entradas, salidas, estado y versión contractual propios.
Una etapa dependiente solo puede consumir artefactos publicados y verificados de
la etapa anterior. Ingesta, limpieza, entrenamiento, evaluación y predicción no
se fusionarán en un notebook ni en una función monolítica.

### III. Causalidad y verdad observada

- Ninguna transformación de entrenamiento o evaluación puede consultar el futuro.
- `observado`, `contexto`, `limpio`, `imputado`, `corregido`, `atipico` y
  `rechazado` son estados distintos y conservan su procedencia.
- Un valor imputado puede alimentar contexto histórico aprobado, pero nunca se
  convierte en objetivo observado.
- Los Excel crudos no son entrada admisible para entrenamiento, evaluación o
  predicción; la matriz limpia publicada es su única fuente de demanda.

### IV. Universo autorizado de proyección

- `medidoralimentador.xlsx`, hoja `alimentador_medidor`, columna `Medidor`, es la
  lista maestra de autorización.
- Un medidor proyectable pertenece simultáneamente al catálogo autorizado, a la
  matriz limpia y al conjunto con historia observada suficiente.
- React puede desmarcar medidores elegibles, pero no agregar medidores externos.
  Todos los elegibles aparecen seleccionados por defecto.

### V. Evidencia antes que estado visual

Se distinguirán comando ejecutado, archivo escrito, artefacto reabierto,
contenido verificado y aceptación operativa. Un indicador `SUCCESS` o `OK` no
sustituye hashes, reconciliaciones, pruebas ni artefactos. Todo resultado publicado
se vincula con `run_id`, configuración resuelta, versiones, entradas y backend
efectivo.

### VI. Validez científica

- La validación es temporal y fuera de muestra, con horizontes equivalentes al uso
  declarado y referencias simples obligatorias.
- Las comparaciones conservan métricas, incertidumbre, tamaño de muestra,
  exclusiones y costo. La significancia estadística no decide sola.
- Una proyección de largo plazo se etiqueta como planeamiento basado en supuestos;
  no se presenta como precisión operacional si ese horizonte no fue validado.

### VII. Arquitectura local, modular y recuperable

El producto será un monolito modular local: núcleo Python, FastAPI en loopback,
worker Python separado, SQLite solo para control, React/TypeScript y artefactos
Parquet/JSON/CSV. Se permite una operación pesada activa; los trabajos son
reanudables y no sobrescriben ejecuciones anteriores.

### VIII. Specs, TDD y convergencia

Cada cambio sustancial sigue `spec.md -> plan.md -> tasks.md -> implementación ->
convergencia`. Las pruebas se escriben antes de la implementación cuando cambie
comportamiento. La aceptación exige pruebas focalizadas, suite completa y el nivel
de ejecución real proporcional al cambio.

### IX. Contexto y memoria para agentes

`AGENTS.md` contendrá solo reglas estables. Skills locales cargarán contexto por
módulo. La memoria registrará fuente, fecha, alcance y estado (`confirmado`,
`historico`, `pendiente`, `obsoleto`); nunca promoverá una evidencia antigua a
validación actual.

## Puertas globales

1. Cobertura de inventario reconciliada.
2. Contratos y hashes de la etapa anterior verificados.
3. Ausencia de fuga temporal demostrada.
4. Catálogo de medidores aplicado antes de entrenamiento y predicción.
5. Artefactos reabiertos y validados antes de publicar.
6. Suite completa y smoke end-to-end satisfactorios antes de declarar convergencia.

## Gobierno

Una modificación de estos principios requiere una spec aprobada, explicación del
impacto, versión nueva de la constitución y pruebas que preserven o migren los
contratos afectados. Las specs subordinadas no pueden debilitar esta constitución.
