# Roadmap: reestructuración integral de Prodma

**Estado:** diseño aprobado para especificación
**Fecha:** 2026-10-10
**Constitución:** `.specify/memory/constitution.md`

## Resultado buscado

Convertir el notebook y los módulos parciales actuales en una aplicación local
completa y auditable que descubre todas las entradas, procesa y limpia toda la
cobertura, entrena y evalúa modelos, publica proyecciones autorizadas y permite
controlar el flujo desde React sin cambiar la finalidad eléctrica de Prodma.

## Decisiones aprobadas

- Monolito modular local con núcleo Python, FastAPI, worker, SQLite y React.
- Todas las fuentes actuales y futuras se inventarían; ninguna omisión silenciosa.
- Originales de solo lectura y matriz limpia como fuente productiva posterior.
- Proyección restringida por `medidoralimentador.xlsx`.
- Todos los medidores elegibles seleccionados por defecto; React permite reducir
  el subconjunto, nunca ampliarlo.
- Horizonte editable en React y cierre de 2027 como valor inicial
  (`2028-01-01 00:00`, extremo exclusivo).
- `AGENTS.md`, `prodma-context` y `prodma-memory` formarán el contrato para agentes.

## Brecha comprobada

- Existen implementaciones modulares de ingesta, limpieza y entrenamiento.
- `04_evaluacion` y `05_prediccion` no poseen aún sus módulos productivos completos.
- `prodma.pipeline` continúa orquestando el notebook.
- `config/default.yaml` enumera años fijos, contrario al descubrimiento futuro.
- No existen API FastAPI, worker local, frontend React ni skills del repositorio.

## Mapa de sub-specs

| Orden | Spec | Entrega independiente | Depende de |
|---:|---|---|---|
| 1 | `001-custodia-inventario` | Inventario total, hashes y catálogo autorizado | Constitución |
| 2 | `002-ingesta-preprocesamiento` | Parquet estandarizado y auditoría por hoja | 001 |
| 3 | `003-limpieza-masiva` | Matriz limpia única y reconciliación total | 002 |
| 4 | `004-entrenamiento-modelos` | Modelos/checkpoints reproducibles | 003 |
| 5 | `005-evaluacion-seleccion` | Backtests, diagnóstico y ranking | 004 |
| 6 | `006-prediccion-entregables` | Pronósticos, escenarios y PowerFactory | 005 |
| 7 | `007-backend-api-worker` | Orquestación local y contratos HTTP/eventos | 001–006 |
| 8 | `008-frontend-react` | Control y visualización local | 007 |
| 9 | `009-agentes-contexto-memoria` | Instrucciones y contexto progresivo | 001–008 |
| 10 | `010-migracion-convergencia` | Sustitución segura del notebook productivo | 001–009 |

Cada sub-spec tendrá su propio `plan.md` y `tasks.md` después de ser revisada y
aprobada. La implementación avanzará en el orden indicado; una spec puede preparar
interfaces futuras, pero no declarar completa una dependencia pendiente.

## Flujo objetivo

```text
raíces de entrada
  -> inventario inmutable
  -> ingesta estandarizada
  -> limpieza y matriz única
  -> entrenamiento
  -> evaluación y ranking
  -> predicción autorizada
  -> entregables y visualizaciones

React <-> FastAPI <-> orquestador <-> worker local
                         |
                         +-> SQLite (control)
                         +-> resultados/<run_id>/ (evidencia científica)
```

## Trazabilidad global

Cada requisito usa el identificador `Snnn-FR-nnn`. Los planes deberán mapearlo a
módulos y tareas; las pruebas usarán el identificador en nombre o docstring cuando
sea práctico; el manifiesto de ejecución conservará los gates aplicables. Una
matriz de convergencia final demostrará `requisito -> código -> prueba -> artefacto`.

## Definición de terminado

- Las diez sub-specs convergen sin requisitos sin implementar.
- Todos los archivos de entrada están reconciliados en un inventario.
- Las cinco etapas se ejecutan sin lógica productiva en el notebook.
- La selección de medidores y el horizonte quedan demostrados desde React hasta
  los entregables.
- Los artefactos se reabren, sus hashes coinciden y no contienen resultados fuera
  del universo autorizado.
- Las pruebas unitarias, contractuales, integración, frontend y end-to-end pasan.
- Una ejecución real documenta cobertura, backend efectivo, tiempos y limitaciones.

## Preservación

No se borrarán originales, resultados históricos ni cambios del usuario. La spec
anterior de 2026-09-20 permanece como antecedente; cuando una regla difiera,
prevalecen esta constitución y las sub-specs aprobadas más recientes.
