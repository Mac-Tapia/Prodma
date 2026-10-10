# Spec 009: agentes, contexto y memoria

**Entrada:** contratos aprobados de Specs 001–008.
**Resultado:** cualquier agente compatible obtiene contexto mínimo y verificable.

## Requisitos funcionales

- **S009-FR-001:** crear `AGENTS.md` raíz con propósito, arquitectura, fuentes,
  reglas no negociables, comandos y puertas de evidencia.
- **S009-FR-002:** crear `.agents/skills/prodma-context/SKILL.md` para clasificar la
  consulta por módulo y cargar solo contratos, código y pruebas pertinentes.
- **S009-FR-003:** crear `.agents/skills/prodma-memory/SKILL.md` con memoria de dos
  niveles: índice corto y documentos profundos bajo `docs/agent-memory/`.
- **S009-FR-004:** cada entrada de memoria tendrá afirmación, fuente, fecha de
  verificación, alcance, estado y criterio de obsolescencia.
- **S009-FR-005:** la memoria solo se actualiza por instrucción humana explícita o
  por un flujo aprobado que presente el diff antes de escribir.
- **S009-FR-006:** una memoria histórica nunca se presenta como estado actual sin
  verificación; resultados efectivos se referencian por ruta/hash, no se copian como
  afirmaciones sueltas.
- **S009-FR-007:** incluir referencias a constitución, roadmap, specs y mapa de
  módulos sin duplicar su contenido extenso en `AGENTS.md`.
- **S009-FR-008:** añadir punteros mínimos de compatibilidad para otros agentes solo
  cuando lean una convención diferente; `AGENTS.md` sigue siendo la autoridad.
- **S009-FR-009:** validar longitud, enlaces, triggers y carga progresiva para evitar
  inflación de tokens.

## Estructura objetivo

```text
AGENTS.md
.agents/skills/prodma-context/{SKILL.md,references/,scripts/}
.agents/skills/prodma-memory/{SKILL.md,references/,scripts/}
docs/agent-memory/{index.md,glossary.md,current-state.md,decisions/,known-failures/,evidence-index/}
```

## Aceptación

- Consultas representativas cargan únicamente el contexto del módulo implicado.
- Una entrada obsoleta se etiqueta y no contamina una conclusión actual.
- Enlaces y scripts funcionan en PowerShell desde la raíz.
- Una auditoría compara instrucciones con specs y detecta contradicciones.

## Sustento

La separación `AGENTS.md`/skills sigue la guía oficial de OpenAI; la recuperación
selectiva se apoya en investigación de contexto estructural para repositorios citada
en `specs/000-reestructuracion-integral/research.md`.
