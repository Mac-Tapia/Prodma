# Plan de implementación — Agentes, contexto y memoria

> **For agentic workers:** REQUIRED: Follow this plan task-by-task with TDD. Project memory is advisory until verified against the current workspace and must never override source-of-truth artifacts.

**Goal:** Hacer que cualquier agente pueda comprender y operar Prodma con instrucciones cortas, contexto progresivo y memoria auditable sin consumir innecesariamente el contexto.

**Architecture:** `AGENTS.md` será el punto de entrada. Dos skills locales separarán lectura contextual de escritura/consulta de memoria. La memoria tendrá una capa estable y un registro operativo append-only con procedencia y fecha.

**Tech Stack:** Markdown/YAML, PowerShell, Python validation tests, SHA-256.

---

## Global constraints

- Las instrucciones deben ser concisas y enlazar documentación especializada.
- Ninguna memoria se acepta como hecho vigente sin verificarla contra archivos actuales.
- No almacenar secretos, datos personales ni datasets masivos en memoria textual.
- Toda entrada mutable registra fecha, fuente, alcance y estado de verificación.
- Mantener compatibilidad con agentes que solo descubren `AGENTS.md`.

## Task 1: Punto de entrada AGENTS.md

**Files:**
- Create: `AGENTS.md`
- Create: `docs/agents/project-map.md`
- Test: `tests/agent/test_agent_instructions.py`

**Step 1 — Write failing tests:** Validar existencia, secciones obligatorias, comandos vigentes, límites de seguridad y enlaces resolubles.

**Step 2 — Run to verify RED:** `pytest tests/agent/test_agent_instructions.py -q`

**Step 3 — Implement minimally:** Documentar propósito, módulos, flujo, fuentes de verdad, gates, comandos y política de evidencia.

**Step 4 — Run to verify GREEN:** `pytest tests/agent/test_agent_instructions.py -q`

**Step 5 — Commit:** `git add AGENTS.md docs/agents tests/agent && git commit -m "docs: add agent entrypoint for Prodma"`

## Task 2: Skill de contexto progresivo

**Files:**
- Create: `.agents/skills/prodma-context/SKILL.md`
- Create: `.agents/skills/prodma-context/references/module-map.md`
- Create: `.agents/skills/prodma-context/references/data-contracts.md`
- Create: `.agents/skills/prodma-context/references/evidence-gates.md`
- Create: `.agents/skills/prodma-context/scripts/inspect-project.ps1`
- Test: `tests/agent/test_prodma_context_skill.py`

**Step 1 — Write failing tests:** Verificar frontmatter, rutas, instrucciones de carga progresiva y salida determinista del inspector.

**Step 2 — Run to verify RED:** `pytest tests/agent/test_prodma_context_skill.py -q`

**Step 3 — Implement minimally:** Skill que primero inspeccione estado y luego cargue solo la referencia del módulo implicado.

**Step 4 — Run to verify GREEN:** `pytest tests/agent/test_prodma_context_skill.py -q`

**Step 5 — Commit:** `git add .agents/skills/prodma-context tests/agent && git commit -m "feat: add token-efficient Prodma context skill"`

## Task 3: Skill y almacenamiento de memoria auditable

**Files:**
- Create: `.agents/skills/prodma-memory/SKILL.md`
- Create: `.agents/skills/prodma-memory/references/schema.md`
- Create: `.agents/skills/prodma-memory/scripts/search-memory.ps1`
- Create: `.agents/skills/prodma-memory/scripts/append-memory.ps1`
- Create: `docs/agent-memory/stable.md`
- Create: `docs/agent-memory/operational.jsonl`
- Test: `tests/agent/test_prodma_memory_skill.py`

**Step 1 — Write failing tests:** Cubrir búsqueda limitada, append-only, campos de procedencia, rechazo de secretos y deduplicación por hash.

**Step 2 — Run to verify RED:** `pytest tests/agent/test_prodma_memory_skill.py -q`

**Step 3 — Implement minimally:** Definir esquema, consultas acotadas y escritura explícita con verificación y hash.

**Step 4 — Run to verify GREEN:** `pytest tests/agent/test_prodma_memory_skill.py -q`

**Step 5 — Commit:** `git add .agents/skills/prodma-memory docs/agent-memory tests/agent && git commit -m "feat: add auditable two-tier project memory"`

## Task 4: Validación integral y presupuesto de contexto

**Files:**
- Create: `scripts/validate_agent_context.ps1`
- Create: `tests/agent/test_context_budget.py`
- Modify: `AGENTS.md`

**Step 1 — Write failing tests:** Detectar enlaces rotos, comandos obsoletos, archivos demasiado extensos y carga inicial que exceda el presupuesto definido.

**Step 2 — Run to verify RED:** `pytest tests/agent/test_context_budget.py -q`

**Step 3 — Implement minimally:** Validador y límites documentados; mover detalle a referencias cuando sea necesario.

**Step 4 — Verify:** `pytest tests/agent -q` y `pwsh -File scripts/validate_agent_context.ps1`

**Step 5 — Commit:** `git add AGENTS.md .agents docs/agents docs/agent-memory scripts tests/agent && git commit -m "test: validate agent context and memory contracts"`

## Review focus

- Instrucciones contradictorias o demasiado amplias.
- Memoria presentada como confirmación actual sin evidencia.
- Scripts que incluyan datos sensibles o generen contexto excesivo.
- Compatibilidad de rutas y comandos en Windows.
