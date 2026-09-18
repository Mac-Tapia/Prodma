# Economía nacional (PBI mensual, BCRPData)

Fuente: BCRPData (<https://estadisticas.bcrp.gob.pe/estadisticas/series/mensuales/resultados/PN01773AM/html>) vía su API pública `.../api/[codigo]/json/[inicio]/[fin]/esp`.

**Código confirmado por el usuario el 2026-09-17:** `PN01773AM` — "PBI Desestacionalizado - mensual" (índice 2007=100). Es PBI **nacional**, no regional de Ica (BCRP no publica PBI regional con frecuencia mensual) — se usa como proxy de actividad económica general, no como medida directa de la concesión.

```python
from prodma.variables_externas import CODIGO_BCRP_PBI_MENSUAL, plantilla_bcrp
df = plantilla_bcrp(CODIGO_BCRP_PBI_MENSUAL, '2023-1', '2026-12')
```

Esquema (clave `serie_id, periodo`): `serie_id, periodo, valor, fuente, fecha_descarga`.
