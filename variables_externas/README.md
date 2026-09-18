# Variables externas (exógenas)

Generado por `tools/generar_variables_externas.py` el 2026-09-17T14:10:39.
Implementa la sección 2 de `Plan_implementacion_demanda_Electro_Dunas.md` en lo que es
reproducible sin inventar datos ni asumir códigos de serie no confirmados.

## Implementado (datos reales en este folder)

| Carpeta | Archivo | Frecuencia | Cobertura |
|---|---|---|---|
| `calendario/` | `feriados_peru_2023_2036.csv` | diaria | 2023–2036 (horizonte completo, igual que `CFG['fin_plan']`) |
| `calendario/` | `feriados_15min_2023_2026.csv` | 15 min | 2023-01-01 00:15..2026-09-11 23:45:00 — **misma malla que `datahistorica/`** |
| `temperatura/` | `temperatura_horaria_*.csv` | horaria | 2023-01-01..hasta donde ERA5 tenga reanálisis disponible |
| `temperatura/` | `temperatura_15min_*.csv` | 15 min | 2023-01-01 00:15..2026-09-11 23:45:00 — **misma malla que `datahistorica/`**, interpolada linealmente entre horas (columna `interpolado`). Estado: OK: horaria 32400 filas (2023-01-01..2026-09-11); 15min 129599 filas (97200 interpoladas) |
| `economia/` | `pbi_mensual_bcrp_2023_2026.csv` | mensual | BCRPData, serie `PN01773AM` (PBI desestacionalizado, índice 2007=100), código confirmado por el usuario. Estado: OK: 42 filas (Ene.2023..Jun.2026) |
| `poblacion/` | `poblacion_ica_2023_2026.csv` | anual (30 jun) | 2023–2026. Solo 2022 y 2024 son observación oficial INEI; el resto está **interpolado/extrapolado linealmente** (columna `dato`) — ver limitación abajo. |

Las versiones `_15min` tienen la misma frecuencia y el mismo rango de fechas que el
dataset de demanda histórico (2023 en adelante, cada 15 minutos), pensadas para unirse
fila a fila con `datahistorica/` sin reindexar. Las versiones originales (diaria/horaria)
se conservan también: son la fuente real, antes de cualquier interpolación.

**Limitaciones que el plan exige declarar explícitamente (no ocultar):**

- `calendario/…csv`, columna `norma`: queda vacía. La biblioteca `holidays` no fue
  contrastada contra <https://www.gob.pe/feriados> (esa página no se pudo leer al
  redactar el plan); verificar traslados de feriados por decreto antes de un uso
  operativo.
- `temperatura/…csv` es **reanálisis ERA5, no medición directa de una estación
  SENAMHI**, y usa **una sola coordenada** (`ica_centro`,
  lat=-14.0678, lon=-75.7286) para representar toda
  el área de concesión de Electro Dunas. El plan advierte explícitamente que no se
  evaluó representatividad por alimentador/zona.
- `temperatura_15min_*.csv`: solo la marca `:00` de cada hora es la observación real de
  Open-Meteo; `:15/:30/:45` son interpolación lineal (columna `interpolado=True`), nunca
  una medición nueva.
- Para el pronóstico recursivo hasta `CFG['fin_plan']` (2037), donde no existe
  temperatura real futura, el notebook usa una **climatología** (mediana histórica por
  mes/hora, `construir_climatologia_mensual_horaria` en
  `src/prodma/variables_externas.py`) como supuesto declarado — no un pronóstico
  meteorológico real.
- `economia/pbi_mensual_bcrp_*.csv` es PBI **nacional**, no regional de Ica (BCRP no
  publica PBI regional mensual); el plan advierte que un indicador nacional es una
  aproximación, no una medida directa de la actividad económica de la concesión.
- `poblacion/poblacion_ica_*.csv`: **solo 2022 y 2024 son observación oficial de INEI**
  (proyección al 30 de junio); 2023, 2025 y 2026 son interpolación/extrapolación lineal
  entre esos dos puntos, marcada explícitamente en la columna `dato` — nunca se presenta
  como observación oficial. Reemplazar por la serie completa de INEI 2018-2026 en cuanto
  se pueda extraer del PDF oficial o de una fuente tabular confirmada.

## Pendiente (no implementado aquí; ver README dentro de cada carpeta)

| Carpeta | Por qué no se generó |
|---|---|
| `clientes/` | Requiere el padrón comercial interno de Electro Dunas (sistema comercial); no se identificó una fuente pública con el detalle por alimentador/segmento que exige el plan. |
| `produccion_regional/` | INEI (PBI departamental) y BCRP/MINEM (síntesis regional, boletín minero) solo están disponibles como publicación/PDF; el plan exige extraer y verificar cada tabla manualmente antes de automatizar, no scrapear el PDF a ciegas. |

Cada carpeta pendiente tiene su propio `README.md` con el esquema mínimo de columnas
(sección 3 del plan) para cuando la fuente esté disponible.
