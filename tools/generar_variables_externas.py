"""Genera variables_externas/ a partir de lo que el plan de fuentes permite
automatizar hoy sin inventar datos (ver Plan_implementacion_demanda_Electro_Dunas.md,
seccion 2). Ejecutar una sola vez (o cuando cambien los años de interés) con:

    .\\.venv\\Scripts\\python.exe tools\\generar_variables_externas.py

No requiere paquetes fuera de requirements.txt (usa `holidays`, ya instalado, y
`urllib` de la biblioteca estándar para llamar a la API pública de Open-Meteo).
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'src'))

import pandas as pd  # noqa: E402

from prodma.variables_externas import (  # noqa: E402
    CODIGO_BCRP_PBI_MENSUAL,
    descargar_temperatura_openmeteo,
    expandir_calendario_a_15min,
    expandir_temperatura_a_15min,
    generar_calendario_peru,
    generar_poblacion_ica_interpolada,
    plantilla_bcrp,
)

# Mismo rango que CFG['fin_plan'] del notebook (celda 25: holidays hasta 2037 exclusive).
ANIO_INICIO_CALENDARIO = 2023
ANIO_FIN_CALENDARIO = 2036

# Rango del dataset de demanda (demanda2023..demanda2026.xlsx, misma fecha de inicio de
# intervalo que CFG['fecha_inicio'] del notebook): las versiones "15min" de calendario y
# temperatura deben tener la MISMA cantidad de filas/frecuencia que ese dataset.
FECHA_INICIO_15MIN = '2023-01-01 00:15'
ANIO_FIN_TEMPERATURA = 2026  # para nombrar los archivos hourly (rango deseado, se recorta abajo)

# Coordenada única, representativa de Ica (capital de la concesión de Electro Dunas).
# PENDIENTE (ver README generado): reemplazar por una coordenada por zona/alimentador
# cuando exista catalogo_red con ubicaciones reales; el plan advierte explícitamente
# que no se debe usar una sola temperatura de Ica para todos los alimentadores sin
# evaluar representatividad.
ZONA_TEMPERATURA = dict(etiqueta_zona='ica_centro', lat=-14.0678, lon=-75.7286)


def main() -> None:
    out = RAIZ / 'variables_externas'
    (out / 'calendario').mkdir(parents=True, exist_ok=True)
    (out / 'temperatura').mkdir(parents=True, exist_ok=True)
    (out / 'clientes').mkdir(parents=True, exist_ok=True)
    (out / 'economia').mkdir(parents=True, exist_ok=True)
    (out / 'poblacion').mkdir(parents=True, exist_ok=True)
    (out / 'produccion_regional').mkdir(parents=True, exist_ok=True)

    print('Generando calendario Perú', ANIO_INICIO_CALENDARIO, '-', ANIO_FIN_CALENDARIO, '...')
    calendario = generar_calendario_peru(ANIO_INICIO_CALENDARIO, ANIO_FIN_CALENDARIO)
    ruta_calendario = out / 'calendario' / f'feriados_peru_{ANIO_INICIO_CALENDARIO}_{ANIO_FIN_CALENDARIO}.csv'
    calendario.to_csv(ruta_calendario, index=False)
    print('  ->', ruta_calendario, f'({len(calendario)} filas, diario)')

    # El archivo ERA5 de Open-Meteo tiene un retraso de varios días respecto al presente;
    # se recorta automáticamente a la última fecha con reanálisis disponible y se
    # documenta la fecha real obtenida (el plan exige declarar, no ocultar, el corte).
    fecha_fin_deseada = pd.Timestamp(f'{ANIO_FIN_TEMPERATURA}-12-31')
    fecha_fin_segura = min(fecha_fin_deseada, pd.Timestamp.now().normalize() - pd.Timedelta(days=6))
    fecha_fin_horaria = fecha_fin_segura.strftime('%Y-%m-%d')
    fecha_inicio_horaria = FECHA_INICIO_15MIN[:10]  # Open-Meteo trabaja por día calendario

    print(f'Descargando temperatura horaria Open-Meteo {fecha_inicio_horaria}..{fecha_fin_horaria} '
          f'para {ZONA_TEMPERATURA["etiqueta_zona"]} ...')
    try:
        temperatura = descargar_temperatura_openmeteo(
            lat=ZONA_TEMPERATURA['lat'], lon=ZONA_TEMPERATURA['lon'],
            fecha_inicio=fecha_inicio_horaria, fecha_fin=fecha_fin_horaria,
            etiqueta_zona=ZONA_TEMPERATURA['etiqueta_zona'])
        ruta_temp_horaria = out / 'temperatura' / (
            f'temperatura_horaria_{ZONA_TEMPERATURA["etiqueta_zona"]}_'
            f'{ANIO_INICIO_CALENDARIO}_{ANIO_FIN_TEMPERATURA}.csv')
        temperatura.to_csv(ruta_temp_horaria, index=False)
        print('  ->', ruta_temp_horaria, f'({len(temperatura)} filas, hasta {fecha_fin_horaria})')

        # Misma malla de 15 min que datahistorica/ (fin de intervalo, 00:15 de inicio),
        # recortada a la última hora completa realmente descargada.
        fecha_fin_15min = (pd.Timestamp(fecha_fin_horaria) + pd.Timedelta(hours=23, minutes=45))
        print(f'Expandiendo temperatura a 15 min ({FECHA_INICIO_15MIN}..{fecha_fin_15min}) '
              f'-- misma malla que datahistorica/ ...')
        temperatura_15min = expandir_temperatura_a_15min(
            temperatura, fecha_inicio=FECHA_INICIO_15MIN, fecha_fin=str(fecha_fin_15min))
        ruta_temp_15min = out / 'temperatura' / (
            f'temperatura_15min_{ZONA_TEMPERATURA["etiqueta_zona"]}_'
            f'{ANIO_INICIO_CALENDARIO}_{ANIO_FIN_TEMPERATURA}.csv')
        temperatura_15min.to_csv(ruta_temp_15min, index=False)
        n_interp = int(temperatura_15min['interpolado'].sum())
        print('  ->', ruta_temp_15min, f'({len(temperatura_15min)} filas, '
              f'{n_interp} interpoladas de {len(temperatura_15min)})')
        estado_temperatura = (f'OK: horaria {len(temperatura)} filas '
                               f'({fecha_inicio_horaria}..{fecha_fin_horaria}); '
                               f'15min {len(temperatura_15min)} filas ({n_interp} interpoladas)')
    except RuntimeError as exc:
        estado_temperatura = f'ERROR: {exc}'
        print('  ERROR al descargar/expandir temperatura:', exc)
        ruta_temp_15min = None
        fecha_fin_15min = None

    if ruta_temp_15min is not None:
        print(f'Expandiendo calendario a 15 min ({FECHA_INICIO_15MIN}..{fecha_fin_15min}) '
              f'-- misma malla que datahistorica/ ...')
        calendario_15min = expandir_calendario_a_15min(
            calendario, fecha_inicio=FECHA_INICIO_15MIN, fecha_fin=str(fecha_fin_15min))
        ruta_cal_15min = out / 'calendario' / (
            f'feriados_15min_{ANIO_INICIO_CALENDARIO}_{ANIO_FIN_TEMPERATURA}.csv')
        calendario_15min.to_csv(ruta_cal_15min, index=False)
        print('  ->', ruta_cal_15min, f'({len(calendario_15min)} filas, misma malla que demanda)')

    print('Descargando PBI mensual BCRPData (', CODIGO_BCRP_PBI_MENSUAL, ') ...')
    try:
        pbi = plantilla_bcrp(CODIGO_BCRP_PBI_MENSUAL, f'{ANIO_INICIO_CALENDARIO}-1', '2026-12')
        ruta_pbi = out / 'economia' / f'pbi_mensual_bcrp_{ANIO_INICIO_CALENDARIO}_2026.csv'
        pbi.to_csv(ruta_pbi, index=False)
        estado_pbi = f'OK: {len(pbi)} filas ({pbi.periodo.iloc[0]}..{pbi.periodo.iloc[-1]})'
        print('  ->', ruta_pbi, f'({len(pbi)} filas)')
    except Exception as exc:  # noqa: BLE001 - se documenta el estado, no se detiene la generacion
        estado_pbi = f'ERROR: {exc}'
        print('  ERROR al descargar PBI:', exc)

    print('Generando población de Ica (2 puntos INEI observados + interpolación/extrapolación declarada) ...')
    poblacion = generar_poblacion_ica_interpolada(ANIO_INICIO_CALENDARIO, ANIO_FIN_TEMPERATURA)
    ruta_poblacion = out / 'poblacion' / f'poblacion_ica_{ANIO_INICIO_CALENDARIO}_{ANIO_FIN_TEMPERATURA}.csv'
    poblacion.to_csv(ruta_poblacion, index=False)
    print('  ->', ruta_poblacion, f'({len(poblacion)} filas,',
          int((poblacion.dato == "observado_inei").sum()), 'observadas de INEI)')

    generar_readme(out, estado_temperatura, fecha_fin_15min, estado_pbi)
    print('\nListo. Ver', out / 'README.md', 'para el detalle de fuentes y pendientes.')


def generar_readme(out: Path, estado_temperatura: str, fecha_fin_15min, estado_pbi: str) -> None:
    contenido = f"""# Variables externas (exógenas)

Generado por `tools/generar_variables_externas.py` el {pd.Timestamp.now().isoformat(timespec='seconds')}.
Implementa la sección 2 de `Plan_implementacion_demanda_Electro_Dunas.md` en lo que es
reproducible sin inventar datos ni asumir códigos de serie no confirmados.

## Implementado (datos reales en este folder)

| Carpeta | Archivo | Frecuencia | Cobertura |
|---|---|---|---|
| `calendario/` | `feriados_peru_{ANIO_INICIO_CALENDARIO}_{ANIO_FIN_CALENDARIO}.csv` | diaria | {ANIO_INICIO_CALENDARIO}–{ANIO_FIN_CALENDARIO} (horizonte completo, igual que `CFG['fin_plan']`) |
| `calendario/` | `feriados_15min_{ANIO_INICIO_CALENDARIO}_{ANIO_FIN_TEMPERATURA}.csv` | 15 min | {FECHA_INICIO_15MIN}..{fecha_fin_15min} — **misma malla que `datahistorica/`** |
| `temperatura/` | `temperatura_horaria_*.csv` | horaria | {FECHA_INICIO_15MIN[:10]}..hasta donde ERA5 tenga reanálisis disponible |
| `temperatura/` | `temperatura_15min_*.csv` | 15 min | {FECHA_INICIO_15MIN}..{fecha_fin_15min} — **misma malla que `datahistorica/`**, interpolada linealmente entre horas (columna `interpolado`). Estado: {estado_temperatura} |
| `economia/` | `pbi_mensual_bcrp_{ANIO_INICIO_CALENDARIO}_2026.csv` | mensual | BCRPData, serie `{CODIGO_BCRP_PBI_MENSUAL}` (PBI desestacionalizado, índice 2007=100), código confirmado por el usuario. Estado: {estado_pbi} |
| `poblacion/` | `poblacion_ica_{ANIO_INICIO_CALENDARIO}_{ANIO_FIN_TEMPERATURA}.csv` | anual (30 jun) | {ANIO_INICIO_CALENDARIO}–{ANIO_FIN_TEMPERATURA}. Solo 2022 y 2024 son observación oficial INEI; el resto está **interpolado/extrapolado linealmente** (columna `dato`) — ver limitación abajo. |

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
  SENAMHI**, y usa **una sola coordenada** (`{ZONA_TEMPERATURA["etiqueta_zona"]}`,
  lat={ZONA_TEMPERATURA['lat']}, lon={ZONA_TEMPERATURA['lon']}) para representar toda
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
"""
    (out / 'README.md').write_text(contenido, encoding='utf-8')

    (out / 'clientes' / 'README.md').write_text(
        "# Clientes mensuales (PENDIENTE)\n\n"
        "Fuente requerida: sistema comercial y padrón histórico de Electro Dunas "
        "(no es un dato público). Ver Plan_implementacion_demanda_Electro_Dunas.md, sección 2.\n\n"
        "Esquema esperado (sección 3 del plan), clave `alimentador, mes, segmento`:\n\n"
        "`alimentador, mes, segmento, activos, altas, bajas, fecha_disponible`\n\n"
        "No crear este archivo con ceros o supuestos: el plan exige registrar un pendiente "
        "como PENDIENTE, nunca como una observación de valor cero.\n",
        encoding='utf-8')

    (out / 'economia' / 'README.md').write_text(
        "# Economía nacional (PBI mensual, BCRPData)\n\n"
        "Fuente: BCRPData (<https://estadisticas.bcrp.gob.pe/estadisticas/series/mensuales/"
        "resultados/PN01773AM/html>) vía su API pública "
        "`.../api/[codigo]/json/[inicio]/[fin]/esp`.\n\n"
        "**Código confirmado por el usuario el 2026-09-17:** `PN01773AM` — "
        "\"PBI Desestacionalizado - mensual\" (índice 2007=100). Es PBI **nacional**, no "
        "regional de Ica (BCRP no publica PBI regional con frecuencia mensual) — se usa "
        "como proxy de actividad económica general, no como medida directa de la concesión.\n\n"
        "```python\n"
        "from prodma.variables_externas import CODIGO_BCRP_PBI_MENSUAL, plantilla_bcrp\n"
        "df = plantilla_bcrp(CODIGO_BCRP_PBI_MENSUAL, '2023-1', '2026-12')\n"
        "```\n\n"
        "Esquema (clave `serie_id, periodo`): `serie_id, periodo, valor, fuente, fecha_descarga`.\n",
        encoding='utf-8')

    (out / 'poblacion' / 'README.md').write_text(
        "# Población de Ica (INEI, 2 puntos observados + interpolación declarada)\n\n"
        "Fuente: INEI, \"Perú: Población Total Proyectada al 30 de Junio de cada año, según "
        "Departamento, Provincia y Distrito, 2018-2026\" "
        "(<https://www.gob.pe/institucion/inei/informes-publicaciones/6894980-"
        "peru-poblacion-total-proyectada-al-30-de-junio-de-cada-ano-segun-departamento-"
        "provincia-y-distrito-2018-2026>).\n\n"
        "**Solo se pudieron confirmar 2 cifras reales** (la tabla completa 2018-2026 del PDF "
        "oficial no fue extraíble automáticamente):\n\n"
        "- 2022-06-30: 1,020,050 habitantes (3.05% de la población proyectada del Perú)\n"
        "- 2024-06-30: 1,062,346 habitantes (3.12% de la población proyectada del Perú)\n\n"
        "El resto de los años (2023, 2025, 2026) se interpola/extrapola linealmente entre "
        "esos dos puntos vía `generar_poblacion_ica_interpolada()` "
        "(`src/prodma/variables_externas.py`), marcado explícitamente en la columna `dato` "
        "(`observado_inei` / `interpolado_lineal` / `extrapolado_lineal`) — nunca se "
        "presenta como observación oficial. Reemplazar por la serie completa de INEI en "
        "cuanto se pueda extraer del PDF oficial o de una fuente tabular confirmada.\n",
        encoding='utf-8')

    (out / 'produccion_regional' / 'README.md').write_text(
        "# Actividad económica y producción regional (PENDIENTE, solo PDF)\n\n"
        "Fuentes identificadas por el plan, sin API:\n\n"
        "- INEI, PBI por departamentos 2007-2024 (Ica): "
        "<https://www.gob.pe/institucion/inei/informes-publicaciones/7567597-peru-producto-bruto-interno-por-departamentos-2007-2024>\n"
        "- BCRP, Síntesis de actividad económica de Ica (informe mensual, sucursal Huancayo): "
        "<https://www.bcrp.gob.pe/docs/Sucursales/Huancayo/2026/presentacion-ica-02-2026.pdf>\n"
        "- MINEM, Boletín Estadístico Minero: "
        "<https://www.gob.pe/institucion/minem/informes-publicaciones/8031856-boletin-estadistico-minero-febrero-2026>\n\n"
        "El plan exige extraer y verificar cada tabla (unidades, continuidad, cobertura) "
        "manualmente antes de automatizar cualquier descarga; no concatenar variaciones "
        "interanuales como si fueran niveles, ni sumar toneladas de metales distintos como "
        "una sola variable.\n",
        encoding='utf-8')


if __name__ == '__main__':
    main()
