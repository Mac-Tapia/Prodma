"""Variables externas (exogenas) para las proyecciones de demanda.

Implementa lo que el plan de fuentes (Plan_implementacion_demanda_Electro_Dunas.md,
seccion 2) permite automatizar de forma reproducible y sin inventar datos:

- Calendario de feriados/dias no laborables de Peru (biblioteca `holidays`, offline).
- Temperatura horaria de reanalisis via la API publica de Open-Meteo.
- Version de ambas a 15 minutos, en la MISMA malla que `datahistorica/` (misma cantidad
  de filas que el dataset de demanda, 2023-01-01 hasta el mes mas reciente disponible).
- Climatologia (mes,hora) de temperatura para usar como exogena en TODO el horizonte
  del pronostico (incluido el recursivo hasta CFG['fin_plan'], donde no existe
  temperatura real futura).

- PBI mensual (BCRPData, serie PN01773AM, "PBI Desestacionalizado", indice 2007=100),
  confirmada por el usuario para reemplazar el codigo pendiente.
- Poblacion de Ica: solo 2 puntos son observacion oficial real (INEI, proyeccion al 30
  de junio de 2022 y 2024); el resto del rango 2023-2026 se interpola/extrapola
  linealmente y queda marcado explicitamente como tal (nunca como observado).

Numero de clientes y produccion minera regional NO se implementan aqui: el propio plan
los deja como fuentes internas (Electro Dunas) o publicaciones solo disponibles como PDF
(INEI/MINEM) que exigen extraccion y verificacion manual de cada tabla.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

DIAS_ES = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']


def generar_calendario_peru(anio_inicio: int, anio_fin: int) -> pd.DataFrame:
    """Calendario diario Peru: feriados, fin de semana y no_laborable.

    anio_fin es INCLUSIVE. Fuente offline (biblioteca `holidays`, paquete PyPI
    `holidays`), no una descarga de https://www.gob.pe/feriados: esa pagina no pudo
    leerse en la revision del plan, asi que la columna `norma` queda vacia y debe
    verificarse manualmente antes de un uso operativo (traslados de feriados por
    decreto especifico de un anio no siempre estan reflejados en la biblioteca).
    """
    if anio_fin < anio_inicio:
        raise ValueError('anio_fin debe ser >= anio_inicio')
    feriados = holidays.country_holidays('PE', years=range(anio_inicio, anio_fin + 1))
    idx = pd.date_range(f'{anio_inicio}-01-01', f'{anio_fin}-12-31', freq='D', name='fecha')
    cal = pd.DataFrame(index=idx)
    cal['dia_semana'] = [DIAS_ES[d] for d in idx.dayofweek]
    cal['feriado'] = [d.date() in feriados for d in idx]
    cal['nombre'] = [feriados.get(d.date(), '') for d in idx]
    cal['fin_semana'] = idx.dayofweek >= 5
    cal['no_laborable'] = cal['feriado'] | cal['fin_semana']
    cal['ambito'] = 'nacional'
    cal['norma'] = ''  # pendiente de verificacion contra https://www.gob.pe/feriados
    return cal.reset_index()


def descargar_temperatura_openmeteo(lat: float, lon: float, fecha_inicio: str, fecha_fin: str,
                                     etiqueta_zona: str, zona_horaria: str = 'America/Lima',
                                     timeout: int = 60) -> pd.DataFrame:
    """Temperatura horaria de reanalisis (ERA5, via Open-Meteo Historical Weather API).

    Es reanalisis, NO medicion directa de una estacion SENAMHI (el plan lo advierte
    explicitamente): se usa como aproximacion reproducible mientras no haya una
    descarga verificada de estaciones SENAMHI por zona de carga. `etiqueta_zona` debe
    documentar a que zona/alimentador representa esta coordenada; con una sola
    coordenada para todo Electro Dunas, el plan exige declarar que no se evaluo
    representatividad por alimentador (seccion 2 y 6).
    """
    parametros = {
        'latitude': lat, 'longitude': lon,
        'start_date': fecha_inicio, 'end_date': fecha_fin,
        'hourly': 'temperature_2m', 'timezone': zona_horaria,
    }
    url = 'https://archive-api.open-meteo.com/v1/archive?' + urllib.parse.urlencode(parametros)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(f'No se pudo descargar temperatura de Open-Meteo ({url}): {exc}') from exc
    horas = data['hourly']['time']
    temps = data['hourly']['temperature_2m']
    df = pd.DataFrame({
        'fecha_hora': pd.to_datetime(horas),
        'temperatura_c': temps,
    })
    df['zona'] = etiqueta_zona
    df['latitud'] = data.get('latitude', lat)
    df['longitud'] = data.get('longitude', lon)
    df['fuente'] = 'open-meteo_archive_era5'
    df['modelo'] = 'ERA5/ERA5-Land (reanalisis)'
    df['calidad'] = 'reanalisis_no_es_medicion_directa'
    return df


def expandir_temperatura_a_15min(temperatura: pd.DataFrame, fecha_inicio: str, fecha_fin: str,
                                  columna_valor: str = 'temperatura_c') -> pd.DataFrame:
    """Reindexa temperatura HORARIA a la misma malla de 15 min que `datahistorica/`
    (fin de intervalo, ej. `fecha_inicio='2023-01-01 00:15'`, `freq='15min'`), para que
    tenga la misma cantidad de filas/frecuencia que el dataset de demanda.

    Los valores en `:15/:30/:45` se obtienen por interpolacion temporal lineal entre las
    dos horas observadas mas cercanas (`:00` se conserva tal cual, sin interpolar). Cada
    fila queda marcada en `interpolado` -- el plan exige distinguir explicitamente lo
    interpolado de lo retenido, nunca presentarlo como una observacion nueva.
    """
    t = temperatura.copy()
    t['fecha_hora'] = pd.to_datetime(t['fecha_hora'])
    t = t.drop_duplicates('fecha_hora').set_index('fecha_hora').sort_index()
    idx_15min = pd.date_range(fecha_inicio, fecha_fin, freq='15min', name='fecha_hora')
    malla_unida = t.index.union(idx_15min)
    valor_interpolado = t[columna_valor].reindex(malla_unida).interpolate(method='time', limit_direction='both')
    salida = pd.DataFrame({columna_valor: valor_interpolado.reindex(idx_15min)}).reset_index()
    salida['interpolado'] = ~salida['fecha_hora'].isin(t.index)
    metadatos = t.drop(columns=[columna_valor], errors='ignore')
    for col in ('zona', 'latitud', 'longitud', 'fuente', 'modelo'):
        if col in metadatos.columns:
            salida[col] = metadatos[col].reindex(idx_15min, method='nearest').to_numpy()
    salida['calidad'] = np.where(salida['interpolado'], 'interpolado_lineal_desde_horario',
                                  'reanalisis_no_es_medicion_directa')
    return salida


def expandir_calendario_a_15min(calendario: pd.DataFrame, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    """Repite cada fila diaria del calendario en sus 96 intervalos de 15 min, sobre la
    misma malla que `datahistorica/`, para que tenga la misma cantidad de filas que el
    dataset de demanda. No agrega informacion nueva: es la misma fila del dia, repetida.
    """
    cal = calendario.copy()
    cal['fecha'] = pd.to_datetime(cal['fecha']).dt.date
    cal = cal.set_index('fecha')
    idx_15min = pd.date_range(fecha_inicio, fecha_fin, freq='15min', name='fecha_hora')
    salida = cal.loc[idx_15min.date].reset_index(drop=True)
    salida.insert(0, 'fecha_hora', idx_15min)
    return salida


def construir_climatologia_mensual_horaria(temperatura: pd.DataFrame,
                                            columna_valor: str = 'temperatura_c') -> pd.Series:
    """Climatologia (mes, hora) -> mediana de temperatura, sobre observaciones HORARIAS
    reales (no usar la version interpolada a 15 min: solo agregaria ruido de la
    interpolacion sin aportar informacion nueva a una climatologia por hora).

    Uso previsto: 'temperatura esperada' para CUALQUIER fecha, pasada o futura -- un
    supuesto climatologico declarado, nunca un pronostico meteorologico real. Debe
    aplicarse de forma CONSISTENTE en entrenamiento y en el horizonte futuro (no mezclar
    temperatura observada en entrenamiento con climatologia solo en el futuro: eso
    filtraria al modelo una señal real que no existe de la misma forma en la proyeccion).
    """
    t = temperatura.copy()
    t['fecha_hora'] = pd.to_datetime(t['fecha_hora'])
    return t.groupby([t['fecha_hora'].dt.month, t['fecha_hora'].dt.hour])[columna_valor].median()


def temperatura_climatologica(indice: pd.DatetimeIndex, climatologia: pd.Series) -> pd.Series:
    """Mapea la climatologia (mes,hora)->mediana a cualquier indice de fechas (incluido
    un horizonte futuro fuera de la cobertura real de `variables_externas/temperatura/`)."""
    claves = pd.MultiIndex.from_arrays([indice.month, indice.hour])
    valores = pd.Series(claves.map(climatologia), index=indice, dtype=float)
    faltantes = valores.isna()
    if faltantes.any():
        valores.loc[faltantes] = float(climatologia.median())
    return valores


_MESES_ES = {'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12}


def cargar_pbi_mensual(ruta_csv) -> pd.Series:
    """Lee `pbi_mensual_bcrp_*.csv` (columnas serie_id,periodo,valor,...; periodo tipo
    'Ene.2023') y devuelve una Serie indexada por el primer dia de cada mes."""
    df = pd.read_csv(ruta_csv)
    partes = df['periodo'].str.extract(r'(?P<mes>[A-Za-z]{3})\.(?P<anio>\d{4})')
    mes_num = partes['mes'].str.lower().map(_MESES_ES)
    fecha = pd.to_datetime(partes['anio'] + '-' + mes_num.astype(str) + '-01')
    serie = pd.Series(df['valor'].to_numpy(dtype=float), index=fecha).sort_index()
    serie.index.name = 'mes'
    return serie


def pbi_en_fecha(indice: pd.DatetimeIndex, pbi_mensual: pd.Series) -> pd.Series:
    """Mapea el PBI mensual a cualquier indice de fechas. Para fechas posteriores a la
    ultima observacion real (BCRP publica con rezago; el horizonte de pronostico llega
    a CFG['fin_plan'], años despues del ultimo dato), se mantiene el ULTIMO valor
    observado constante -- una proyeccion economica real esta fuera de alcance de este
    proyecto; mantener el ultimo nivel conocido es la asuncion mas simple y declarada,
    no una prediccion de crecimiento economico futuro."""
    mes_de_cada_fecha = indice.to_period('M').to_timestamp()
    valores = pd.Series(mes_de_cada_fecha.map(pbi_mensual), index=indice, dtype=float)
    ultimo_valor = float(pbi_mensual.iloc[-1])
    valores = valores.fillna(ultimo_valor)
    return valores


def poblacion_en_fecha(indice: pd.DatetimeIndex, poblacion_anual: pd.Series) -> pd.Series:
    """Mapea poblacion anual (indexada por año, ej. de generar_poblacion_ica_interpolada)
    a cualquier indice de fechas: cada fecha toma el valor de SU año calendario. Para
    años fuera del rango generado, usa el mas cercano (el primero o el ultimo)."""
    anios_disponibles = poblacion_anual.index.to_numpy()
    anio_de_cada_fecha = np.clip(indice.year, anios_disponibles.min(), anios_disponibles.max())
    return pd.Series(poblacion_anual.reindex(anio_de_cada_fecha).to_numpy(), index=indice, dtype=float)


CODIGO_BCRP_PBI_MENSUAL = 'PN01773AM'
"""PBI Desestacionalizado - mensual (indice 2007=100). Confirmado por el usuario el
2026-09-17 (no es un valor por defecto elegido sin confirmacion); verificar vigencia en
https://estadisticas.bcrp.gob.pe/estadisticas/series/mensuales/resultados/PN01773AM/html
si cambia la fuente."""


def plantilla_bcrp(codigo_serie: str, fecha_inicio: str, fecha_fin: str, timeout: int = 60) -> pd.DataFrame:
    """Descarga una serie del API de BCRPData ya con un codigo CONFIRMADO por el usuario.

    El plan (seccion 2) es explicito: el codigo de serie (PBI real, indice de actividad,
    etc.) esta "pendiente de seleccion" y no debe asumirse. Esta funcion NO elige un
    codigo por defecto: se debe pasar `codigo_serie` ya verificado en
    https://estadisticas.bcrp.gob.pe/estadisticas/series/ para evitar reportar una
    serie equivocada como si fuera la exigida por el estudio. Para el PBI mensual usar
    `CODIGO_BCRP_PBI_MENSUAL` (confirmado por el usuario).
    """
    url = (f'https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{codigo_serie}'
           f'/json/{fecha_inicio}/{fecha_fin}/esp')
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.loads(resp.read())
    periodos = data.get('periods', [])
    filas = [dict(serie_id=codigo_serie, periodo=p.get('name'),
                  valor=p.get('values', [None])[0]) for p in periodos]
    df = pd.DataFrame(filas)
    df['fuente'] = 'BCRPData'
    df['fecha_descarga'] = pd.Timestamp.now().isoformat()
    return df


# Poblacion proyectada de Ica al 30 de junio -- SOLO estos dos anios son observacion
# oficial real de INEI (proyecciones de poblacion por departamento). Fuente: INEI,
# "Peru: Poblacion Total Proyectada al 30 de Junio de cada año, según Departamento,
# Provincia y Distrito, 2018-2026" (gob.pe/institucion/inei/informes-publicaciones/
# 6894980-...), cifras reportadas en prensa a partir del informe INEI:
#   2022-06-30: 1,020,050 habitantes (3.05% de la poblacion proyectada del Peru)
#   2024-06-30: 1,062,346 habitantes (3.12% de la poblacion proyectada del Peru)
# NO se pudo extraer de forma automatica el resto de la serie 2018-2026 del PDF oficial
# (tabla no extraible por herramientas automaticas); no se debe rellenar 2023/2025/2026
# con una cifra inventada.
POBLACION_ICA_OBSERVADA = {
    '2022-06-30': 1_020_050,
    '2024-06-30': 1_062_346,
}


def generar_poblacion_ica_interpolada(anio_inicio: int, anio_fin: int) -> pd.DataFrame:
    """Poblacion de Ica al 30 de junio de cada año, `anio_inicio..anio_fin` inclusive.

    Solo 2022 y 2024 son observacion oficial real de INEI (ver POBLACION_ICA_OBSERVADA);
    el resto de los años se interpola (entre 2022 y 2024) o se extrapola (fuera de ese
    rango) linealmente con la tasa de crecimiento anual observada entre esos dos puntos,
    y queda marcado explicitamente en la columna `dato` -- nunca se presenta como
    observacion oficial. Reemplazar por la serie completa de INEI en cuanto se pueda
    extraer del PDF oficial o de una fuente tabular confirmada.
    """
    anios_obs = sorted(int(f[:4]) for f in POBLACION_ICA_OBSERVADA)
    valores_obs = [POBLACION_ICA_OBSERVADA[f'{a}-06-30'] for a in anios_obs]
    tasa_anual = (valores_obs[-1] - valores_obs[0]) / (anios_obs[-1] - anios_obs[0])
    filas = []
    for anio in range(anio_inicio, anio_fin + 1):
        fecha = f'{anio}-06-30'
        if fecha in POBLACION_ICA_OBSERVADA:
            filas.append(dict(fecha=fecha, poblacion=POBLACION_ICA_OBSERVADA[fecha], dato='observado_inei'))
        else:
            estimado = round(valores_obs[0] + tasa_anual * (anio - anios_obs[0]))
            dato = 'interpolado_lineal' if anios_obs[0] < anio < anios_obs[-1] else 'extrapolado_lineal'
            filas.append(dict(fecha=fecha, poblacion=estimado, dato=dato))
    df = pd.DataFrame(filas)
    df['fuente'] = 'INEI (2 puntos observados) + interpolacion/extrapolacion lineal declarada'
    return df
