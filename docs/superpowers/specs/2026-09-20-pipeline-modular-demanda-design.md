# Pipeline modular de demanda electrica - Especificacion de diseno

## 1. Objetivo

Transformar `notebook/proyecciondemanda.ipynb` en una interfaz ligera sobre un
pipeline Python reproducible, reanudable y auditable. El flujo tendra cinco
etapas independientes: ingesta, limpieza y estadistica descriptiva,
entrenamiento, evaluacion y seleccion, y prediccion final.

La implementacion debe corregir el defecto actual por el que la limpieza se
exporta pero el entrenamiento vuelve a consumir el dato crudo. El dato limpio
publicado por la etapa 2 sera la unica entrada admisible de las etapas 3 a 5.

## 2. Restricciones globales

- Los Excel originales de `datahistorica/` y `nuevosclientes/` son de solo
  lectura.
- Python 3.11 y el entorno local `.venv` siguen siendo la plataforma objetivo.
- La frecuencia canonica es 15 minutos y la zona de negocio es
  `America/Lima`.
- Ninguna transformacion puede consultar valores posteriores al instante que
  transforma cuando se use para entrenamiento o evaluacion.
- Imputar contexto no convierte una fila en objetivo observado. Las mascaras
  `observado`, `imputado` y `metodo_imputacion` se conservaran hasta la
  evaluacion.
- Los modelos no se seleccionaran por ajuste dentro de muestra ni por una sola
  prueba estadistica.
- No se afirmara validez para un horizonte que no haya sido evaluado mediante
  backtesting del mismo horizonte.
- Los errores de una serie o modelo quedan registrados y no se ocultan; solo
  continuan las unidades independientes que conserven entradas validas.
- Cada etapa escribe exclusivamente dentro de su directorio de ejecucion.

## 3. Interfaz y ciclo de ejecucion

La interfaz principal sera:

```powershell
.\.venv\Scripts\python.exe -m prodma.pipeline run --config config/default.yaml
.\.venv\Scripts\python.exe -m prodma.pipeline stage 02_limpieza --run-id <id>
.\.venv\Scripts\python.exe -m prodma.pipeline resume --run-id <id>
.\.venv\Scripts\python.exe -m prodma.pipeline status --run-id <id>
```

El notebook no definira funciones productivas. Solo permitira editar una
configuracion, invocar la misma API publica que usa la CLI, leer manifiestos y
mostrar tablas o figuras ya producidas.

Cada ejecucion tiene un `run_id` inmutable y un manifiesto raiz. Una etapa se
puede reutilizar solamente si coinciden los hashes de entradas, la
configuracion relevante y la version de su contrato. La escritura del
manifiesto y de archivos finales sera atomica.

## 4. Estructura de artefactos

```text
resultados/<run_id>/
|-- manifest.json
|-- config_resuelta.json
|-- 01_ingesta/
|   |-- parquet/
|   |-- auditoria/
|   `-- logs/
|-- 02_limpieza/
|   |-- datos/
|   |-- csv/
|   |-- figuras/
|   |-- estadistica_descriptiva/
|   |-- auditoria/
|   `-- logs/
|-- 03_entrenamiento/
|   |-- modelos/
|   |-- checkpoints/
|   |-- figuras/
|   |-- csv/
|   |-- times/
|   |-- traces/
|   `-- logs/
|-- 04_evaluacion/
|   |-- metricas/
|   |-- inferencia_parametrica/
|   |-- inferencia_no_parametrica/
|   |-- diagnosticos/
|   |-- ranking/
|   |-- figuras/
|   `-- logs/
`-- 05_prediccion/
    |-- pronosticos/
    |-- intervalos/
    |-- escenarios/
    |-- figuras/
    |-- entregables/
    `-- logs/
```

Los CSV, figuras, checkpoints, tiempos y trazas del entrenamiento no se
mezclaran con los de evaluacion o prediccion.

## 5. Etapa 1: ingesta estandarizada

### Responsabilidad

Leer `.xls` y `.xlsx`, descubrir hojas, resolver el esquema, normalizar fechas
y publicar un parquet por libro y medidor sin modificar valores de medicion.

### Controles

- Inventario de archivos con SHA-256, tamano y fecha.
- Encabezados obligatorios, alias declarados y primera columna coincidente.
- Fecha valida, alineacion a 15 minutos con tolerancia configurable y registro
  de marcas rechazadas o redondeadas.
- Duplicados coincidentes conservados; contradicciones convertidas en nulo y
  auditadas.
- Convencion de intervalo, zona horaria, unidad y disponibilidad de P, Q, S,
  tension, corriente y frecuencia.

### Salida contractual

`01_ingesta/parquet/<medidor>/<anio>.parquet`, un catalogo de series y un
informe de calidad de ingesta. Esta etapa no imputa, recorta ni consolida.

## 6. Etapa 2: limpieza, matriz y estadistica descriptiva

### Responsabilidad

Unir los anos de cada medidor, aplicar reglas causales y fisicas, y construir
la unica matriz consolidada limpia admitida por el resto del pipeline.

### Representacion de datos

Por variable se preservaran como minimo:

- `<variable>_observado`: valor original valido.
- `<variable>_contexto`: valor causalmente imputado para crear rezagos.
- `<variable>_limpio`: observado corregido solo cuando una regla aprobada lo
  marque como error instrumental.
- `<variable>_es_observado`, `<variable>_es_imputado`,
  `<variable>_metodo_imputacion`, `<variable>_es_atipico` y
  `<variable>_es_recortado`.

El objetivo de entrenamiento sera `<variable>_limpio` restringido a
`<variable>_es_observado=True`. Los valores imputados solo podran alimentar
rezagos y ventanas historicas.

### Limpieza temporal y de medicion

- Reemplazo de infinitos por nulo y malla completa de 15 minutos.
- Relleno hacia adelante limitado y sustitucion por semana previa observada,
  ambos marcados y configurables.
- Deteccion robusta de picos mediante mediana y MAD moviles calculadas solo
  con pasado. La clasificacion punto aislado/cambio sostenido no utilizara
  ventanas centradas ni observaciones futuras.
- Deteccion separada de cambio de nivel, sensor congelado, huecos, secuencias
  de cero y variacion casi nula.
- Reglas electricas configurables: finitud, rangos de tension/frecuencia,
  coherencia `S >= abs(P)`, `S >= abs(Q)`, factor de potencia y discrepancia
  entre S medida y `hypot(P,Q)`. Una violacion se marca antes de decidir si es
  corregible; no se inventan limites universales sin metadatos del medidor.

### Estadistica descriptiva

Se producira por medidor, variable, ano y particion relevante:

- conteos observados, faltantes e imputados y longitud de huecos;
- media, mediana, desviacion, MAD, minimo, maximo, cuantiles, IQR y
  coeficiente de variacion cuando sea valido;
- ceros, negativos, atipicos, recortes, cambios de nivel y sensor congelado;
- cobertura temporal, factor de carga, energia integrada cuando la unidad lo
  permita y coherencia P-Q-S;
- histogramas, boxplots, curvas crudo/limpio, mapa de faltantes y perfiles
  diarios, semanales y mensuales.

### Salida contractual

`02_limpieza/datos/matriz_limpia.parquet`, catalogo de variables, mascaras de
calidad, descriptivos y auditoria fila a fila de toda correccion. El manifiesto
incluira el hash de la matriz limpia.

## 7. Etapa 3: prefiltro y entrenamiento

### Candidatos

El registro inicial incluira referencias estacionales y los modelos ya
soportados: Ridge, Random Forest, XGBoost, LightGBM, CatBoost, MLP, ARIMA,
SARIMA/SARIMAX, Holt-Winters, Prophet y LSTM. Cada candidato declarara
resolucion, variables requeridas, soporte de exogenas, backend y recursos.

No todos los modelos tienen que entrenarse a 15 minutos: los modelos clasicos
podran usar una resolucion justificada, pero sus resultados solo seran
comparables dentro del mismo objetivo, horizonte y resolucion de evaluacion.

### Prefiltro

Antes de la optimizacion costosa, cada candidato ejecutara un smoke fit y un
backtest reducido. Queda `APROBADO`, `RECHAZADO` o `NO_DISPONIBLE` usando reglas
configuradas: excepciones, predicciones no finitas, tiempo o memoria maximos,
historia insuficiente y desempeno claramente inferior a la referencia.

El prefiltro nunca consulta el test final. Los modelos aprobados pasan al
entrenamiento completo con seleccion de variables e hiperparametros dentro de
validacion temporal anidada o un esquema equivalente que mantenga aislada la
prueba final.

### Artefactos exclusivos

- `modelos/`: modelos finales serializados y metadatos.
- `checkpoints/`: estados reanudables de busqueda y entrenamiento.
- `figuras/`: aprendizaje, convergencia e importancia de variables.
- `csv/`: ensayos, parametros y estado del prefiltro.
- `times/`: tiempo por fase/modelo y uso de recursos.
- `traces/`: historial de Optuna, curvas de perdida y eventos estructurados.
- `logs/`: advertencias y excepciones completas.

Un checkpoint solo se reutilizara si coinciden matriz, particiones,
configuracion, columnas, semilla, biblioteca y backend.

## 8. Etapa 4: evaluacion estadistica y seleccion

### Evaluacion predictiva

Se emplearan cortes cronologicos y origen rodante con horizontes iguales a los
que se pretenda operar. Se calcularan RMSE, MAE, WAPE, MASE/RMSSE, sesgo,
error de energia, pico y hora del pico. MAPE o R2 se publicaran solo cuando
sean interpretables.

### Diagnosticos e inferencia

- Residuos: ACF/PACF, Ljung-Box, Durbin-Watson como diagnostico auxiliar,
  heterocedasticidad y estabilidad temporal.
- Pruebas parametricas solo cuando sus supuestos sean defendibles, incluyendo
  comparaciones pareadas de errores y modelos lineales de referencia.
- Pruebas no parametricas: Wilcoxon pareado para dos modelos; Friedman para
  multiples modelos sobre los mismos bloques, con post-hoc y correccion por
  comparaciones multiples cuando corresponda.
- Diebold-Mariano o una variante compatible se usara para diferencias de
  precision predictiva por horizonte, verificando dependencia serial y
  longitud suficiente.

Los p-valores se acompanaran de tamano de efecto, intervalos y numero de
observaciones. Fallar una prueba de normalidad no descarta automaticamente un
modelo; la inferencia es evidencia complementaria.

### Seleccion final

La seleccion sera por medidor, variable y horizonte. Primero exige integridad,
finitud, reproducibilidad y mejora fuera de muestra frente a referencias;
despues usa una regla multicriterio versionada con error central, sesgo,
estabilidad, costo y complejidad. Los empates favorecen el modelo mas simple.
El ranking conservara todos los candidatos, causas de exclusion y evidencia.

## 9. Etapa 5: prediccion final

El ganador se reentrenara con todos los objetivos observados disponibles,
manteniendo las transformaciones ajustadas solo con ese historial. La
prediccion empezara exactamente despues de la ultima observacion elegible, sin
solapamientos ni huecos fabricados.

Se separaran:

- pronostico operativo para horizontes validados;
- planeamiento de largo plazo mediante perfiles y supuestos explicitamente
  versionados;
- incorporacion de clientes confirmados y escenarios.

Los intervalos solo se denominaran predictivos si su cobertura fue calibrada
fuera de muestra. En otro caso se etiquetaran como bandas exploratorias. Cada
archivo indicara modelo, version, origen, horizonte, unidad, escenario y
calidad.

## 10. Modulos Python objetivo

```text
src/prodma/
|-- config.py
|-- artifacts.py
|-- pipeline.py
|-- stages/
|   |-- ingestion.py
|   |-- cleaning.py
|   |-- training.py
|   |-- evaluation.py
|   `-- prediction.py
|-- ingesta/excel.py
|-- limpieza/
|   |-- temporal.py
|   |-- missing.py
|   |-- outliers.py
|   |-- electrical.py
|   `-- descriptive.py
|-- modeling/
|   |-- registry.py
|   |-- features.py
|   |-- prefilter.py
|   |-- train.py
|   `-- checkpoints.py
|-- evaluacion/
|   |-- backtesting.py
|   |-- metrics.py
|   |-- inference.py
|   `-- ranking.py
`-- prediccion/
    |-- operational.py
    |-- planning.py
    `-- exports.py
```

Los modulos existentes se migraran gradualmente o se convertiran en
adaptadores; no se mantendran dos implementaciones productivas de una misma
regla.

## 11. Manejo de errores y estados

Cada etapa termina en `PENDING`, `RUNNING`, `SUCCESS`, `PARTIAL`, `FAILED` o
`SKIPPED`. Una etapa dependiente no arranca si falta un artefacto contractual.
Los errores estructurales de entrada detienen la ejecucion; un modelo aislado
puede fallar sin tumbar los restantes. Una prediccion no se publica si no hay
ganador valido.

## 12. Estrategia de pruebas

- Unitarias para cada regla temporal, fisica, estadistica y de rutas.
- Pruebas explicitas de no fuga: alterar el futuro no cambia ninguna salida
  anterior al origen.
- Pruebas de contrato entre etapas y de aislamiento de carpetas.
- Pruebas de regresion para que el entrenamiento consuma el hash exacto de
  `matriz_limpia.parquet`.
- Integracion con libros sinteticos `.xls/.xlsx` y, cuando esten disponibles,
  smoke tests de solo lectura sobre los libros reales.
- Pruebas de reanudacion, checkpoint incompatible, escritura atomica y fallo
  parcial de un modelo.
- Aceptacion de CLI y ejecucion del notebook ligero sin estado heredado.

La implementacion seguira ciclos RED-GREEN-REFACTOR. No se considerara
completa hasta ejecutar la suite total y un pipeline smoke de extremo a
extremo.

## 13. Migracion

1. Crear contratos de configuracion, artefactos y manifiestos.
2. Integrar la ingesta modular existente como etapa 1.
3. Extraer y corregir la limpieza, publicar la matriz unica y sus descriptivos.
4. Extraer features, registro, prefiltro y entrenamiento.
5. Extraer evaluacion, inferencia y ranking.
6. Extraer prediccion y exportaciones.
7. Reducir el notebook a consumidor de la API.
8. Retirar las celdas duplicadas solo despues de que las pruebas de paridad y
   el smoke completo sean satisfactorios.

No se borraran resultados historicos ni cambios locales del usuario.

## 14. Criterios de aceptacion

- Cada etapa escribe unicamente en su carpeta y posee manifiesto verificable.
- La etapa 2 produce la unica matriz limpia y el entrenamiento demuestra por
  hash que la consumio.
- Ninguna imputacion se puntua como objetivo observado.
- Las pruebas demuestran ausencia de consulta futura en limpieza y features.
- El entrenamiento guarda por separado figuras, checkpoints, CSV, tiempos,
  trazas, modelos y logs.
- El prefiltro y entrenamiento completo usan conjuntos temporales distintos
  del test final.
- Evaluacion publica metricas, diagnosticos, inferencia parametrica y no
  parametrica, ranking y causas de exclusion.
- La etapa 5 solo publica series con ganador valido y horizonte respaldado.
- El notebook no contiene implementaciones duplicadas.
- La suite completa y el smoke end-to-end terminan sin fallos.

## 15. Sustento cientifico

- Tashman, L. J. (2000), *Out-of-sample tests of forecasting accuracy: an
  analysis and review*, International Journal of Forecasting,
  https://doi.org/10.1016/S0169-2070(00)00065-0.
- Cerqueira, V., Torgo, L. y Mozetic, I. (2020), *Evaluating time series
  forecasting models: an empirical study on performance estimation methods*,
  Machine Learning, https://doi.org/10.1007/s10994-020-05867-9.
- Hong, T. (2010), *Short Term Electric Load Forecasting*, tesis doctoral,
  North Carolina State University,
  https://repository.lib.ncsu.edu/items/c8f6d98e-3422-4a0b-9d5e-91cfeaa0de87.
- Hasanov, M. (2022), *Load forecasting using machine learning for smart meter
  data processing*, tesis de maestria, disponible en
  https://www.researchgate.net/publication/363108666.
- Meng, Y. et al. (2023), *Short-term electricity load forecasting based on a
  novel data preprocessing system and data reconstruction strategy*, Journal
  of Building Engineering, https://doi.org/10.1016/j.jobe.2023.107432.
- Hewamalage, H. et al. (2023), *Forecast evaluation for data scientists:
  common pitfalls and best practices*, Data Mining and Knowledge Discovery,
  https://doi.org/10.1007/s10618-022-00894-5.

Estas referencias justifican la evaluacion fuera de muestra, la preservacion
del orden temporal, la comparacion contra referencias y el tratamiento
explicito de faltantes y atipicos. No sustituyen la validacion empirica sobre
los datos reales de Electro Dunas.
