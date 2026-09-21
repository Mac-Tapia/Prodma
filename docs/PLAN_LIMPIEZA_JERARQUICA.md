# Plan metodologico de limpieza jerarquica e imputacion de demanda

## 1. Objetivo

Construir un dataset de demanda de 15 minutos completo para analisis y
pronostico, preservando las mediciones originales, sin eliminar atipicos y sin
presentar valores estimados como observaciones reales.

Toda decision adoptada debe registrarse en `docs/evidencia_cientifica.yaml` con,
como minimo, un articulo indexado, una tesis de maestria y una tesis doctoral,
fuentes en dos idiomas, el codigo, la prueba automatizada y el artefacto que la
demuestra. Una referencia bibliografica no sustituye la validacion sobre los
datos reales del proyecto.

La limpieza no consistira en aplicar todos los algoritmos disponibles. Cada
serie y cada hueco recibiran el metodo que demuestre menor error en una
validacion con datos conocidos ocultados artificialmente.

## 2. Productos separados

Se mantendran cuatro productos con funciones distintas:

1. `datos_observados.parquet`: datos originales normalizados, inmutables.
2. `matriz_historica_limpia.parquet`: medidores establecidos, con P/Q/S y
   reconstrucciones aceptadas.
3. `matriz_medidores_recientes.parquet`: medidores incorporados en el ultimo
   ano, sin rellenar fechas anteriores a su puesta en servicio.
4. `matriz_calidad.parquet`: mascara observada/estimada, metodo, incertidumbre,
   atipicos, cambios de nivel y controles electricos.

Los medidores sin ninguna observacion util se registraran en
`medidores_sin_datos.csv` y no se imputaran sin una fuente externa aprobada.

## 3. Taxonomia obligatoria por celda

Cada punto P/Q/S tendra:

- `valor_original`;
- `valor_limpio`;
- `es_observado`;
- `es_estimado`;
- `metodo_imputacion`;
- `longitud_hueco`;
- `causa_probable`;
- `es_atipico`;
- `es_cambio_nivel`;
- `incertidumbre_inferior` e `incertidumbre_superior`;
- `apto_como_objetivo`.

Solo `es_observado=True` podra entrar como objetivo de entrenamiento o como
valor real al calcular metricas.

## 4. Fase A: inventario y sanidad temporal

1. Verificar SHA-256, ano, medidor, unidad, intervalo y zona horaria.
2. Alinear a la malla de 15 minutos dentro de la tolerancia aprobada.
3. Conservar duplicados coincidentes y convertir contradicciones en faltantes
   auditados.
4. Determinar para cada medidor primera y ultima observacion real.
5. No crear valores antes de la puesta en servicio ni despues del retiro.
6. Clasificar huecos por longitud y posicion: interior, inicial, final o serie
   completamente ausente.

## 5. Fase B: segmentacion de medidores

### Establecidos

Medidores presentes antes del ultimo ano y con P o Q observada. Entran en la
competencia de imputacion y posteriormente en entrenamiento.

### Recientes

Medidores cuya primera aparicion corresponde al ultimo ano disponible. Se
aislan en su propio dataset. No se reconstruye historia anterior. Se podran
pronosticar cuando acumulen la cobertura minima definida por backtesting.

### Sin datos

Medidores presentes en libros pero con P y Q completamente vacias. Se ponen en
cuarentena. Solo se estimaran mediante topologia, capacidad, medidor analogo o
otra evidencia operativa aprobada; dichas estimaciones nunca se marcaran como
observadas.

### Retirados o renombrados

Los identificadores que no aparecen en el catalogo vigente quedan fuera de la
matriz operativa. Un renombre solo se unifica mediante una tabla de identidad
confirmada, nunca por similitud textual automatica.

## 6. Fase C: atipicos sin eliminacion

1. Conservar siempre el valor original.
2. Detectar candidatos mediante mediana y MAD moviles robustas.
3. Usar solamente pasado para el dataset de pronostico; no usar ventanas
   centradas.
4. Separar picos aislados, cambio de nivel, secuencia congelada, cero sostenido
   y violacion electrica.
5. No clasificar un cambio sostenido como atipico puntual.
6. Crear dos vistas:
   - `observado_original`, para auditoria;
   - `valor_modelable`, corregido solo cuando la evidencia indique error de
     medicion.
7. Comparar el pronostico con y sin correccion mediante backtesting. Si la
   correccion no mejora el resultado o altera energia injustificadamente, se
   conserva el valor original como modelable.

## 7. Fase D: jerarquia de imputacion

La longitud se mide en intervalos de 15 minutos.

### Nivel 0: sin imputacion

- Valores observados validos.
- Fechas fuera de la vida operativa del medidor.
- Series completamente vacias sin donante aprobado.

### Nivel 1: hueco de 1 a 4 intervalos

Comparar:

- arrastre causal limitado;
- interpolacion lineal para reconstruccion historica;
- interpolacion PCHIP cuando los bordes y la forma sean suficientes.

La interpolacion que use el extremo futuro no se utilizara durante una
simulacion de pronostico en tiempo real.

### Nivel 2: hueco de 5 a 96 intervalos

Comparar:

- mediana del mismo bloque de 15 minutos en dias equivalentes;
- perfil del mismo dia de semana de las cuatro semanas anteriores;
- KNN de dias semejantes usando contexto anterior, calendario y temperatura;
- copy-paste de perfiles homologos con ajuste robusto de nivel.

### Nivel 3: hueco de 97 a 672 intervalos

Comparar:

- KNN por perfiles semanales;
- regresion robusta con calendario, temperatura y medidores correlacionados;
- modelos ARIMA/SARIMAX entrenados solo con informacion disponible;
- descomposicion estacional y reconstruccion del residuo.

### Nivel 4: hueco mayor de una semana

Comparar:

- factorizacion matricial de bajo rango;
- donantes del mismo cluster de forma de carga;
- modelos multivariantes que utilicen medidores correlacionados y exogenas;
- imputacion multiple para representar incertidumbre.

No se aceptara un metodo de hueco largo por haber completado todas las celdas;
debe superar las referencias en bloques artificiales equivalentes.

## 8. Fase E: coherencia electrica P-Q-S

1. Validar unidades y convencion de signos.
2. Conservar `S_medida` separada de `S_calculada = hypot(P,Q)`.
3. Si S medida falta, `S_calculada` puede publicarse como derivada, no como
   medicion imputada.
4. Comprobar `S >= abs(P)`, `S >= abs(Q)` y factor de potencia admisible.
5. No reconstruir el signo de Q a partir de P y S sin una regla operativa
   confirmada.
6. Medir el error de balance electrico antes y despues de cada imputacion.

## 9. Seleccion cientifica del metodo

Para cada medidor, variable y clase de hueco:

1. Elegir periodos completos que representen meses, dias laborables,
   fines de semana, horas punta y valle.
2. Ocultar bloques conocidos con longitudes 1, 4, 16, 96, 672 y superiores.
3. Repetir el experimento en varios origenes cronologicos.
4. Ajustar cada metodo exclusivamente con el tramo permitido.
5. Medir MAE, RMSE, WAPE, sesgo, error de energia, error de pico y violaciones
   P-Q-S.
6. Aplicar Friedman para multiples metodos y comparaciones pareadas con
   Wilcoxon y correccion de Holm cuando corresponda.
7. Reportar tamano de efecto e intervalos, no solamente p-valores.
8. Elegir el metodo mas simple cuyo error y estabilidad cumplan los umbrales.
9. Si ningun metodo supera la referencia, conservar el faltante y declarar la
   serie no apta para ese uso.

## 10. Prevencion de fuga temporal

- Las decisiones para el instante `t` del dataset de pronostico solo usan
  informacion disponible antes de `t`.
- Escaladores, KNN, clusters, descomposiciones y modelos de imputacion se
  ajustan nuevamente dentro de cada fold temporal.
- El test final no participa en seleccion de metodo ni hiperparametros.
- La reconstruccion historica bidireccional se guarda como producto distinto y
  no se utiliza para simular disponibilidad operacional.

## 11. Incertidumbre y trazabilidad

Cada imputacion tendra intervalo o dispersion empirica calculada con los
bloques artificiales. Las tablas de salida incluiran version del algoritmo,
parametros, fecha, hash de entrada, donantes utilizados y error esperado para
esa clase de hueco.

Se publicaran por separado:

- porcentaje observado e imputado;
- cobertura por medidor y variable;
- error validado por metodo y longitud de hueco;
- energia agregada antes y despues;
- numero de atipicos conservados y corregidos;
- medidores recientes y sin datos;
- causas de rechazo para entrenamiento.

## 12. Criterios de aceptacion

- Cero eliminaciones silenciosas de atipicos.
- Cero valores anteriores a la puesta en servicio de medidores recientes.
- Cero valores imputados tratados como objetivos observados.
- Cien por ciento de las imputaciones con metodo y mascara.
- Cero fugas en pruebas de invariancia al modificar el futuro.
- Metodos elegidos mediante bloques ocultos comparables al hueco real.
- Balance P-Q-S dentro de tolerancia declarada.
- Dataset principal, recientes, sin datos y calidad almacenados por separado.
- Reejecucion determinista con hashes y manifiesto.

## 13. Orden de implementacion

1. Congelar inventario y clasificacion de medidores.
2. Extraer catalogos de huecos y causas.
3. Implementar referencias simples y pruebas de ocultamiento.
4. Implementar KNN de perfiles y copy-paste estacional.
5. Implementar regresion multivariante y factorizacion matricial.
6. Incorporar coherencia P-Q-S e incertidumbre.
7. Ejecutar competencia por serie/clase de hueco.
8. Publicar matrices separadas y reporte metodologico.
9. Habilitar para entrenamiento solo series que superen cobertura y error.

## 14. Sustento cientifico

- Wang, Tsai y Lin (2021), comparacion de interpolacion lineal, ARIMA, KNN,
  MLP y SVR para datos electricos faltantes, *Expert Systems with
  Applications*, https://doi.org/10.1016/j.eswa.2021.114743.
- Farrugia (2020), imputacion KNN mediante perfiles semejantes y validacion
  sobre 335 perfiles, tesis de maestria, Universidad de Malta,
  https://www.um.edu.mt/library/oar/handle/123456789/98708.
- Ismagulov (2021), comparacion de imputacion simple y multiple con variables
  correlacionadas, tesis de maestria, UPV/EHU,
  http://hdl.handle.net/10810/52468.
- Lotfipoor, Patidar y Jenkins (2023), imputacion de demanda electrica mediante
  Transformer y clustering, *Energy and Buildings* 300, 113675.
- Lotfipoor (2024), marco integrado de imputacion y pronostico con Transformer
  y K-means, tesis doctoral, Heriot-Watt University,
  http://hdl.handle.net/10399/5167.

La literatura orienta los candidatos; los datos de Electro Dunas y el
backtesting de huecos artificiales determinan cual se adopta realmente.
