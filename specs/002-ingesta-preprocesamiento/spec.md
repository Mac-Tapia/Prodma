# Spec 002: ingesta y preprocesamiento exhaustivos

**Entrada:** inventario aceptado de Spec 001.
**Resultado:** representación estandarizada sin limpiar ni inventar mediciones.

## Historias

- Como analista, quiero que todas las hojas y columnas queden contabilizadas.
- Como desarrollador, quiero un esquema estable independiente del formato anual.

## Requisitos funcionales

- **S002-FR-001:** abrir cada archivo soportado desde su ruta inventariada y
  enumerar todas sus hojas/tablas antes de seleccionar datos.
- **S002-FR-002:** resolver encabezados mediante alias versionados y política
  determinista de primera coincidencia de izquierda a derecha; toda ambigüedad se
  audita.
- **S002-FR-003:** preservar valor original, hoja, fila, columna, unidad declarada,
  medidor y marca temporal antes de cualquier transformación.
- **S002-FR-004:** normalizar fechas a `America/Lima` y frecuencia canónica de 15
  minutos, registrando marcas rechazadas o redondeadas.
- **S002-FR-005:** detectar duplicados coincidentes y contradictorios sin escoger
  silenciosamente un valor.
- **S002-FR-006:** publicar un Parquet estandarizado por libro/hoja/medidor y un
  catálogo de cobertura. Ingesta no imputa, recorta ni elimina atípicos.
- **S002-FR-007:** reconciliar archivos, hojas, filas, columnas y series entre
  inventario, procesado, cuarentena y fallo.

## Contrato de salida

Cada registro incluye instante, medidor, variable, valor original, unidad, fuente y
banderas de validez estructural. Los nombres físicos de archivos no sustituyen los
identificadores extraídos y validados.

## Aceptación

- Un libro con varias hojas produce evidencia para todas, incluso las rechazadas.
- Columnas duplicadas siguen la política aprobada y dejan diagnóstico.
- Los valores numéricos aceptados son iguales a los originales antes de limpieza.
- Los totales de cobertura cierran exactamente con Spec 001.

## Trazabilidad prevista

Módulos objetivo: `prodma/ingestion/`, adaptando `prodma/ingesta/excel.py`.
Pruebas con `.xls/.xlsx`, esquemas sintéticos, libros reales en solo lectura y
reconciliación de cobertura. Evidencia en `01_ingesta/`.
