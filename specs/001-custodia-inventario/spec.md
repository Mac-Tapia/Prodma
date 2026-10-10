# Spec 001: custodia, inventario y universo autorizado

**Entrada:** roadmap `000-reestructuracion-integral`
**Resultado:** catálogo reproducible de todas las entradas antes de interpretarlas.

## Historias

- Como operador, quiero saber qué se encontró y qué se procesó para detectar
  omisiones.
- Como auditor, quiero verificar que ningún original fue modificado.
- Como responsable de proyección, quiero que el catálogo de medidores limite el
  universo posible.

## Requisitos funcionales

- **S001-FR-001:** descubrir recursivamente todos los archivos actuales y futuros
  bajo `datahistorica/`, `nuevosclientes/`, `variables_externas/` y las entradas
  raíz declaradas, sin una lista fija de años.
- **S001-FR-002:** registrar ruta relativa, tamaño, mtime, SHA-256, extensión,
  firma/tipo detectado y estado de cada archivo.
- **S001-FR-003:** clasificar cada archivo como `SUPPORTED`, `QUARANTINED`,
  `DUPLICATE` o `FAILED`; una ruta declarada que no existe se registra como
  `MISSING_EXPECTED`. Ninguna entrada encontrada ni expectativa configurada queda
  fuera del recuento.
- **S001-FR-004:** tratar originales como solo lectura y comprobar su hash al final
  de cada etapa que los consulte.
- **S001-FR-005:** leer `medidoralimentador.xlsx` buscando la hoja
  `alimentador_medidor` y el encabezado normalizado exacto `Medidor`; espacios y
  mayúsculas se normalizan, pero guiones y demás caracteres identificadores no.
- **S001-FR-006:** deduplicar identificadores exactos para autorización y registrar
  todas las filas de origen; duplicados conflictivos se bloquean.
- **S001-FR-007:** publicar `input_inventory.jsonl`, `input_summary.json` y
  `authorized_meters.parquet` con hashes.

## Fallo cerrado

Falta del catálogo, fuente obligatoria ausente, encabezado ambiguo, cambio de bytes
durante la ejecución o inventario no reconciliado impiden iniciar ingesta. Un
archivo no compatible queda visible en cuarentena y bloquea la aceptación total
hasta decisión documentada.

## Aceptación

- La suma por estados equivale al total descubierto más las expectativas ausentes
  declaradas.
- Agregar un archivo nuevo cambia el inventario sin editar configuración de años.
- Alterar un original durante la corrida produce fallo verificable.
- Los 112 registros actuales del catálogo conservan procedencia y producen 110
  identificadores exactos únicos; el resultado no presupone que esas cifras sean
  permanentes.

## Trazabilidad prevista

Módulos objetivo: `prodma/custody/`, `prodma/catalog/`. Pruebas: descubrimiento,
hash, cuarentena, duplicados y catálogo real en solo lectura. Evidencia: inventario,
resumen, catálogo autorizado y manifiesto de etapa.
