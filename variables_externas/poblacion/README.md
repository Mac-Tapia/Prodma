# Población de Ica (INEI, 2 puntos observados + interpolación declarada)

Fuente: INEI, "Perú: Población Total Proyectada al 30 de Junio de cada año, según Departamento, Provincia y Distrito, 2018-2026" (<https://www.gob.pe/institucion/inei/informes-publicaciones/6894980-peru-poblacion-total-proyectada-al-30-de-junio-de-cada-ano-segun-departamento-provincia-y-distrito-2018-2026>).

**Solo se pudieron confirmar 2 cifras reales** (la tabla completa 2018-2026 del PDF oficial no fue extraíble automáticamente):

- 2022-06-30: 1,020,050 habitantes (3.05% de la población proyectada del Perú)
- 2024-06-30: 1,062,346 habitantes (3.12% de la población proyectada del Perú)

El resto de los años (2023, 2025, 2026) se interpola/extrapola linealmente entre esos dos puntos vía `generar_poblacion_ica_interpolada()` (`src/prodma/variables_externas.py`), marcado explícitamente en la columna `dato` (`observado_inei` / `interpolado_lineal` / `extrapolado_lineal`) — nunca se presenta como observación oficial. Reemplazar por la serie completa de INEI en cuanto se pueda extraer del PDF oficial o de una fuente tabular confirmada.
