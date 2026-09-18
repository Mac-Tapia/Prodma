# Prodma — Proyección de demanda eléctrica

Pipeline local (VS Code / Jupyter) para pronóstico de demanda eléctrica por medidor (P/Q, cada 15 min): lectura de históricos anuales, limpieza causal, selección de modelos por validación temporal, pronóstico operativo recursivo, incorporación de nuevos clientes, escenarios de planificación y una tabla de demandas para DIgSILENT PowerFactory.

## Modelos comparados

Por cada medidor y variable (P/Q) se comparan **9 candidatos a 15 minutos** (el ranking que decide el pronóstico operativo): Ridge, RandomForest, XGBoost, LightGBM, CatBoost y MLP (selección de features por algoritmo genético + búsqueda bayesiana de hiperparámetros con Optuna, corrección de heterocedasticidad), más 3 benchmarks de persistencia estacional. Por separado, una comparación **exigida por el plan** corre SeasonalNaive, ARIMA, SARIMA, Holt-Winters, Prophet y LSTM, cada uno sobre el dataset a la resolución que necesita para rendir a su potencial (diaria para los clásicos y Prophet, horaria para LSTM) — no compiten por el pronóstico operativo, solo documentan la comparación contra esos benchmarks. Todos los modelos, en ambas comparaciones, pasan por la misma batería de métricas y diagnósticos estadísticos (RMSE/MAE/R², Durbin-Watson, Ljung-Box, Breusch-Pagan/White).

## Figuras generadas

Por cada medidor, la curva histórica (`curva_<medidor>.png`) y la de proyección (`proyeccion_<medidor>.png`) usan el mismo formato: un solo panel con P (azul), Q (verde) y S/KVA (morado, medido cuando el medidor lo trae o derivado como `hypot(P,Q)` para el pronóstico) superpuestos; en la de proyección, el tramo pronosticado continúa en línea punteada justo donde termina el histórico, sin salto visual. Además, cada uno de los 6 modelos ML (no solo el ganador) genera su propia figura por medidor/variable con su curva de convergencia/aprendizaje (error de validación por ronda de boosting en XGBoost/LightGBM/CatBoost, loss por iteración en MLP, o coeficientes/importancia de features en Ridge/RandomForest) junto a su panel de prueba real-vs-predicho.

## Estructura del proyecto

```
Prodma/
├── datahistorica/        # Históricos anuales de entrada (demanda2023.xlsx … demanda2026.xlsx) — solo lectura
├── nuevosclientes/       # Archivo de altas de nuevos clientes — solo lectura
├── variables_externas/   # Exógenas para las proyecciones (calendario, temperatura; ver su README)
├── notebook/
│   └── proyecciondemanda.ipynb   # Notebook principal (orquestación local, sin Colab)
├── src/prodma/           # Paquete Python con la lógica del pipeline (en construcción modular)
├── tools/                # Scripts de mantenimiento (generar variables externas, actualizar el notebook)
├── tests/                # Pruebas (unit/integration)
├── docs/                 # Documentación y planes de implementación
├── resultados/           # Salidas por ejecución (ejecucion_<fecha>/) — generadas, no versionadas
├── requirements.txt      # Dependencias CPU
├── requirements-gpu.txt  # Dependencias adicionales para entrenamiento con GPU (CUDA)
└── .venv/                # Entorno virtual local (no versionado)
```

Los archivos de entrada en `datahistorica/` y `nuevosclientes/` nunca se modifican ni se mueven: la ubicación de estas carpetas en la raíz del proyecto es parte del contrato que valida el propio notebook y el pipeline (ver `docs/superpowers/plans/`).

## Variables externas (exógenas)

`variables_externas/` cubre lo que `Plan_implementacion_demanda_Electro_Dunas.md` (sección 2) permite automatizar sin inventar datos: calendario de feriados de Perú (offline, biblioteca `holidays`), temperatura horaria de reanálisis (Open-Meteo), PBI mensual nacional (BCRPData, código confirmado) y población de Ica (INEI, 2 puntos observados + interpolación/extrapolación declarada). Número de clientes y producción minera regional quedan documentados como **pendientes** en su propio `README.md` (requieren datos internos de Electro Dunas o verificación manual de una publicación en PDF) — nunca se rellenan con ceros o supuestos. Regenerar con:

```powershell
.\.venv\Scripts\python.exe tools\generar_variables_externas.py
```

Ver [variables_externas/README.md](variables_externas/README.md) para fuentes, cobertura y limitaciones declaradas.

## Preparar el entorno

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si vas a entrenar con GPU (NVIDIA/CUDA), instala además:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

En VS Code, selecciona `.venv\Scripts\python.exe` como intérprete/kernel del notebook.

## Ejecutar

Abre `notebook/proyecciondemanda.ipynb` en VS Code con el kernel `.venv` y ejecuta las celdas en orden. Cada ejecución crea una carpeta independiente en `resultados/ejecucion_<fecha>_<hora>/` sin sobrescribir ejecuciones previas.

Para correrlo **desde terminal**, de cero hasta tener los resultados listos para entregar, usa la CLI del proyecto (envuelve el notebook y lo ejecuta por fases, registrando su propio kernel automáticamente):

```powershell
.\.venv\Scripts\python.exe -m pip install -e . --no-deps   # una sola vez
.\.venv\Scripts\python.exe -m prodma.pipeline --listar-fases
.\.venv\Scripts\python.exe -m prodma.pipeline --modo rapido --backend auto
```

Guía completa, con todas las fases y cómo interpretar los resultados, en [docs/MANUAL_EJECUCION.md](docs/MANUAL_EJECUCION.md).

## Estado de la modularización

La lógica del notebook se está extrayendo progresivamente a módulos reutilizables en `src/prodma/`, con pruebas en `tests/`. El plan detallado (tareas, arquitectura objetivo y criterios de aceptación) está en [docs/superpowers/plans/2026-09-16-demanda-electro-dunas-gpu-local.md](docs/superpowers/plans/2026-09-16-demanda-electro-dunas-gpu-local.md).
