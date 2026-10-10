# Sustento técnico y científico

## Método de desarrollo y trazabilidad

| Fuente | Aporte al diseño |
|---|---|
| GitHub, *Spec-Driven Development* y *Spec of Specs* | Separar intención, plan, tareas, implementación y convergencia; dividir cambios que exceden un ciclo manejable. https://github.github.com/spec-kit/ |
| Göknil, A. (2011), tesis doctoral, *Traceability of Requirements and Software Architecture for Change Management* | Vincular requisitos con arquitectura, código y pruebas para analizar cambios. https://doi.org/10.3990/1.9789036531757 |
| Vanzetti, J. J. (2006), tesis de maestría, *Un modelo del proceso de desarrollo de software guiado por la traceability* | Usar trazabilidad como eje del proceso desde requisitos hasta arquitectura. https://sedici.unlp.edu.ar/handle/10915/4076 |
| Koboyatshwene, T. y Ayalew, Y. (2025), *Requirements Traceability: A Systematic Literature Review* | Mantener trazabilidad durante todo el ciclo y no depender de herramientas parciales. https://doi.org/10.1145/3672608.3707952 |
| Manik, L. P. (2015), tesis EngD, *Modular software architecture for a large complex codebase* | Reducir entrelazamiento mediante límites e interfaces explícitas. https://research.tue.nl/en/publications/modular-software-architecture-for-a-large-complex-codebase/ |

## Pronóstico, limpieza y evaluación

| Fuente | Aporte al diseño |
|---|---|
| Hong, T. (2010), tesis doctoral, *Short Term Electric Load Forecasting* | Evaluación y modelamiento específicos de carga eléctrica. https://repository.lib.ncsu.edu/items/c8f6d98e-3422-4a0b-9d5e-91cfeaa0de87 |
| Hasanov, M. (2022), tesis de maestría, *Load forecasting using machine learning for smart meter data processing* | Preprocesamiento y modelamiento de mediciones inteligentes. https://www.researchgate.net/publication/363108666 |
| Tashman, L. J. (2000), *Out-of-sample tests of forecasting accuracy* | Validación fuera de muestra y orígenes de pronóstico. https://doi.org/10.1016/S0169-2070(00)00065-0 |
| Cerqueira, V., Torgo, L. y Mozetič, I. (2020), *Evaluating time series forecasting models* | Selección de métodos de estimación temporal del rendimiento. https://doi.org/10.1007/s10994-020-05867-9 |
| Hewamalage, H. et al. (2023), *Forecast evaluation for data scientists* | Evitar métricas inadecuadas, fugas y comparaciones inconsistentes. https://doi.org/10.1007/s10618-022-00894-5 |
| Meng, Y. et al. (2023), *Short-term electricity load forecasting based on a novel data preprocessing system* | Tratar el preprocesamiento como componente explícito del sistema predictivo. https://doi.org/10.1016/j.jobe.2023.107432 |

## Contexto para agentes

| Fuente | Aporte al diseño |
|---|---|
| OpenAI, *Using skills to accelerate OSS maintenance* | Mantener política estable en `AGENTS.md` y flujos especializados en skills locales. https://developers.openai.com/blog/skills-agents-sdk |
| Pan, Z. et al. (2024), *Enhancing Repository-Level Code Generation with Integrated Contextual Information* | Recuperar dependencias y contexto relevante en lugar de enviar todo el repositorio. https://arxiv.org/abs/2406.03283 |
| GraphCoder (IEEE, 2024) | La recuperación estructural mejora la generación a nivel de repositorio con menor costo de contexto. https://ieeexplore.ieee.org/document/10765009/ |

Estas fuentes sustentan decisiones de arquitectura y método; no demuestran por sí
solas que Prodma funcione. La aceptación depende de pruebas y ejecuciones reales.
