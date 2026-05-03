# Credit Risk & Monte Carlo Portfolio Analysis
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e)
![uv](https://img.shields.io/badge/uv-package%20manager-7C3AED)

Análisis de riesgo crediticio end-to-end: desde la probabilidad individual de default hasta la distribución de pérdidas de cartera bajo escenarios de stress.

## ¿Qué hace este proyecto?

Pipeline completo de credit risk modeling sobre el [Credit Risk Dataset de Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset):

1. **Preprocesamiento y EDA** — limpieza de tipos, valores faltantes, outliers (IQR winsorización)
2. **Estimación de PD individual** — regresión logística, test t de Welch, coeficientes exportados
3. **Simulación Monte Carlo** — 10.000 escenarios de pérdida de cartera (Bernoulli por préstamo). 
   > *Motor de simulación 100% vectorizado con NumPy: genera matrices de (n_simulaciones × n_exposiciones) en memoria, eliminando bucles Python. Permite escalar a portafolios de 10.000+ exposiciones manteniendo tiempos de ejecución sub-segundo.*
4. **Métricas de riesgo** — Expected Loss, VaR 95%/99%, Expected Shortfall
5. **Stress testing** — impacto de deterioro de PD (×1.5) sobre toda la distribución de pérdidas
6. **Descomposición de EL** — contribución por quintil de PD (EL = PD × EAD × LGD)

## Estructura del proyecto

```
credit-risk-montecarlo-analysis/
├── src/credit_risk/         ← Paquete Python instalable
│   ├── config.py            ← Parámetros centralizados (MonteCarloConfig, Paths)
│   ├── preprocessing.py     ← Stages 1-3: ingesta, EDA, outliers
│   ├── model.py             ← Stage 4: regresión logística, estimación PD
│   ├── visualization.py     ← Stage 5: gráficos de inferencia (lets_plot)
│   ├── monte_carlo.py       ← Stage 6: simulación Monte Carlo
│   └── risk_decomposition.py← Stage 7: descomposición EL por quintil de PD
├── scripts/
│   └── run_pipeline.py      ← Entry point — ejecuta el pipeline completo
├── notebooks/
│   └── exploracion_completa.ipynb ← EDA interactivo
├── tests/
│   ├── test_model.py        ← Tests para estimación de PD
│   └── test_monte_carlo.py  ← Tests para simulación Monte Carlo
├── data/                    ← No versionado (ver data/README.md)
└── output/                  ← Tablas y gráficos generados
```

## Instalación y configuración

### Requisitos

- Python 3.12+
- [uv](https://astral.sh/uv/) (recomendado)

### Setup en Windows

```powershell
# 1. Instalar uv (si no lo tenés)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clonar el repositorio
git clone https://github.com/maidabeltran0-png/credit-risk-montecarlo-analysis.git
cd credit-risk-montecarlo-analysis

# 3. Instalar dependencias
uv sync

# 4. Descargar el dataset de Kaggle y guardarlo en:
#    data/raw/credit_risk_dataset.csv
#    (ver data/README.md para el link)
```

## Cómo ejecutar

### Pipeline completo (recomendado)

```powershell
uv run python scripts/run_pipeline.py
```

Esto ejecuta los 7 stages en orden y genera todos los archivos en `output/`.

### EDA interactivo

```powershell
uv run jupyter lab notebooks/exploracion_completa.ipynb
```

### Tests

```powershell
uv run pytest tests/ -v
```

## Contexto financiero

| Métrica | Qué mide |
|---|---|
| **Expected Loss (EL)** | Pérdida promedio esperada de la cartera |
| **VaR 95%** | Pérdida máxima con 95% de confianza |
| **Expected Shortfall (ES)** | Pérdida promedio en los peores escenarios (más allá del VaR) |

El modelo estima la **Probability of Default (PD)** individual con regresión logística y la combina con la **Exposure at Default (EAD)** y la **Loss Given Default (LGD = 45%)** para calcular la pérdida esperada individual:

> **EL_i = PD_i × EAD_i × LGD**

La simulación Monte Carlo repite este proceso 10.000 veces para obtener la **distribución completa de pérdidas de cartera**.

### Common Random Numbers (CRN) en Stress Testing

Para comparar escenarios (Base vs. Estrés), el motor de simulación implementa **Common Random Numbers (CRN)**, una Técnica de Reducción de Varianza (VRT). El principio es:

> Si un crédito "defaultea" con una extracción aleatoria U=0.62 en el escenario base, también "defaulteará" en el escenario estresado (porque su PD estresada es mayor). Estamos comparando *el mismo portafolio en condiciones distintas*, no dos realizaciones aleatorias distintas.

La varianza del estimador de la diferencia `Pérdida_Estrés - Pérdida_Base` se reduce porque los dos procesos están correlacionados positivamente. Matemáticamente:

```text
Var(X_estrés - X_base) = Var(X_estrés) + Var(X_base) - 2·Cov(X_estrés, X_base)
```

Al usar CRN, la Covarianza es positiva (`Cov > 0`), lo que reduce la varianza total del comparador. Esta técnica es estándar en modelos de stress testing bajo **Basilea III** y marcos de **DFAST/CCAR** para aislar el impacto real del estrés del ruido inherente a la simulación, especialmente en métricas de cola (VaR y Expected Shortfall).

## Stack técnico

Python 3.12 · pandas · numpy · statsmodels · scipy · matplotlib · seaborn · lets-plot

## Roadmap

- [ ] Validación del modelo con KS statistic y AUC-ROC (train/test split)
- [ ] Stress testing con Common Random Numbers (CRN) para comparación base vs. adverso vs. severamente adverso
- [ ] Dashboard interactivo con Streamlit + Plotly
- [ ] Auditoría de datos faltantes y pipeline de imputación

## Autora

Maida Beltrán
