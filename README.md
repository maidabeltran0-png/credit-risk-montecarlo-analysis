# Credit Risk & Monte Carlo Portfolio Analysis

Análisis de riesgo crediticio end-to-end: desde la probabilidad individual de default hasta la distribución de pérdidas de cartera bajo escenarios de stress.

## ¿Qué hace este proyecto?

Pipeline completo de credit risk modeling sobre el [Credit Risk Dataset de Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset):

1. **Preprocesamiento y EDA** — limpieza de tipos, valores faltantes, outliers (IQR winsorización)
2. **Estimación de PD individual** — regresión logística, test t de Welch, coeficientes exportados
3. **Simulación Monte Carlo** — 10.000 escenarios de pérdida de cartera (Bernoulli por préstamo)
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
git clone https://github.com/maidabeltran0-png/trabajo-final-credit-risk-default.git
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

## Stack técnico

Python 3.12 · pandas · numpy · statsmodels · scipy · matplotlib · seaborn · lets-plot

## Autora

Maida Beltrán
