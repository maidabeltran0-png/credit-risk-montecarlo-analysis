# Data

Este directorio contiene los datos del proyecto. **No se versionan en git** (ver `.gitignore`).

## Cómo obtener el dataset

Descargar el dataset de Kaggle:

**[Credit Risk Dataset — Laotse (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)**

Guardar el archivo descargado como:

```
data/raw/credit_risk_dataset.csv
```

## Estructura

```
data/
├── raw/                 ← Dataset original descargado de Kaggle
│   └── credit_risk_dataset.csv
├── clean/               ← Generado por Stage 1 (limpieza de tipos y faltantes)
│   └── credit_risk_clean.csv
└── processed/           ← Generado por Stage 3 (winsorización + imputación)
    └── credit_risk_processed.csv
```

Los archivos en `clean/` y `processed/` se regeneran automáticamente al ejecutar el pipeline.
