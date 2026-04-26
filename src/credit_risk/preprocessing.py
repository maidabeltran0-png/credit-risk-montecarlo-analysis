"""
preprocessing.py
----------------
Data pipeline stages 1–3: ingestion, EDA, and outlier treatment.

Stage 1 — Ingestion & Cleaning
    Load raw CSV, cast column types, handle missing values.

Stage 2 — Exploratory Data Analysis
    Descriptive statistics, histograms, target distribution, correlations.

Stage 3 — Outlier Treatment & Final Preprocessing
    IQR-based winsorization, median imputation, export processed dataset.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk.logger_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default column groups (can be overridden per call)
# ---------------------------------------------------------------------------
_CATEGORICAL_COLS = ["person_home_ownership", "loan_intent", "loan_grade", "loan_status"]
_NUMERIC_COLS = ["person_age", "person_income", "loan_amnt", "loan_int_rate", "person_emp_length"]
_KEY_COLS_FOR_CLEANING = ["loan_int_rate", "person_age", "person_income"]
_OUTLIER_EXCLUDE = ["loan_status"]


# ===========================================================================
# STAGE 1 — Ingestion & Cleaning
# ===========================================================================

def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw credit risk dataset from a CSV file.

    Args:
        path: Absolute path to the raw CSV file.

    Returns:
        DataFrame with all columns as loaded from disk.
    """
    df = pd.read_csv(path)
    logger.info("Raw dataset loaded — rows: %d, cols: %d", df.shape[0], df.shape[1])
    return df


def convert_dtypes(
    df: pd.DataFrame,
    categorical_cols: list = _CATEGORICAL_COLS,
    numeric_cols: list = _NUMERIC_COLS,
) -> pd.DataFrame:
    """Cast columns to their intended types (categorical / numeric).

    Args:
        df: Input DataFrame.
        categorical_cols: Columns to cast to ``category`` dtype.
        numeric_cols: Columns to coerce to numeric (invalid values → NaN).

    Returns:
        DataFrame with corrected dtypes.
    """
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    logger.info("Column dtypes converted — categoricals: %s", categorical_cols)
    return df


def report_missing_values(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute and export a missing-values report.

    Args:
        df: DataFrame to inspect.
        output_dir: Directory to write the CSV report.

    Returns:
        DataFrame with columns ``count`` and ``pct`` per original column.
    """
    missing = df.isna().sum().to_frame(name="count")
    missing["pct"] = (df.isna().mean() * 100).round(2)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing.to_csv(output_dir / "missing_values_initial.csv")
    logger.info("Missing values report saved — %s", output_dir)
    return missing


def clean_missing_values(
    df: pd.DataFrame,
    key_columns: list = _KEY_COLS_FOR_CLEANING,
) -> pd.DataFrame:
    """Drop rows with missing values in key modeling columns.

    Assumption: missingness is MCAR (< 5% of rows), so listwise deletion
    does not introduce systematic bias.

    Args:
        df: Input DataFrame.
        key_columns: Columns that must not contain NaN.

    Returns:
        DataFrame without rows that have nulls in ``key_columns``.
    """
    df = df.replace("?", np.nan)
    rows_before = len(df)
    df = df.dropna(subset=key_columns)
    logger.info("Rows dropped (missing in key cols): %d", rows_before - len(df))
    return df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame as CSV, creating parent directories as needed.

    Args:
        df: DataFrame to save.
        path: Absolute output path (including filename).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("DataFrame saved → %s", path)


def save_stage1_statistics(
    df: pd.DataFrame,
    numeric_cols: list = _NUMERIC_COLS,
    output_dir: Path = None,
) -> None:
    """Export descriptive statistics for numeric columns post-cleaning.

    Args:
        df: Cleaned DataFrame.
        numeric_cols: Columns to describe.
        output_dir: Destination directory for the CSV.
    """
    if output_dir is None:
        from credit_risk.config import Paths
        output_dir = Paths.output_tables
    output_dir.mkdir(parents=True, exist_ok=True)
    df[numeric_cols].describe().to_csv(output_dir / "descriptive_stats_post_cleaning.csv")
    logger.info("Post-cleaning statistics saved → %s", output_dir)


# ===========================================================================
# STAGE 2 — Exploratory Data Analysis
# ===========================================================================

def load_clean_data(path: Path) -> pd.DataFrame:
    """Load the cleaned dataset from disk.

    Args:
        path: Absolute path to the cleaned CSV.

    Returns:
        Loaded DataFrame.
    """
    df = pd.read_csv(path)
    logger.info("Clean dataset loaded — shape: %s", df.shape)
    return df


def calculate_statistics(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute full descriptive statistics and null counts, then export.

    Args:
        df: Input DataFrame.
        output_dir: Directory for output CSVs.

    Returns:
        Descriptive statistics DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    desc = df.describe(include="all")
    desc.to_csv(output_dir / "descriptive_statistics.csv")
    df.isna().sum().to_csv(output_dir / "null_counts.csv")
    logger.info("Descriptive statistics exported → %s", output_dir)
    return desc


def generate_histograms_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save histograms and boxplots for all numeric columns.

    Args:
        df: Input DataFrame.
        output_dir: Directory to write PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:
        plt.figure()
        plt.hist(df[col].dropna(), bins=30)
        plt.title(f"Histogram — {col}")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_{col}.png", dpi=150)
        plt.close()

        plt.figure()
        plt.boxplot(df[col].dropna())
        plt.title(f"Boxplot — {col}")
        plt.tight_layout()
        plt.savefig(output_dir / f"box_{col}.png", dpi=150)
        plt.close()

    logger.info("Histograms and boxplots generated for %d columns", len(num_cols))


def analyze_target(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze the distribution of ``loan_status`` and save a bar chart.

    Args:
        df: DataFrame containing ``loan_status``.
        output_dir: Directory to write the chart and CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dist = df["loan_status"].value_counts(normalize=True)
    dist.to_csv(output_dir / "loan_status_distribution.csv")

    plt.figure()
    df["loan_status"].value_counts().plot(kind="bar")
    plt.title("Target distribution — loan_status")
    plt.tight_layout()
    plt.savefig(output_dir / "loan_status_distribution.png", dpi=150)
    plt.close()
    logger.info("Target distribution exported — default rate: %.2f%%", dist.get(1, 0) * 100)


def calculate_correlations(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute the Pearson correlation matrix and save heatmap.

    Args:
        df: Input DataFrame.
        output_dir: Directory for CSV and PNG output.

    Returns:
        Correlation matrix as DataFrame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr = df[num_cols].corr()
    corr.to_csv(output_dir / "correlation_matrix.csv")

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close()
    logger.info("Correlation matrix exported (%d numeric cols)", len(num_cols))
    return corr


def export_eda_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Export a concise EDA summary table (column, null count, dtype).

    Args:
        df: Input DataFrame.
        output_dir: Destination directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({
        "column": df.columns,
        "null_count": df.isna().sum().values,
        "dtype": df.dtypes.astype(str).values,
    })
    summary.to_csv(output_dir / "eda_summary.csv", index=False)
    logger.info("EDA summary exported → %s", output_dir)


# ===========================================================================
# STAGE 3 — Outlier Treatment & Final Preprocessing
# ===========================================================================

def detect_outliers(df: pd.DataFrame, exclude: list = _OUTLIER_EXCLUDE) -> pd.Series:
    """Count IQR-based outliers per numeric column.

    Args:
        df: Input DataFrame.
        exclude: Numeric columns to skip (e.g., binary target variables).

    Returns:
        Series mapping column name → outlier count.
    """
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(
        exclude, errors="ignore"
    )
    counts: dict = {}
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        counts[col] = int(mask.sum())

    result = pd.Series(counts)
    logger.info("Outliers detected:\n%s", result.to_string())
    return result


def winsorize(df: pd.DataFrame, exclude: list = _OUTLIER_EXCLUDE) -> pd.DataFrame:
    """Cap outliers to the IQR fence (1.5× rule) for all numeric columns.

    Args:
        df: Input DataFrame.
        exclude: Columns to skip during winsorization.

    Returns:
        DataFrame with outlier values capped at the IQR boundaries.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(
        exclude, errors="ignore"
    )
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = np.where(df[col] < lower, lower, np.where(df[col] > upper, upper, df[col]))
    logger.info("Winsorization applied (excluded: %s)", exclude)
    return df


def impute_missing(df: pd.DataFrame, column: str = "person_emp_length") -> pd.DataFrame:
    """Impute missing values in a column using its median.

    Args:
        df: Input DataFrame.
        column: Column to impute (default: ``person_emp_length``).

    Returns:
        DataFrame with NaN values replaced by the column median.
    """
    df = df.copy()
    median = df[column].median()
    n_missing = int(df[column].isna().sum())
    df[column] = df[column].fillna(median)
    logger.info("Imputed %d missing in '%s' with median=%.2f", n_missing, column, median)
    return df


def save_processed(df: pd.DataFrame, path: Path) -> None:
    """Save the fully processed DataFrame ready for modeling.

    Args:
        df: Processed DataFrame.
        path: Output path for the processed CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Processed dataset saved → %s | shape: %s", path, df.shape)


def export_pre_post_comparison(
    stats_pre: pd.DataFrame,
    stats_post: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Export pre/post outlier treatment statistics and their delta.

    Args:
        stats_pre: ``describe()`` output before treatment.
        stats_post: ``describe()`` output after treatment.
        output_dir: Directory for the three CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_pre.to_csv(output_dir / "stats_pre_treatment.csv")
    stats_post.to_csv(output_dir / "stats_post_treatment.csv")
    (stats_post - stats_pre).to_csv(output_dir / "stats_delta_pre_post.csv")
    logger.info("Pre/post treatment comparison exported → %s", output_dir)
