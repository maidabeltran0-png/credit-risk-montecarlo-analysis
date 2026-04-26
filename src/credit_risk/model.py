"""
model.py
--------
Stage 4: Statistical inference and logistic regression for PD estimation.

Fits a logistic regression model using ``statsmodels`` and produces:
- Welch t-test comparing income between default / non-default groups.
- Model coefficients table (CSV).
- Individual PD estimates (``pd_hat``) appended to the processed dataset.
- Scatter plot of predicted PD vs. income.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_ind

from credit_risk.logger_config import get_logger

logger = get_logger(__name__)

_DEFAULT_FEATURES = [
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_cred_hist_length",
]


def load_processed_data(path: Path) -> pd.DataFrame:
    """Load the processed dataset (output of stage 3).

    Args:
        path: Path to ``credit_risk_processed.csv``.

    Returns:
        DataFrame ready for model fitting.
    """
    df = pd.read_csv(path)
    logger.info("Processed dataset loaded — shape: %s", df.shape)
    return df


def validate_target(df: pd.DataFrame, target: str = "loan_status") -> None:
    """Verify the target column has at least two distinct classes.

    Args:
        df: DataFrame containing the target column.
        target: Name of the binary target column.

    Raises:
        ValueError: If the target column has only one class.
    """
    dist = df[target].value_counts(normalize=True)
    logger.info("Target distribution:\n%s", dist.to_string())
    if df[target].nunique() < 2:
        raise ValueError(f"'{target}' has a single class — cannot fit a model.")


def hypothesis_test(
    df: pd.DataFrame,
    output_dir: Path,
    feature: str = "person_income",
    target: str = "loan_status",
) -> None:
    """Perform a Welch t-test comparing a feature across default / non-default groups.

    H0: mean(feature | default) == mean(feature | no default)
    H1: means differ (two-sided)

    Args:
        df: DataFrame with target and feature columns.
        output_dir: Directory to write the results CSV.
        feature: Continuous variable to compare between groups.
        target: Binary target (1 = default, 0 = no default).
    """
    group_default = df[df[target] == 1][feature]
    group_no_default = df[df[target] == 0][feature]

    logger.info(
        "Groups — default: %d obs | no-default: %d obs",
        len(group_default), len(group_no_default),
    )

    if len(group_default) < 30 or len(group_no_default) < 30:
        logger.warning("Sample too small for t-test (< 30 per group)")
        return

    t_stat, p_value = ttest_ind(group_default, group_no_default, equal_var=False)
    logger.info("Welch t-test — t=%.4f | p=%.6f", t_stat, p_value)

    if p_value < 0.05:
        logger.info("H0 rejected: significant income difference between groups (p < 0.05)")
    else:
        logger.warning("H0 not rejected: no significant income difference (p >= 0.05)")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({
        "group": ["Default", "No Default"],
        f"mean_{feature}": [group_default.mean(), group_no_default.mean()],
    })
    results.to_csv(output_dir / "hypothesis_test_income.csv", index=False)


def fit_logistic_model(
    df: pd.DataFrame,
    features: list = _DEFAULT_FEATURES,
    target: str = "loan_status",
) -> sm.Logit:
    """Fit a logistic regression model using statsmodels.

    Args:
        df: DataFrame with features and target.
        features: List of predictor column names.
        target: Binary outcome variable name.

    Returns:
        Fitted ``LogitResults`` object from statsmodels.
    """
    X = sm.add_constant(df[features])
    y = df[target]
    model = sm.Logit(y, X).fit(maxiter=100, disp=False)
    logger.info("Logistic model fitted — pseudo R²: %.4f", model.prsquared)
    return model


def save_model_results(
    df: pd.DataFrame,
    model: sm.Logit,
    features: list,
    output_tables: Path,
    output_figures: Path,
) -> pd.DataFrame:
    """Persist model coefficients, PD predictions, and diagnostic chart.

    Args:
        df: Original DataFrame.
        model: Fitted logistic model.
        features: Features used during fitting.
        output_tables: Directory for CSV exports.
        output_figures: Directory for PNG charts.

    Returns:
        DataFrame with ``pd_hat`` column appended.
    """
    output_tables.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)

    # Coefficients
    coef_table = model.summary2().tables[1]
    coef_table.to_csv(output_tables / "logistic_regression_coefficients.csv")
    logger.info("Model coefficients exported")

    # PD predictions
    X = sm.add_constant(df[features])
    df = df.copy()
    df["pd_hat"] = model.predict(X)
    logger.info("pd_hat — mean: %.4f | min: %.4f | max: %.4f",
                df["pd_hat"].mean(), df["pd_hat"].min(), df["pd_hat"].max())

    # Scatter chart
    plt.figure()
    sns.scatterplot(x=df["person_income"], y=df["pd_hat"], alpha=0.3)
    plt.xlabel("Income")
    plt.ylabel("Predicted PD (pd_hat)")
    plt.title("Predicted probability of default vs. income")
    plt.tight_layout()
    plt.savefig(output_figures / "pd_hat_vs_income.png", dpi=300)
    plt.close()
    logger.info("PD chart saved → %s", output_figures)

    return df
