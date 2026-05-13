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
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ks_2samp, ttest_ind
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

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


class SplitResult(NamedTuple):
    """Return type for split_dataset(). Access via .train and .test.

    Using NamedTuple instead of a plain tuple prevents bugs caused by
    accidentally inverting the unpacking order (test, train = split_dataset(...)).
    """
    train: pd.DataFrame
    test: pd.DataFrame


def split_dataset(
    df: pd.DataFrame,
    target: str = "loan_status",
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitResult:
    """Stratified train/test split preserving class balance.

    Stratification is critical for imbalanced credit datasets where
    default rates are typically 10-30% of the portfolio.

    Args:
        df: Full processed DataFrame.
        target: Binary target column name.
        test_size: Proportion of data for test set (default: 20%).
        random_state: Seed for reproducibility.

    Returns:
        SplitResult with attributes .train (80%) and .test (20%).
        Also supports positional unpacking: df_train, df_test = split_dataset(df).
    """
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target],
    )
    logger.info(
        "Split — train: %d | test: %d | test default rate: %.2f%%",
        len(df_train), len(df_test),
        df_test[target].mean() * 100,
    )
    return SplitResult(train=df_train, test=df_test)


def calculate_ks_statistic(
    y_true: pd.Series,
    y_score: np.ndarray,
) -> tuple[float, float]:
    """Compute the KS statistic for a binary classification model.

    The KS statistic measures the maximum separation between the cumulative
    distribution of scores for good payers (y=0) and bad payers (y=1).
    It is the standard model validation metric in Argentine banking and
    fintech (aligned with BCRA credit risk guidelines).

    Interpretation:
        KS < 0.20  → Poor discriminatory power
        0.20–0.40  → Acceptable
        0.40–0.60  → Good
        KS > 0.60  → Excellent (rare in practice)

    Args:
        y_true: Binary ground truth labels (1 = default).
        y_score: Predicted default probabilities (pd_hat).

    Returns:
        Tuple of (ks_stat, ks_pvalue):
            ks_stat: KS statistic in [0, 1]. Higher = better discrimination.
            ks_pvalue: p-value for H0: "both score distributions are equal".
                A p-value < 0.05 indicates statistically significant separation.
    """
    scores_good = y_score[y_true == 0]
    scores_bad = y_score[y_true == 1]
    ks_stat, ks_pvalue = ks_2samp(scores_good, scores_bad)
    logger.info(
        "KS Statistic: %.4f | p-value: %.6f%s",
        ks_stat,
        ks_pvalue,
        " [significant]" if ks_pvalue < 0.05 else " [NOT significant — check sample size]",
    )
    return ks_stat, ks_pvalue


def _plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> plt.Figure:
    """Generate ROC curve figure without writing to disk.

    Separated from calculate_auc_roc() to allow testing chart generation
    without filesystem side effects.

    Args:
        fpr: False positive rates from roc_curve().
        tpr: True positive rates from roc_curve().
        auc: Area under the ROC curve.

    Returns:
        Matplotlib Figure with the ROC curve plotted.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"AUC-ROC = {auc:.4f}", linewidth=2, color="#2563eb")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random classifier (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Logistic Regression PD Model")
    ax.legend()
    fig.tight_layout()
    return fig


def calculate_auc_roc(
    y_true: pd.Series,
    y_score: np.ndarray,
    output_figures: Path,
) -> float:
    """Compute AUC-ROC and save the ROC curve chart.

    Chart generation is delegated to _plot_roc_curve() (pure function).
    This function handles only the I/O (saving to disk).

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted probabilities.
        output_figures: Directory for PNG output.

    Returns:
        AUC-ROC score (float in [0, 1]).
    """
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig = _plot_roc_curve(fpr, tpr, auc)
    output_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figures / "roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("AUC-ROC: %.4f | chart saved → %s", auc, output_figures / "roc_curve.png")
    return auc


def _plot_ks_distributions(
    scores_good: np.ndarray,
    scores_bad: np.ndarray,
    ks_stat: float,
) -> plt.Figure:
    """Generate overlapping score distribution histograms without writing to disk.

    Visualizes the maximum separation between good and bad payer score
    distributions — the core concept behind the KS Statistic.

    Args:
        scores_good: Predicted PD scores for non-defaulting borrowers (y=0).
        scores_bad: Predicted PD scores for defaulting borrowers (y=1).
        ks_stat: Pre-computed KS statistic to annotate on the chart.

    Returns:
        Matplotlib Figure with overlapping histograms and KS annotation.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 50)

    ax.hist(
        scores_good, bins=bins, alpha=0.6, color="#2563eb", density=True,
        label=f"Buenos pagadores — y=0 (n={len(scores_good):,})",
    )
    ax.hist(
        scores_bad, bins=bins, alpha=0.6, color="#dc2626", density=True,
        label=f"Malos pagadores — y=1 (n={len(scores_bad):,})",
    )

    ax.set_xlabel("Score de PD predicho (mayor = mayor riesgo de default)", fontsize=11)
    ax.set_ylabel("Densidad", fontsize=11)
    ax.set_title(
        f"Distribución de scores por clase — KS Statistic = {ks_stat:.4f}",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.text(
        0.72, 0.88, f"KS = {ks_stat:.4f}",
        transform=ax.transAxes, fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7", edgecolor="#d97706", alpha=0.9),
    )

    fig.tight_layout()
    return fig


def plot_ks_distributions(
    y_true: pd.Series,
    y_score: np.ndarray,
    ks_stat: float,
    output_figures: Path,
) -> None:
    """Generate and save the KS score distribution chart.

    Orchestrates _plot_ks_distributions() (pure) and handles the filesystem I/O.
    Output: output_figures/ks_distributions.png

    Args:
        y_true: Binary ground truth labels (1 = default).
        y_score: Predicted default probabilities.
        ks_stat: Pre-computed KS statistic (from calculate_ks_statistic()).
        output_figures: Directory for PNG output.
    """
    scores_good = y_score[y_true == 0]
    scores_bad = y_score[y_true == 1]

    fig = _plot_ks_distributions(scores_good, scores_bad, ks_stat)
    output_figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figures / "ks_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        "KS distribution chart saved → %s",
        output_figures / "ks_distributions.png",
    )


def save_validation_report(
    ks_stat: float,
    ks_pvalue: float,
    auc_roc: float,
    output_tables: Path,
) -> None:
    """Export model validation metrics to CSV.

    Args:
        ks_stat: Kolmogorov-Smirnov statistic.
        ks_pvalue: p-value for the KS statistic.
        auc_roc: Area Under ROC Curve.
        output_tables: Directory for CSV output.
    """
    def ks_interpretation(ks: float) -> str:
        if ks < 0.20: return "Poor"
        if ks < 0.40: return "Acceptable"
        if ks < 0.60: return "Good"
        return "Excellent"

    report = pd.DataFrame([
        {"metric": "KS Statistic", "value": round(ks_stat, 4),
         "interpretation": ks_interpretation(ks_stat),
         "reference": "BCRA / Argentine banking standard"},
        {"metric": "KS p-value", "value": round(ks_pvalue, 6),
         "interpretation": "< 0.05 → separation is statistically significant",
         "reference": "scipy.stats.ks_2samp two-sided test"},
        {"metric": "AUC-ROC", "value": round(auc_roc, 4),
         "interpretation": "Random = 0.5 | Perfect = 1.0",
         "reference": "International ML standard"},
    ])
    output_tables.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_tables / "model_validation_report.csv", index=False)
    logger.info("Validation report saved:\n%s", report.to_string(index=False))
