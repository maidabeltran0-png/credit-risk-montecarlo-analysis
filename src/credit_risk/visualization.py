"""
visualization.py
----------------
Stage 5: Inference charts using lets-plot.

Generates publication-quality charts that communicate the core findings
of the logistic regression model:
  1. Average income by credit status (default vs. no default).
  2. Predicted PD vs. income with LOESS smoother.
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from lets_plot import (
    LetsPlot,
    aes,
    element_rect,
    element_text,
    geom_bar,
    geom_point,
    geom_smooth,
    ggplot,
    ggsave,
    labs,
    theme,
    theme_minimal,
)

from credit_risk.logger_config import get_logger

logger = get_logger(__name__)

LetsPlot.setup_html()

_DEFAULT_FEATURES = [
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_cred_hist_length",
]

_THEME_BASE = (
    theme_minimal()
    + theme(
        panel_background=element_rect(fill="#fafafa"),
        plot_background=element_rect(fill="#fafafa"),
        axis_title_x=element_text(size=12, face="bold"),
        axis_title_y=element_text(size=12, face="bold"),
        plot_title=element_text(size=15, face="bold"),
        plot_caption=element_text(size=10, color="gray20", hjust=0),
    )
)


def load_processed_data(path: Path) -> pd.DataFrame:
    """Load the processed dataset for visualization.

    Args:
        path: Path to the processed CSV containing ``pd_hat``.

    Returns:
        Loaded DataFrame.
    """
    df = pd.read_csv(path)
    logger.info("Dataset loaded for visualization — shape: %s", df.shape)
    return df


def ensure_pd_hat(
    df: pd.DataFrame,
    features: list = _DEFAULT_FEATURES,
) -> pd.DataFrame:
    """Compute ``pd_hat`` if not already present in the DataFrame.

    This is a fallback for cases where stage 4 was not run. Prefer
    loading the dataset after stage 4 has appended ``pd_hat``.

    Args:
        df: DataFrame that may or may not contain ``pd_hat``.
        features: Predictor columns for the logistic model.

    Returns:
        DataFrame guaranteed to have ``pd_hat`` column.
    """
    if "pd_hat" in df.columns:
        return df
    logger.warning("pd_hat not found — fitting logistic model to compute it")
    X = sm.add_constant(df[features])
    y = df["loan_status"]
    model = sm.Logit(y, X).fit(disp=0)
    df = df.copy()
    df["pd_hat"] = model.predict(X)
    logger.info("pd_hat computed — mean: %.4f", df["pd_hat"].mean())
    return df


def generate_income_chart(df: pd.DataFrame) -> object:
    """Bar chart: average income by credit status (default vs. no default).

    Args:
        df: DataFrame with ``loan_status`` and ``person_income``.

    Returns:
        lets-plot ``ggplot`` object ready for display or export.
    """
    summary = (
        df.groupby("loan_status")["person_income"]
        .mean()
        .reset_index()
    )
    summary["Status"] = summary["loan_status"].map({0: "No Default", 1: "Default"})

    caption = (
        "Borrowers who default show significantly lower average incomes,\n"
        "consistent with the Welch t-test result (p < 0.05)."
    )

    p = (
        ggplot(summary, aes(x="Status", y="person_income", fill="Status"))
        + geom_bar(stat="identity", width=0.6)
        + labs(
            title="Average income by credit status",
            subtitle="Comparison between performing and defaulting borrowers",
            caption=caption,
            x="Credit status",
            y="Average income",
        )
        + _THEME_BASE
        + theme(legend_position="none")
    )
    logger.info("Income bar chart generated")
    return p


def generate_pd_chart(df: pd.DataFrame, sample_size: int = 4000) -> object:
    """Scatter plot of predicted PD vs. income with LOESS smoother.

    Args:
        df: DataFrame with ``person_income`` and ``pd_hat``.
        sample_size: Number of points to plot (random sample for clarity).

    Returns:
        lets-plot ``ggplot`` object ready for display or export.
    """
    caption = (
        "The logistic model shows a negative relationship between income and\n"
        "predicted default probability: higher income → lower predicted credit risk."
    )

    p = (
        ggplot(df.sample(sample_size, random_state=42), aes(x="person_income", y="pd_hat"))
        + geom_point(alpha=0.3)
        + geom_smooth(method="loess", se=True)
        + labs(
            title="Predicted default probability vs. income",
            subtitle="Logistic regression output — sampled for readability",
            caption=caption,
            x="Income",
            y="Predicted PD (pd_hat)",
        )
        + _THEME_BASE
    )
    logger.info("PD scatter chart generated")
    return p


def save_charts(p_income: object, p_pd: object, output_dir: Path) -> None:
    """Export both inference charts to PNG files.

    Args:
        p_income: Income bar chart.
        p_pd: PD scatter chart.
        output_dir: Destination directory for PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ggsave(p_income, str(output_dir / "income_by_credit_status.png"), dpi=300)
    ggsave(p_pd, str(output_dir / "pd_hat_vs_income.png"), dpi=300)
    logger.info("Inference charts saved → %s", output_dir)
