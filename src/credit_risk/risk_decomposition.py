"""
risk_decomposition.py
---------------------
Stage 7: Expected Loss decomposition by PD quintile.

Breaks down the portfolio's total Expected Loss (EL) into segments
based on the individual default probability (``pd_hat``), following the
standard banking formula:

    EL_i = PD_i × EAD_i × LGD

Segmenting by PD quintile reveals which borrower risk tier concentrates
the most credit risk — essential for limit-setting and provisioning decisions.
"""

from pathlib import Path

import pandas as pd

from credit_risk.config import MonteCarloConfig
from credit_risk.logger_config import get_logger

logger = get_logger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Load the processed dataset with ``pd_hat`` for EL decomposition.

    Args:
        path: Path to the processed CSV.

    Returns:
        DataFrame with ``pd_hat`` and ``loan_amnt`` columns.
    """
    df = pd.read_csv(path)
    logger.info("Dataset loaded for risk decomposition — rows: %d", len(df))
    return df


def calculate_individual_el(df: pd.DataFrame, lgd: float) -> pd.DataFrame:
    """Compute the expected loss for every individual loan.

    Formula: EL_i = PD_i × EAD_i × LGD

    Args:
        df: DataFrame with ``pd_hat`` (PD) and ``loan_amnt`` (EAD).
        lgd: Loss Given Default — constant proportion.

    Returns:
        DataFrame with ``el_individual`` column appended.
    """
    df = df.copy()
    df["el_individual"] = df["pd_hat"] * df["loan_amnt"] * lgd
    total_el = df["el_individual"].sum()
    logger.info("Portfolio Expected Loss: {:,.2f}".format(total_el))
    return df


def decompose_by_quintiles(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Aggregate EL by PD quintile and export the decomposition table.

    Quintile labels:
        Q1 (Low PD) → Q5 (High PD)

    Columns in output:
        pd_quintile, n_loans, mean_pd, total_el, mean_el_per_loan

    Args:
        df: DataFrame with ``pd_hat`` and ``el_individual``.
        output_dir: Directory for the CSV export.

    Returns:
        Summary DataFrame with EL metrics per quintile.
    """
    df = df.copy()
    df["pd_quintile"] = pd.qcut(
        df["pd_hat"],
        q=5,
        labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"],
    )

    summary = (
        df.groupby("pd_quintile", observed=True)
        .agg(
            n_loans=("pd_hat", "count"),
            mean_pd=("pd_hat", "mean"),
            total_el=("el_individual", "sum"),
            mean_el_per_loan=("el_individual", "mean"),
        )
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "el_decomposition_by_quintile.csv", index=False)
    logger.info("EL decomposition exported → %s", output_dir)
    logger.info("Decomposition by quintile:\n%s", summary.to_string(index=False))
    return summary
