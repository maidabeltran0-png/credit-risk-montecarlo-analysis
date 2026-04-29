"""
monte_carlo.py
--------------
Stage 6: Monte Carlo portfolio loss simulation and risk metrics.

Implements the Basel II/III credit risk metrics framework:
  - Expected Loss (EL)     = mean of the simulated loss distribution
  - Value at Risk (VaR)    = percentile of the loss distribution
  - Expected Shortfall (ES / CVaR) = conditional mean beyond VaR

Two scenarios are computed and compared:
  - Base scenario    : uses PD estimates from the logistic model (pd_hat)
  - Stress scenario  : PD multiplied by ``stress_pd_multiplier`` (capped at 1.0)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_risk.config import MonteCarloConfig
from credit_risk.logger_config import get_logger

logger = get_logger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Load the processed dataset with ``pd_hat`` for simulation.

    Args:
        path: Path to the processed CSV.

    Returns:
        DataFrame with at least ``pd_hat`` and ``loan_amnt`` columns.
    """
    df = pd.read_csv(path)
    logger.info("Dataset loaded for Monte Carlo — loans: %d", len(df))
    return df


def simulate_losses(
    pd_values: np.ndarray,
    ead_values: np.ndarray,
    lgd: float,
    n_simulations: int,
    seed: int = 42,
) -> np.ndarray:
    """Simulate portfolio losses using a fully vectorized approach.

    Generates a full (n_simulations x n_exposures) matrix of random draws
    to avoid Python loops, resulting in sub-second execution for large portfolios.

    Args:
        pd_values: Array of individual default probabilities (length = n_loans).
        ead_values: Array of Exposure at Default — loan amounts (length = n_loans).
        lgd: Loss Given Default (constant proportion, e.g. 0.45).
        n_simulations: Number of Monte Carlo iterations.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_simulations,) with total portfolio losses.
    """
    rng = np.random.default_rng(seed)
    
    # Generate full matrix of uniform draws and compare against PDs
    uniform_draws = rng.random((n_simulations, len(pd_values)))
    defaults_matrix = (uniform_draws < pd_values).astype(np.float64)
    
    # Compute total losses via matrix multiplication
    portfolio_losses = defaults_matrix @ (ead_values * lgd)
    
    logger.info("Base simulation complete — %d iterations (Vectorized)", n_simulations)
    return portfolio_losses


def calculate_risk_metrics(
    losses: np.ndarray,
    confidence_levels: list,
) -> pd.DataFrame:
    """Compute EL, VaR, and Expected Shortfall from simulated losses.

    Args:
        losses: Array of simulated portfolio losses.
        confidence_levels: List of quantiles, e.g. [0.95, 0.99].

    Returns:
        DataFrame with columns ``metric`` and ``value``.
    """
    metrics = [{"metric": "Expected Loss", "value": losses.mean()}]
    for level in confidence_levels:
        var = np.percentile(losses, level * 100)
        es = losses[losses >= var].mean()
        label = f"{int(level * 100)}%"
        metrics.append({"metric": f"VaR {label}", "value": var})
        metrics.append({"metric": f"Expected Shortfall {label}", "value": es})

    df_metrics = pd.DataFrame(metrics)
    logger.info("Risk metrics:\n%s", df_metrics.to_string(index=False))
    return df_metrics


def simulate_stress_scenario(
    df: pd.DataFrame,
    lgd: float,
    n_simulations: int,
    stress_multiplier: float,
    seed: int = 42,
) -> tuple[np.ndarray, pd.Series]:
    """Run Monte Carlo under a stressed PD scenario using vectorization.

    Applies ``stress_multiplier`` to ``pd_hat``, capping at 1.0,
    then generates a full random matrix to compute stressed losses.

    Args:
        df: DataFrame with ``pd_hat`` and ``loan_amnt``.
        lgd: Loss Given Default.
        n_simulations: Number of iterations.
        stress_multiplier: Factor applied to base PD (e.g. 1.5 = +50% PD).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (losses_array, stressed_pd_series).
    """
    pd_stress = np.minimum(1.0, stress_multiplier * df["pd_hat"])
    
    rng = np.random.default_rng(seed)
    uniform_draws = rng.random((n_simulations, len(pd_stress)))
    defaults_matrix = (uniform_draws < pd_stress.values).astype(np.float64)
    
    losses_stress = defaults_matrix @ (df["loan_amnt"].values * lgd)
    
    logger.info("Stress simulation complete (×%.1f PD multiplier, Vectorized)", stress_multiplier)
    return losses_stress, pd_stress


def plot_loss_distribution(
    losses: np.ndarray,
    var_levels: list,
    title: str,
    output_path: Path,
) -> None:
    """Save a histogram of simulated losses with VaR reference lines.

    Args:
        losses: Simulated loss array.
        var_levels: Confidence levels for VaR lines (e.g. [0.95, 0.99]).
        title: Chart title.
        output_path: Full path for the PNG output file.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(losses, bins=50)
    for level in var_levels:
        plt.axvline(
            np.percentile(losses, level * 100),
            linestyle="--",
            linewidth=2,
            label=f"VaR {int(level * 100)}%",
        )
    plt.xlabel("Total portfolio loss")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Loss distribution chart saved → %s", output_path)


def plot_normalized_overlay(
    losses_base: np.ndarray,
    losses_stress: np.ndarray,
    el_base: float,
    el_stress: float,
    var_base: float,
    var_stress: float,
    output_path: Path,
) -> None:
    """Overlay normalized loss distributions (Base vs. Stress).

    Normalizing by EL allows a shape comparison independent of scale.

    Args:
        losses_base: Base scenario loss array.
        losses_stress: Stress scenario loss array.
        el_base: Expected Loss (base) — used as divisor.
        el_stress: Expected Loss (stress) — used as divisor.
        var_base: VaR 95% (base).
        var_stress: VaR 95% (stress).
        output_path: Full path for the PNG output file.
    """
    losses_base_norm = losses_base / el_base
    losses_stress_norm = losses_stress / el_stress

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#480694")
    ax.set_facecolor("#9023e96e")

    ax.hist(losses_base_norm, bins=60, density=True, alpha=0.70,
            label="Base", edgecolor="none")
    ax.hist(losses_stress_norm, bins=60, density=True, alpha=0.45,
            color="yellow", label="Stress PD", edgecolor="none")

    ax.axvline(var_base / el_base, linestyle="--", linewidth=2, label="VaR 95% Base")
    ax.axvline(var_stress / el_stress, linestyle="--", linewidth=2, label="VaR 95% Stress")

    ax.grid(True, linestyle=":", alpha=0.25)
    ax.set_title(
        "Normalised loss distribution — Base vs. Stress PD",
        fontsize=13, fontweight="bold", color="white",
    )
    ax.set_xlabel("Loss / Expected Loss", color="white")
    ax.set_ylabel("Density", color="white")
    ax.tick_params(colors="white")
    ax.legend(
        frameon=True, facecolor="#3c096c", edgecolor="white",
        loc="center left", bbox_to_anchor=(1.02, 0.5),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Overlay chart saved → %s", output_path)
