"""
test_monte_carlo.py
-------------------
Unit tests for credit_risk.monte_carlo.

Tests verify correctness of simulation and risk metric functions
using synthetic data, so they run without requiring the real dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from credit_risk.monte_carlo import (
    calculate_risk_metrics,
    simulate_losses,
    simulate_stress_scenario,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_portfolio() -> pd.DataFrame:
    """Create a minimal synthetic portfolio for testing."""
    rng = np.random.default_rng(seed=42)
    n = 200
    return pd.DataFrame({
        "pd_hat": rng.uniform(0.02, 0.40, n),
        "loan_amnt": rng.uniform(2_000, 30_000, n),
    })


@pytest.fixture
def simulated_losses(synthetic_portfolio) -> np.ndarray:
    """Pre-compute a small loss array (100 sims) for metric tests."""
    np.random.seed(42)
    df = synthetic_portfolio
    return simulate_losses(
        pd_values=df["pd_hat"].values,
        ead_values=df["loan_amnt"].values,
        lgd=0.45,
        n_simulations=100,
    )


# ---------------------------------------------------------------------------
# simulate_losses
# ---------------------------------------------------------------------------

def test_simulate_losses_output_shape(synthetic_portfolio):
    """simulate_losses should return an array of length n_simulations."""
    np.random.seed(0)
    df = synthetic_portfolio
    losses = simulate_losses(df["pd_hat"].values, df["loan_amnt"].values, lgd=0.45, n_simulations=50)
    assert losses.shape == (50,), f"Expected shape (50,), got {losses.shape}"


def test_simulate_losses_non_negative(synthetic_portfolio):
    """All simulated losses must be non-negative."""
    np.random.seed(0)
    df = synthetic_portfolio
    losses = simulate_losses(df["pd_hat"].values, df["loan_amnt"].values, lgd=0.45, n_simulations=200)
    assert (losses >= 0).all(), "Found negative loss values"


def test_simulate_losses_reproducible(synthetic_portfolio):
    """Fixing np.random.seed should yield identical results across runs."""
    df = synthetic_portfolio
    np.random.seed(7)
    losses_a = simulate_losses(df["pd_hat"].values, df["loan_amnt"].values, lgd=0.45, n_simulations=100)
    np.random.seed(7)
    losses_b = simulate_losses(df["pd_hat"].values, df["loan_amnt"].values, lgd=0.45, n_simulations=100)
    np.testing.assert_array_equal(losses_a, losses_b)


# ---------------------------------------------------------------------------
# calculate_risk_metrics
# ---------------------------------------------------------------------------

def test_calculate_risk_metrics_columns(simulated_losses):
    """Risk metrics DataFrame must have 'metric' and 'value' columns."""
    metrics = calculate_risk_metrics(simulated_losses, confidence_levels=[0.95, 0.99])
    assert "metric" in metrics.columns
    assert "value" in metrics.columns


def test_calculate_risk_metrics_expected_rows(simulated_losses):
    """With 2 confidence levels, we expect 1 EL + 2 VaR + 2 ES = 5 rows."""
    metrics = calculate_risk_metrics(simulated_losses, confidence_levels=[0.95, 0.99])
    assert len(metrics) == 5, f"Expected 5 metric rows, got {len(metrics)}"


def test_var_greater_than_expected_loss(simulated_losses):
    """VaR 95% should be >= Expected Loss for a right-skewed loss distribution."""
    metrics = calculate_risk_metrics(simulated_losses, confidence_levels=[0.95])
    el = metrics.loc[metrics["metric"] == "Expected Loss", "value"].iloc[0]
    var95 = metrics.loc[metrics["metric"] == "VaR 95%", "value"].iloc[0]
    assert var95 >= el, f"VaR 95% ({var95:.2f}) should be >= EL ({el:.2f})"


def test_es_greater_than_var(simulated_losses):
    """Expected Shortfall (ES) must be >= VaR at the same confidence level."""
    metrics = calculate_risk_metrics(simulated_losses, confidence_levels=[0.95])
    var95 = metrics.loc[metrics["metric"] == "VaR 95%", "value"].iloc[0]
    es95 = metrics.loc[metrics["metric"] == "Expected Shortfall 95%", "value"].iloc[0]
    assert es95 >= var95, f"ES 95% ({es95:.2f}) should be >= VaR 95% ({var95:.2f})"


# ---------------------------------------------------------------------------
# simulate_stress_scenario
# ---------------------------------------------------------------------------

def test_stress_losses_higher_than_base(synthetic_portfolio):
    """Stressed losses should exceed base losses on average (mean test)."""
    np.random.seed(42)
    df = synthetic_portfolio
    losses_base = simulate_losses(df["pd_hat"].values, df["loan_amnt"].values, 0.45, 500)

    np.random.seed(42)
    losses_stress, _ = simulate_stress_scenario(df, lgd=0.45, n_simulations=500, stress_multiplier=1.5)

    assert losses_stress.mean() > losses_base.mean(), (
        f"Stress EL ({losses_stress.mean():.2f}) should exceed base EL ({losses_base.mean():.2f})"
    )


def test_stress_pd_capped_at_one(synthetic_portfolio):
    """Stressed PDs must not exceed 1.0."""
    np.random.seed(0)
    df = synthetic_portfolio.copy()
    df["pd_hat"] = 0.9  # High base PD; multiplier would push above 1.0
    _, pd_stress = simulate_stress_scenario(df, lgd=0.45, n_simulations=10, stress_multiplier=2.0)
    assert (pd_stress <= 1.0).all(), "Stressed PDs should be capped at 1.0"
