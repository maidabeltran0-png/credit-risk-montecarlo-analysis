"""
test_monte_carlo.py
-------------------
Unit tests for credit_risk.monte_carlo.

Tests verify correctness of simulation and risk metric functions
using synthetic data, so they run without requiring the real dataset.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from credit_risk.config import MonteCarloConfig
from credit_risk.monte_carlo import (
    calculate_risk_metrics,
    simulate_losses,
    simulate_stress_scenario,
)


# ---------------------------------------------------------------------------
# Legacy Functions (for testing equivalence and performance)
# ---------------------------------------------------------------------------

def _legacy_simulate_losses(
    pd_values: np.ndarray,
    ead_values: np.ndarray,
    lgd: float,
    n_simulations: int,
) -> np.ndarray:
    """Old loop-based implementation used as baseline."""
    portfolio_losses = np.zeros(n_simulations)
    for s in range(n_simulations):
        defaults = np.random.binomial(1, pd_values)
        portfolio_losses[s] = (defaults * ead_values * lgd).sum()
    return portfolio_losses


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
        seed=MonteCarloConfig().random_seed,
    )


# ---------------------------------------------------------------------------
# simulate_losses
# ---------------------------------------------------------------------------

def test_simulate_losses_output_shape(synthetic_portfolio):
    """simulate_losses should return an array of length n_simulations."""
    np.random.seed(0)
    df = synthetic_portfolio
    losses = simulate_losses(df["pd_hat"].values, df["loan_amnt"].values, lgd=0.45, n_simulations=50, seed=MonteCarloConfig().random_seed)
    assert losses.shape == (50,), f"Expected shape (50,), got {losses.shape}"


def test_simulate_losses_non_negative(synthetic_portfolio):
    """All simulated losses must be non-negative."""
    np.random.seed(0)
    df = synthetic_portfolio
    losses = simulate_losses(df["pd_hat"].values, df["loan_amnt"].values, lgd=0.45, n_simulations=200, seed=MonteCarloConfig().random_seed)
    assert (losses >= 0).all(), "Found negative loss values"


def test_simulate_losses_reproducible(synthetic_portfolio):
    """Fixing np.random.seed should yield identical results across runs."""
    df = synthetic_portfolio
    np.random.seed(7)
    losses_a = simulate_losses(df["pd_hat"].values, df["loan_amnt"].values, lgd=0.45, n_simulations=100, seed=MonteCarloConfig().random_seed)
    np.random.seed(7)
    losses_b = simulate_losses(df["pd_hat"].values, df["loan_amnt"].values, lgd=0.45, n_simulations=100, seed=MonteCarloConfig().random_seed)
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
    losses_base = simulate_losses(df["pd_hat"].values, df["loan_amnt"].values, 0.45, 500, seed=MonteCarloConfig().random_seed)

    np.random.seed(42)
    losses_stress, _ = simulate_stress_scenario(df, lgd=0.45, n_simulations=500, stress_multiplier=1.5, seed=MonteCarloConfig().random_seed)

    assert losses_stress.mean() > losses_base.mean(), (
        f"Stress EL ({losses_stress.mean():.2f}) should exceed base EL ({losses_base.mean():.2f})"
    )


def test_stress_pd_capped_at_one(synthetic_portfolio):
    """Stressed PDs must not exceed 1.0."""
    np.random.seed(0)
    df = synthetic_portfolio.copy()
    df["pd_hat"] = 0.9  # High base PD; multiplier would push above 1.0
    _, pd_stress = simulate_stress_scenario(df, lgd=0.45, n_simulations=10, stress_multiplier=2.0, seed=MonteCarloConfig().random_seed)
    assert (pd_stress <= 1.0).all(), "Stressed PDs should be capped at 1.0"


# ---------------------------------------------------------------------------
# Vectorization Performance and Equivalence
# ---------------------------------------------------------------------------

def test_simulate_losses_equivalence(synthetic_portfolio):
    """Check that the vectorized and legacy functions yield statistically equivalent results."""
    df = synthetic_portfolio
    n_sims = 1000
    lgd = 0.45
    
    # Run legacy
    np.random.seed(42)
    legacy_losses = _legacy_simulate_losses(df["pd_hat"].values, df["loan_amnt"].values, lgd, n_sims)
    
    # Run vectorized
    vectorized_losses = simulate_losses(df["pd_hat"].values, df["loan_amnt"].values, lgd, n_sims, seed=MonteCarloConfig().random_seed)
    
    # We compare the mean (Expected Loss) and VaR 95%
    # Note: They use different RNG streams, so they won't be exactly element-wise equal,
    # but the distribution properties must be very close.
    legacy_mean = legacy_losses.mean()
    vec_mean = vectorized_losses.mean()
    
    legacy_var = np.percentile(legacy_losses, 95)
    vec_var = np.percentile(vectorized_losses, 95)
    
    assert np.isclose(legacy_mean, vec_mean, rtol=0.05), f"Mean diverges: {legacy_mean} vs {vec_mean}"
    assert np.isclose(legacy_var, vec_var, rtol=0.05), f"VaR diverges: {legacy_var} vs {vec_var}"


def test_simulate_losses_performance(synthetic_portfolio):
    """Assert that the vectorized implementation is significantly faster than the loop."""
    df = synthetic_portfolio
    # Expand the portfolio to make the loop noticeably slow
    df_large = pd.concat([df] * 50, ignore_index=True) # 10,000 exposures
    n_sims = 1000
    lgd = 0.45
    
    # Time legacy
    start_legacy = time.perf_counter()
    np.random.seed(42)
    _legacy_simulate_losses(df_large["pd_hat"].values, df_large["loan_amnt"].values, lgd, n_sims)
    time_legacy = time.perf_counter() - start_legacy
    
    # Time vectorized
    start_vec = time.perf_counter()
    simulate_losses(df_large["pd_hat"].values, df_large["loan_amnt"].values, lgd, n_sims, seed=MonteCarloConfig().random_seed)
    time_vec = time.perf_counter() - start_vec
    
    print(f"\nLegacy time: {time_legacy:.4f}s")
    print(f"Vectorized time: {time_vec:.4f}s")
    print(f"Speedup: {time_legacy / time_vec:.2f}x")
    
    assert time_vec < time_legacy, "Vectorized version should be faster"
    assert time_legacy / time_vec > 5.0, "Vectorized version should be at least 5x faster"
