"""
test_model.py
-------------
Unit tests for credit_risk.model.

Tests verify correctness of core modeling functions using synthetic data,
so they run without requiring the real dataset.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from credit_risk.model import fit_logistic_model, validate_target


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_credit_data() -> pd.DataFrame:
    """Create a minimal synthetic credit dataset for testing."""
    rng = np.random.default_rng(seed=0)
    n = 500
    income = rng.normal(50_000, 15_000, n).clip(10_000, 200_000)
    loan_amnt = rng.normal(10_000, 5_000, n).clip(1_000, 50_000)
    loan_int_rate = rng.normal(12, 4, n).clip(4, 30)
    cred_hist = rng.integers(1, 20, n).astype(float)
    # Higher income → lower PD
    logit = -2 + (-0.00002 * income) + (0.0001 * loan_amnt) + (0.05 * loan_int_rate)
    prob = 1 / (1 + np.exp(-logit))
    loan_status = rng.binomial(1, prob, n)
    return pd.DataFrame({
        "person_income": income,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "cb_person_cred_hist_length": cred_hist,
        "loan_status": loan_status,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_validate_target_passes_with_two_classes(synthetic_credit_data):
    """validate_target should not raise when both classes are present."""
    validate_target(synthetic_credit_data)  # No exception expected


def test_validate_target_raises_with_single_class(synthetic_credit_data):
    """validate_target should raise ValueError when only one class exists."""
    df_single = synthetic_credit_data.copy()
    df_single["loan_status"] = 0
    with pytest.raises(ValueError):
        validate_target(df_single)


def test_fit_logistic_model_returns_predictions(synthetic_credit_data):
    """Fitted model should produce PD predictions in [0, 1]."""
    import statsmodels.api as sm
    features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
    fitted = fit_logistic_model(synthetic_credit_data, features=features)
    X = sm.add_constant(synthetic_credit_data[features])
    preds = fitted.predict(X)
    assert preds.min() >= 0.0, "PD predictions must be >= 0"
    assert preds.max() <= 1.0, "PD predictions must be <= 1"


def test_pd_hat_negatively_correlated_with_income(synthetic_credit_data):
    """Higher income should correlate with lower predicted PD (negative relationship)."""
    import statsmodels.api as sm
    features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
    fitted = fit_logistic_model(synthetic_credit_data, features=features)
    X = sm.add_constant(synthetic_credit_data[features])
    df = synthetic_credit_data.copy()
    df["pd_hat"] = fitted.predict(X)
    corr = df[["person_income", "pd_hat"]].corr().loc["person_income", "pd_hat"]
    assert corr < 0, f"Expected negative correlation income/PD, got {corr:.4f}"
