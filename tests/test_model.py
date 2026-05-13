"""
test_model.py
-------------
Unit tests for credit_risk.model.

Tests verify correctness of core modeling functions using synthetic data,
so they run without requiring the real dataset.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from credit_risk.model import (
    fit_logistic_model,
    validate_target,
    split_dataset,
    calculate_ks_statistic,
    calculate_auc_roc,
    save_validation_report,
    _plot_roc_curve,
    _plot_ks_distributions,
    plot_ks_distributions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_credit_data() -> pd.DataFrame:
    """Synthetic credit dataset with realistic class separability.

    Features are drawn from DIFFERENT distributions for good/bad payers
    to ensure KS > 0.20 and AUC > 0.70 consistently.
    This matches the assumption that a real credit dataset has predictive signal.

    Default rate: ~22% (realistic for consumer lending portfolios).
    """
    rng = np.random.default_rng(seed=42)
    n = 500
    n_bad = int(n * 0.22)   # ~22% default rate
    n_good = n - n_bad

    # Good payers: higher income, lower loan amount, lower rate, longer history
    good = pd.DataFrame({
        "person_income":               rng.normal(65_000, 15_000, n_good),
        "loan_amnt":                   rng.normal(8_000, 3_000, n_good),
        "loan_int_rate":               rng.normal(10.5, 2.5, n_good),
        "cb_person_cred_hist_length":  rng.normal(6.0, 2.0, n_good),
        "loan_status": 0,
    })

    # Bad payers: lower income, higher loan amount, higher rate, shorter history
    bad = pd.DataFrame({
        "person_income":               rng.normal(38_000, 12_000, n_bad),
        "loan_amnt":                   rng.normal(14_000, 4_000, n_bad),
        "loan_int_rate":               rng.normal(16.0, 3.5, n_bad),
        "cb_person_cred_hist_length":  rng.normal(2.5, 1.5, n_bad),
        "loan_status": 1,
    })

    df = pd.concat([good, bad], ignore_index=True)

    # Clip to realistic ranges (no negative income, rate between 5% and 35%)
    df["person_income"] = df["person_income"].clip(lower=10_000)
    df["loan_amnt"] = df["loan_amnt"].clip(lower=500)
    df["loan_int_rate"] = df["loan_int_rate"].clip(lower=5.0, upper=35.0)
    df["cb_person_cred_hist_length"] = df["cb_person_cred_hist_length"].clip(lower=0)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Tests for Target Validation
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

# ---------------------------------------------------------------------------
# Tests for Logistic Model Fit
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Task 0.1 Tests
# ---------------------------------------------------------------------------

def test_split_result_has_named_attributes(synthetic_credit_data):
    result = split_dataset(synthetic_credit_data)
    assert hasattr(result, "train"), "SplitResult missing .train attribute"
    assert hasattr(result, "test"), "SplitResult missing .test attribute"
    assert isinstance(result.train, pd.DataFrame)
    assert isinstance(result.test, pd.DataFrame)


def test_split_result_positional_unpacking_still_works(synthetic_credit_data):
    df_train, df_test = split_dataset(synthetic_credit_data)
    assert isinstance(df_train, pd.DataFrame)
    assert isinstance(df_test, pd.DataFrame)
    # Verify order is correct: train must be larger than test
    assert len(df_train) > len(df_test)

# ---------------------------------------------------------------------------
# Task 0.2 Tests
# ---------------------------------------------------------------------------

def test_ks_statistic_returns_tuple_with_pvalue(synthetic_credit_data):
    import statsmodels.api as sm
    features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
    fitted = fit_logistic_model(synthetic_credit_data, features=features)
    X = sm.add_constant(synthetic_credit_data[features])
    y_score = fitted.predict(X).values
    y_true = synthetic_credit_data["loan_status"]
    result = calculate_ks_statistic(y_true, y_score)
    assert isinstance(result, tuple), "calculate_ks_statistic must return a tuple"
    assert len(result) == 2, "tuple must have (ks_stat, ks_pvalue)"
    ks_stat, ks_pvalue = result
    assert 0.0 <= ks_stat <= 1.0
    assert 0.0 <= ks_pvalue <= 1.0

def test_plot_roc_curve_returns_figure():
    """_plot_roc_curve must return a Figure with no filesystem side effects."""
    fpr = np.array([0.0, 0.2, 0.4, 0.6, 1.0])
    tpr = np.array([0.0, 0.5, 0.7, 0.9, 1.0])
    auc = 0.78
    fig = _plot_roc_curve(fpr, tpr, auc)
    assert isinstance(fig, plt.Figure)
    # Verify the figure has exactly one Axes object
    assert len(fig.axes) == 1
    plt.close(fig)  # cleanup


def test_plot_roc_curve_title_contains_auc():
    fpr = np.array([0.0, 0.5, 1.0])
    tpr = np.array([0.0, 0.8, 1.0])
    fig = _plot_roc_curve(fpr, tpr, auc=0.78)
    ax = fig.axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("0.7800" in label for label in legend_labels), (
        f"AUC value not found in legend. Got: {legend_labels}"
    )
    plt.close(fig)

def test_save_validation_report_includes_pvalue_column(tmp_path):
    save_validation_report(
        ks_stat=0.42,
        ks_pvalue=0.003,
        auc_roc=0.78,
        output_tables=tmp_path,
    )
    report = pd.read_csv(tmp_path / "model_validation_report.csv")
    assert "KS p-value" in report["metric"].values, (
        "Validation report must include KS p-value row"
    )
    pvalue_row = report[report["metric"] == "KS p-value"]
    assert float(pvalue_row["value"].iloc[0]) == 0.003

# ---------------------------------------------------------------------------
# Task 1.1 Tests
# ---------------------------------------------------------------------------

def test_plot_ks_distributions_returns_figure():
    """_plot_ks_distributions must return Figure without filesystem side effects."""
    np.random.seed(42)
    scores_good = np.random.beta(2, 8, size=200)   # concentrated near 0
    scores_bad  = np.random.beta(8, 2, size=50)    # concentrated near 1
    fig = _plot_ks_distributions(scores_good, scores_bad, ks_stat=0.42)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    # Two patches groups = two histograms
    assert len(ax.patches) > 0, "Figure has no histogram patches"
    plt.close(fig)

def test_plot_ks_distributions_annotation_contains_ks_value():
    scores_good = np.random.beta(2, 5, size=100)
    scores_bad  = np.random.beta(5, 2, size=30)
    ks_stat = 0.3714
    fig = _plot_ks_distributions(scores_good, scores_bad, ks_stat)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert any("0.3714" in t for t in texts), (
        f"KS value not found in chart annotation. Got texts: {texts}"
    )
    plt.close(fig)

def test_plot_ks_distributions_saves_png(tmp_path, synthetic_credit_data):
    import statsmodels.api as sm
    features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
    fitted = fit_logistic_model(synthetic_credit_data, features=features)
    X = sm.add_constant(synthetic_credit_data[features])
    y_score = fitted.predict(X).values
    y_true = synthetic_credit_data["loan_status"]
    ks_stat, _ = calculate_ks_statistic(y_true, y_score)
    
    plot_ks_distributions(y_true, y_score, ks_stat, output_figures=tmp_path)
    
    assert (tmp_path / "ks_distributions.png").exists(), (
        "ks_distributions.png not found in output_figures"
    )

# ---------------------------------------------------------------------------
# Task 2.1 Tests
# ---------------------------------------------------------------------------

def test_ks_statistic_meets_minimum_bcra_threshold(synthetic_credit_data):
    """KS must exceed 0.20 — the minimum acceptable threshold per BCRA credit
    risk model guidelines for any model used in production scoring.

    This test validates discriminatory power, not just mathematical correctness.
    A KS below 0.20 means the model cannot reliably separate good from bad payers.

    Note: This test depends on synthetic_credit_data having sufficient class
    separability. If this test fails, check the fixture first.
    """
    import statsmodels.api as sm
    features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
    split = split_dataset(synthetic_credit_data)
    fitted = fit_logistic_model(split.train, features=features)
    X_test = sm.add_constant(split.test[features])
    y_score = fitted.predict(X_test).values
    y_true = split.test["loan_status"]
    ks_stat, ks_pvalue = calculate_ks_statistic(y_true, y_score)
    assert ks_stat > 0.20, (
        f"KS = {ks_stat:.4f} is below the BCRA minimum threshold (0.20). "
        f"KS p-value: {ks_pvalue:.6f}. "
        "Check if synthetic_credit_data has sufficient class separability."
    )

def test_auc_roc_meets_minimum_banking_threshold(synthetic_credit_data):
    """AUC must exceed 0.70 — the minimum acceptable for banking applications.

    AUC > 0.5 (tested elsewhere) only proves the model beats random.
    AUC > 0.70 proves the model meets the industry minimum for production use.
    """
    from sklearn.metrics import roc_auc_score
    import statsmodels.api as sm
    features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
    split = split_dataset(synthetic_credit_data)
    fitted = fit_logistic_model(split.train, features=features)
    X_test = sm.add_constant(split.test[features])
    y_score = fitted.predict(X_test).values
    auc = roc_auc_score(split.test["loan_status"], y_score)
    assert auc > 0.70, (
        f"AUC-ROC = {auc:.4f} is below the banking minimum threshold (0.70). "
        "Check if synthetic_credit_data has sufficient predictive signal."
    )

def test_ks_auc_consistency(synthetic_credit_data):
    """KS and AUC must be approximately consistent via the relation KS ≈ 2*(AUC - 0.5).

    This is not an exact equality but a sanity check: if both metrics are
    implemented correctly, they should not diverge by more than 0.15 from
    the theoretical approximation. A large divergence indicates a bug in
    one of the two implementations.
    """
    from sklearn.metrics import roc_auc_score
    import statsmodels.api as sm
    features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
    split = split_dataset(synthetic_credit_data)
    fitted = fit_logistic_model(split.train, features=features)
    X_test = sm.add_constant(split.test[features])
    y_score = fitted.predict(X_test).values
    y_true = split.test["loan_status"]
    ks_stat, _ = calculate_ks_statistic(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    ks_expected_from_auc = 2 * (auc - 0.5)
    divergence = abs(ks_stat - ks_expected_from_auc)
    assert divergence < 0.15, (
        f"KS ({ks_stat:.4f}) diverges too much from 2*(AUC-0.5) = {ks_expected_from_auc:.4f}. "
        f"Divergence: {divergence:.4f}. "
        "This may indicate a bug in calculate_ks_statistic() or roc_auc_score()."
    )

# ---------------------------------------------------------------------------
# Task 2.2 Tests
# ---------------------------------------------------------------------------

def test_ks_statistic_range(synthetic_credit_data):
    """KS must be in [0, 1] for any binary classification output."""
    import statsmodels.api as sm
    features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
    fitted = fit_logistic_model(synthetic_credit_data, features=features)
    X = sm.add_constant(synthetic_credit_data[features])
    y_score = fitted.predict(X).values
    y_true = synthetic_credit_data["loan_status"]
    ks_stat, ks_pvalue = calculate_ks_statistic(y_true, y_score)
    assert 0.0 <= ks_stat <= 1.0, f"KS out of range: {ks_stat}"
    assert 0.0 <= ks_pvalue <= 1.0, f"p-value out of range: {ks_pvalue}"

def test_ks_statistic_is_zero_for_identical_distributions():
    """KS must be ~0 when good and bad scores are drawn from the same distribution."""
    rng = np.random.default_rng(99)
    scores = rng.uniform(0, 1, size=200)
    y_true = pd.Series([0] * 100 + [1] * 100)
    y_score_identical = scores  # same distribution for both classes
    ks_stat, _ = calculate_ks_statistic(y_true, y_score_identical)
    assert ks_stat < 0.25, (
        f"KS should be near 0 for identical distributions, got {ks_stat:.4f}"
    )

def test_ks_statistic_is_high_for_perfectly_separated_scores():
    """KS must be ~1.0 when good and bad scores are perfectly separated."""
    # Perfect separation: all goods have score 0, all bads have score 1
    y_true = pd.Series([0] * 100 + [1] * 100)
    y_score = np.array([0.01] * 100 + [0.99] * 100)
    ks_stat, ks_pvalue = calculate_ks_statistic(y_true, y_score)
    assert ks_stat > 0.90, f"KS should be ~1.0 for perfect separation, got {ks_stat:.4f}"
    assert ks_pvalue < 0.001, f"p-value should be near 0 for perfect separation"


def test_split_preserves_class_balance(synthetic_credit_data):
    """Stratified split must keep default rate similar in train and test."""
    overall_rate = synthetic_credit_data["loan_status"].mean()
    split = split_dataset(synthetic_credit_data, test_size=0.2)
    train_rate = split.train["loan_status"].mean()
    test_rate = split.test["loan_status"].mean()
    assert abs(train_rate - overall_rate) < 0.05, (
        f"Train default rate {train_rate:.3f} diverges from overall {overall_rate:.3f}"
    )
    assert abs(test_rate - overall_rate) < 0.05, (
        f"Test default rate {test_rate:.3f} diverges from overall {overall_rate:.3f}"
    )

def test_split_sizes_respect_test_size_parameter(synthetic_credit_data):
    """Train and test sizes must respect the test_size parameter (default 20%)."""
    n_total = len(synthetic_credit_data)
    split = split_dataset(synthetic_credit_data, test_size=0.2)
    expected_test_size = int(n_total * 0.2)
    # Allow ±1 row difference due to integer rounding in sklearn
    assert abs(len(split.test) - expected_test_size) <= 1, (
        f"Expected test size ~{expected_test_size}, got {len(split.test)}"
    )
    assert len(split.train) + len(split.test) == n_total, (
        "Train + test must equal full dataset (no rows lost)"
    )

def test_split_is_reproducible(synthetic_credit_data):
    """Same random_state must produce identical splits."""
    split_1 = split_dataset(synthetic_credit_data, random_state=42)
    split_2 = split_dataset(synthetic_credit_data, random_state=42)
    pd.testing.assert_frame_equal(split_1.train, split_2.train)
    pd.testing.assert_frame_equal(split_1.test, split_2.test)

def test_split_train_test_have_no_overlap(synthetic_credit_data):
    """Train and test sets must be mutually exclusive (no row appears in both)."""
    split = split_dataset(synthetic_credit_data)
    train_idx = set(split.train.index)
    test_idx = set(split.test.index)
    overlap = train_idx & test_idx
    assert len(overlap) == 0, (
        f"Found {len(overlap)} rows present in both train and test sets"
    )

def test_auc_roc_above_random(synthetic_credit_data):
    """A logistic model on non-trivial data must beat random (AUC > 0.5)."""
    from sklearn.metrics import roc_auc_score
    import statsmodels.api as sm
    features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
    split = split_dataset(synthetic_credit_data)
    fitted = fit_logistic_model(split.train, features=features)
    X_test = sm.add_constant(split.test[features])
    y_score = fitted.predict(X_test).values
    auc = roc_auc_score(split.test["loan_status"], y_score)
    assert auc > 0.5, f"AUC-ROC should exceed random baseline (0.5), got {auc:.4f}"

def test_calculate_auc_roc_saves_png_to_disk(tmp_path, synthetic_credit_data):
    """calculate_auc_roc() must write roc_curve.png to the specified directory."""
    import statsmodels.api as sm
    features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
    split = split_dataset(synthetic_credit_data)
    fitted = fit_logistic_model(split.train, features=features)
    X_test = sm.add_constant(split.test[features])
    y_score = fitted.predict(X_test).values
    y_true = split.test["loan_status"]

    auc = calculate_auc_roc(y_true, y_score, output_figures=tmp_path)

    assert (tmp_path / "roc_curve.png").exists(), "roc_curve.png not found in output_figures"
    assert auc > 0.5

def test_calculate_auc_roc_creates_output_dir_if_missing(tmp_path, synthetic_credit_data):
    """calculate_auc_roc() must create output_figures if it doesn't exist."""
    import statsmodels.api as sm
    features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
    split = split_dataset(synthetic_credit_data)
    fitted = fit_logistic_model(split.train, features=features)
    X_test = sm.add_constant(split.test[features])
    y_score = fitted.predict(X_test).values
    y_true = split.test["loan_status"]

    nested_path = tmp_path / "deep" / "nested" / "figures"
    assert not nested_path.exists()
    calculate_auc_roc(y_true, y_score, output_figures=nested_path)
    assert nested_path.exists()
    assert (nested_path / "roc_curve.png").exists()
