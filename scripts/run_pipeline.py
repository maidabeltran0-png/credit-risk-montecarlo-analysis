"""
run_pipeline.py
---------------
Single entry point for the full credit risk analysis pipeline.

Executes all 7 stages in order:
    Stage 1 — Data ingestion and cleaning
    Stage 2 — Exploratory data analysis
    Stage 3 — Outlier treatment and final preprocessing
    Stage 4 — Logistic regression and PD estimation
    Stage 5 — Inference visualizations
    Stage 6 — Monte Carlo portfolio loss simulation
    Stage 7 — Expected Loss decomposition by PD quintile

Usage (from project root)::

    uv run python scripts/run_pipeline.py

Notes
-----
- ``uv sync`` must be run once before the first execution so that
  the ``credit_risk`` package is installed in the virtual environment.
- All output artefacts are written to ``output/``.
- Simulation results are reproducible: seed is fixed in ``MonteCarloConfig``.
"""

import sys
from pathlib import Path

# Ensure the package is importable when running without `uv sync` install.
# With `uv sync` (recommended), this fallback is never triggered.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from credit_risk.config import MonteCarloConfig, Paths
from credit_risk.logger_config import get_logger
from credit_risk import preprocessing, model, visualization, monte_carlo, risk_decomposition

logger = get_logger(__name__)


def run_stage1_ingestion() -> None:
    """Stage 1: Load raw data, cast types, handle missing values."""
    logger.info("--- Stage 1: Ingestion & Cleaning ---")
    df = preprocessing.load_raw_data(Paths.raw)
    df = preprocessing.convert_dtypes(df)
    preprocessing.report_missing_values(df, Paths.output_tables)
    df = preprocessing.clean_missing_values(df)
    preprocessing.save_dataframe(df, Paths.clean)
    preprocessing.save_stage1_statistics(df, output_dir=Paths.output_tables)
    logger.info("Stage 1 complete — clean dataset: %s", Paths.clean)


def run_stage2_eda() -> None:
    """Stage 2: Descriptive statistics, distributions, correlations."""
    logger.info("--- Stage 2: Exploratory Data Analysis ---")
    df = preprocessing.load_clean_data(Paths.clean)
    preprocessing.calculate_statistics(df, Paths.output_tables)
    preprocessing.generate_histograms_boxplots(df, Paths.output_plots)
    preprocessing.analyze_target(df, Paths.output_plots)
    preprocessing.calculate_correlations(df, Paths.output_tables)
    preprocessing.export_eda_summary(df, Paths.output_tables)
    logger.info("Stage 2 complete")


def run_stage3_preprocessing() -> None:
    """Stage 3: IQR winsorization, median imputation, export processed CSV."""
    logger.info("--- Stage 3: Outlier Treatment & Preprocessing ---")
    df = preprocessing.load_clean_data(Paths.clean)
    stats_pre = df.describe()
    preprocessing.detect_outliers(df)
    df = preprocessing.winsorize(df)
    df = preprocessing.impute_missing(df)
    preprocessing.save_processed(df, Paths.processed)
    preprocessing.export_pre_post_comparison(stats_pre, df.describe(), Paths.output_tables)
    logger.info("Stage 3 complete — processed dataset: %s", Paths.processed)


def run_stage4_model() -> None:
    """Stage 4: Logistic regression, t-test, PD estimation (pd_hat)."""
    logger.info("--- Stage 4: Logistic Regression & PD Estimation ---")
    df = model.load_processed_data(Paths.processed)
    model.validate_target(df)
    model.hypothesis_test(df, Paths.output_tables)

    try:
        features = ["person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]

        # Validation loop (train/test split)
        df_train, df_test = model.split_dataset(df)
        fitted_val = model.fit_logistic_model(df_train, features=features)
        
        import statsmodels.api as sm
        X_test = sm.add_constant(df_test[features])
        y_score_test = fitted_val.predict(X_test)
        y_true_test = df_test["loan_status"]

        ks_stat = model.calculate_ks_statistic(y_true_test, y_score_test.values)
        auc_roc = model.calculate_auc_roc(y_true_test, y_score_test.values, Paths.output_figures)
        model.save_validation_report(ks_stat, auc_roc, Paths.output_tables)
        logger.info("Model validation complete — KS: %.4f | AUC-ROC: %.4f", ks_stat, auc_roc)

        # Fit on full dataset for Monte Carlo scoring
        fitted_model = model.fit_logistic_model(df, features=features)
        df = model.save_model_results(
            df, fitted_model,
            features=features,
            output_tables=Paths.output_tables,
            output_figures=Paths.output_figures,
        )
        # Persist pd_hat back into the processed dataset
        df.to_csv(Paths.processed, index=False)
        logger.info("pd_hat persisted to processed dataset")
    except Exception as exc:
        logger.error("Model fitting failed: %s", exc)
        raise

    logger.info("Stage 4 complete")


def run_stage5_visualizations() -> None:
    """Stage 5: Inference charts (income by status, PD vs. income)."""
    logger.info("--- Stage 5: Visualizations ---")
    df = visualization.load_processed_data(Paths.processed)
    df = visualization.ensure_pd_hat(df)
    p_income = visualization.generate_income_chart(df)
    p_pd = visualization.generate_pd_chart(df)
    p_income.show()
    p_pd.show()
    visualization.save_charts(p_income, p_pd, Paths.output_figures)
    logger.info("Stage 5 complete")


def run_stage6_monte_carlo(cfg: MonteCarloConfig) -> None:
    """Stage 6: Monte Carlo simulation — base and stress scenarios."""
    logger.info("--- Stage 6: Monte Carlo Portfolio Simulation ---")
    np.random.seed(cfg.random_seed)
    Paths.output_monte_carlo.mkdir(parents=True, exist_ok=True)

    df = monte_carlo.load_data(Paths.processed)
    pd_vals = df["pd_hat"].values
    ead_vals = df["loan_amnt"].values

    # Base scenario
    losses_base, draws = monte_carlo.simulate_losses(
        pd_vals, ead_vals, cfg.lgd, cfg.n_simulations, cfg.random_seed
    )
    metrics_base = monte_carlo.calculate_risk_metrics(losses_base, cfg.var_confidence_levels)
    metrics_base.to_csv(Paths.output_monte_carlo / "risk_metrics_base.csv", index=False)

    el_base = losses_base.mean()
    var_95_base = np.percentile(losses_base, 95)
    es_95_base = losses_base[losses_base >= var_95_base].mean()

    monte_carlo.plot_loss_distribution(
        losses_base, cfg.var_confidence_levels,
        "Loss distribution — Monte Carlo Base",
        Paths.output_monte_carlo / "loss_distribution_base.png",
    )

    # Stress scenario
    losses_stress, _ = monte_carlo.simulate_stress_scenario(
        df, cfg.lgd, cfg.n_simulations, cfg.stress_pd_multiplier, cfg.random_seed, uniform_draws=draws
    )
    el_stress = losses_stress.mean()
    var_95_stress = np.percentile(losses_stress, 95)
    es_95_stress = losses_stress[losses_stress >= var_95_stress].mean()

    import pandas as pd
    metrics_stress = pd.DataFrame({
        "metric": ["Expected Loss", "VaR 95%", "Expected Shortfall 95%"],
        "value": [el_stress, var_95_stress, es_95_stress],
    })
    metrics_stress.to_csv(Paths.output_monte_carlo / "risk_metrics_stress.csv", index=False)

    monte_carlo.plot_loss_distribution(
        losses_stress, cfg.var_confidence_levels,
        "Loss distribution — Stress PD",
        Paths.output_monte_carlo / "loss_distribution_stress.png",
    )

    monte_carlo.plot_normalized_overlay(
        losses_base, losses_stress,
        el_base, el_stress,
        var_95_base, var_95_stress,
        Paths.output_monte_carlo / "loss_distribution_overlay.png",
    )

    # Comparison table
    comparison = pd.DataFrame({
        "Metric": ["Expected Loss", "VaR 95%", "Expected Shortfall 95%"],
        "Base": [el_base, var_95_base, es_95_base],
        "Stress": [el_stress, var_95_stress, es_95_stress],
    })
    comparison["Delta_abs"] = comparison["Stress"] - comparison["Base"]
    comparison["Delta_pct"] = (comparison["Delta_abs"] / comparison["Base"] * 100).round(2)
    comparison.to_csv(
        Paths.output_monte_carlo / "base_vs_stress_comparison.csv", index=False,
    )
    logger.info("Base vs. Stress comparison:\n%s", comparison.to_string(index=False))
    logger.info("Stage 6 complete")


def run_stage7_decomposition(cfg: MonteCarloConfig) -> None:
    """Stage 7: EL decomposition by PD quintile."""
    logger.info("--- Stage 7: Risk Decomposition ---")
    df = risk_decomposition.load_data(Paths.processed)
    df = risk_decomposition.calculate_individual_el(df, cfg.lgd)
    risk_decomposition.decompose_by_quintiles(df, Paths.output_tables)
    logger.info("Stage 7 complete")


def main() -> None:
    """Execute the complete credit risk pipeline end-to-end."""
    logger.info("=" * 60)
    logger.info("PIPELINE START: Credit Risk Monte Carlo Analysis")
    logger.info("=" * 60)

    cfg = MonteCarloConfig()
    logger.info(
        "Config — simulations: %d | LGD: %.2f | seed: %d | stress: ×%.1f",
        cfg.n_simulations, cfg.lgd, cfg.random_seed, cfg.stress_pd_multiplier,
    )

    run_stage1_ingestion()
    run_stage2_eda()
    run_stage3_preprocessing()
    run_stage4_model()
    run_stage5_visualizations()
    run_stage6_monte_carlo(cfg)
    run_stage7_decomposition(cfg)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — all outputs written to output/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
