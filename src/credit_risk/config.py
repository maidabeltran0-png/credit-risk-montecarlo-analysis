"""
config.py
---------
Centralized configuration for the credit risk analysis pipeline.

All hardcoded values (Monte Carlo parameters, file paths) live here.
Import ``MonteCarloConfig`` and ``Paths`` from this module instead of
defining constants in individual scripts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Project root resolution
# Depth: src/credit_risk/config.py  →  src/credit_risk/  →  src/  →  root
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class Paths:
    """Centralized path constants for all data and output artifacts.

    All paths are absolute and resolved at import time from the project root,
    so they work regardless of the current working directory.
    """

    project_root: Path = _PROJECT_ROOT

    # Data
    raw: Path = _PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv"
    clean: Path = _PROJECT_ROOT / "data" / "clean" / "credit_risk_clean.csv"
    processed: Path = _PROJECT_ROOT / "data" / "processed" / "credit_risk_processed.csv"

    # Outputs
    output_tables: Path = _PROJECT_ROOT / "output" / "tables"
    output_plots: Path = _PROJECT_ROOT / "output" / "plots"
    output_figures: Path = _PROJECT_ROOT / "output" / "figures"
    output_monte_carlo: Path = _PROJECT_ROOT / "output" / "monte_carlo"


@dataclass
class MonteCarloConfig:
    """Simulation parameters for the Monte Carlo credit risk model.

    Attributes:
        n_simulations       : Number of Monte Carlo iterations.
        stress_pd_multiplier: Scaling factor applied to base PD in stress scenario.
        lgd                 : Loss Given Default — proportion of EAD lost on default.
        var_confidence_levels: Quantiles used for VaR and Expected Shortfall.
        random_seed         : Fixed seed for reproducible simulation results.
    """

    n_simulations: int = 10_000
    stress_pd_multiplier: float = 1.5
    lgd: float = 0.45
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    random_seed: int = 42
