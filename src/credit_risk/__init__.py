"""
credit_risk
-----------
Credit Risk Monte Carlo Analysis — Python package.

Modules
-------
config              : Centralized parameters and path constants.
logger_config       : Reusable logging helper.
preprocessing       : Data ingestion, EDA, outlier treatment (stages 1–3).
model               : Logistic regression and PD estimation (stage 4).
visualization       : Inference charts with lets-plot (stage 5).
monte_carlo         : Portfolio loss simulation and risk metrics (stage 6).
risk_decomposition  : EL decomposition by PD quintile (stage 7).
"""

from credit_risk.config import MonteCarloConfig, Paths
from credit_risk.logger_config import get_logger

__version__ = "0.1.0"
__all__ = ["MonteCarloConfig", "Paths", "get_logger"]
