"""
logger_config.py
----------------
Reusable logging helper for all pipeline modules.

Usage::

    from credit_risk.logger_config import get_logger
    logger = get_logger(__name__)
    logger.info("Stage started")
    logger.warning("Missing values detected in column: %s", col)
    logger.error("Model fitting failed: %s", exc)
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Configure and return a standard logger with timestamp formatting.

    Calling this function multiple times with the same ``name`` returns the
    same logger instance (Python's logging module is idempotent by design).

    Args:
        name: Logger name — pass ``__name__`` from the calling module.

    Returns:
        Configured ``logging.Logger`` instance writing to stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(name)
