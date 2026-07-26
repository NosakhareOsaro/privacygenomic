"""equitable_rwe: doubly-robust causal outcome estimation with integrated
subgroup fairness auditing for real-world clinical evidence.
"""

from equitable_rwe.api import EquitableSurvivalEstimate, fit_equitable_survival_estimate

__version__ = "0.1.0"

__all__ = ["EquitableSurvivalEstimate", "fit_equitable_survival_estimate", "__version__"]
