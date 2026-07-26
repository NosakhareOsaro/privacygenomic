"""Causal inference: propensity score matching, IPTW, a doubly-robust
IPTW-adjusted Cox estimator, DoWhy backdoor adjustment with automated
refutation, and E-value sensitivity analysis.
"""

from equitable_rwe.causal.doubly_robust import DoublyRobustResult, fit_doubly_robust_survival
from equitable_rwe.causal.dowhy_backdoor import (
    DoWhyBackdoorResult,
    RefutationResult,
    fit_dowhy_backdoor,
)
from equitable_rwe.causal.evalue import EValueResult, evalue
from equitable_rwe.causal.iptw import compute_iptw_weights, effective_sample_size
from equitable_rwe.causal.propensity import estimate_propensity_scores
from equitable_rwe.causal.psm import MatchResult, match_on_propensity

__all__ = [
    "DoublyRobustResult",
    "fit_doubly_robust_survival",
    "DoWhyBackdoorResult",
    "RefutationResult",
    "fit_dowhy_backdoor",
    "EValueResult",
    "evalue",
    "compute_iptw_weights",
    "effective_sample_size",
    "estimate_propensity_scores",
    "MatchResult",
    "match_on_propensity",
]
