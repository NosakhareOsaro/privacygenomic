"""Subgroup fairness auditing: an AEQUITAS-style classifier audit
(demographic parity, equalized odds, calibration) and a causal-estimator
reliability audit (per-subgroup effective sample size and covariate
balance for a doubly-robust survival estimate).
"""

from equitable_rwe.fairness.audit import FairnessReport, audit_fairness
from equitable_rwe.fairness.subgroup_causal import (
    CausalFairnessReport,
    audit_causal_subgroup_fairness,
)

__all__ = [
    "FairnessReport",
    "audit_fairness",
    "CausalFairnessReport",
    "audit_causal_subgroup_fairness",
]
