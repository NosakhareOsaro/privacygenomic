"""The package's headline API: one call returning both a causal estimate
and a subgroup fairness report.

Most doubly-robust causal estimators stop at a point estimate and a
confidence interval. This function additionally answers, in the same
call, a question that estimate alone cannot: is the evidence behind it
equally strong for every subgroup that matters, or does the topline
number rest on much weaker support for some of them? See
:mod:`equitable_rwe.causal.doubly_robust` and
:mod:`equitable_rwe.fairness.subgroup_causal` for the two halves this
combines.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from equitable_rwe.causal.doubly_robust import DoublyRobustResult, fit_doubly_robust_survival
from equitable_rwe.fairness.subgroup_causal import (
    CausalFairnessReport,
    audit_causal_subgroup_fairness,
)


@dataclass
class EquitableSurvivalEstimate:
    """A doubly-robust causal survival estimate plus its subgroup fairness report.

    Attributes
    ----------
    causal_estimate : DoublyRobustResult
        The doubly-robust IPTW-weighted, covariate-adjusted Cox estimate
        on the full cohort.
    fairness_report : CausalFairnessReport
        Per-subgroup reliability audit of that estimate.
    """

    causal_estimate: DoublyRobustResult
    fairness_report: CausalFairnessReport

    @property
    def hazard_ratio(self) -> float:
        """Shortcut for ``causal_estimate.hazard_ratio``."""
        return self.causal_estimate.hazard_ratio

    @property
    def is_equitable(self) -> bool:
        """``True`` if the estimate is adequately supported for every subgroup.

        This says nothing about whether the *effect itself* differs by
        subgroup (see ``fairness_report.per_subgroup`` for that) — only
        whether there is enough evidence, and adequate covariate balance,
        behind the estimate in every subgroup to trust it at all.
        """
        return self.fairness_report.all_subgroups_adequately_supported


def fit_equitable_survival_estimate(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    treatment_col: str,
    covariates: list[str],
    subgroup_col: str,
) -> EquitableSurvivalEstimate:
    """Fit a doubly-robust survival estimate with an integrated subgroup fairness audit.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per subject.
    duration_col, event_col : str
        Follow-up time and event indicator columns.
    treatment_col : str
        Binary (0/1) treatment indicator whose effect is being estimated.
    covariates : list[str]
        Covariates for the propensity model and outcome-model adjustment
        (the source of this estimator's double robustness — see
        :func:`equitable_rwe.causal.doubly_robust.fit_doubly_robust_survival`).
    subgroup_col : str
        Column defining the subgroups to audit for estimate reliability
        (e.g. a demographic or geographic category).

    Returns
    -------
    EquitableSurvivalEstimate

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 1200
    >>> x1 = rng.normal(size=n)
    >>> treat = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x1)))
    >>> subgroup = rng.choice(["majority", "minority"], size=n, p=[0.9, 0.1])
    >>> duration = rng.exponential(30 * np.exp(0.3 * treat + 0.3 * x1))
    >>> df = pd.DataFrame(
    ...     {
    ...         "T": duration,
    ...         "E": rng.binomial(1, 0.9, n),
    ...         "treat": treat,
    ...         "x1": x1,
    ...         "group": subgroup,
    ...     }
    ... )
    >>> result = fit_equitable_survival_estimate(df, "T", "E", "treat", ["x1"], "group")
    >>> result.hazard_ratio > 0
    True
    >>> isinstance(result.is_equitable, bool)
    True
    """
    causal_estimate = fit_doubly_robust_survival(
        df, duration_col, event_col, treatment_col, covariates
    )
    fairness_report = audit_causal_subgroup_fairness(
        df, duration_col, event_col, treatment_col, covariates, subgroup_col
    )
    return EquitableSurvivalEstimate(
        causal_estimate=causal_estimate, fairness_report=fairness_report
    )
