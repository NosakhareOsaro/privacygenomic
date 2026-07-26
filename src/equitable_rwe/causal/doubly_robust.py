"""Doubly-robust IPTW-adjusted Cox regression.

Combines inverse probability of treatment weighting
(:mod:`equitable_rwe.causal.iptw`) with covariate adjustment in the same
Cox model — the standard "doubly robust" construction for a marginal
treatment effect: the estimate remains consistent if *either* the
propensity model *or* the outcome (Cox) model is correctly specified, not
only if both are. This is **not** a fully augmented AIPW/TMLE estimator
with an explicit second-order-robust augmentation term; it is the
simpler, widely used "weight-and-adjust" doubly robust device (see e.g.
Austin, 2013; Funk et al., 2011). See
``docs/architecture/adr-0003-causal-model-choices.md`` for why this
project stops here rather than implementing a full AIPW survival
estimator.

This is the causal half of this project's headline API: combined with
:mod:`equitable_rwe.fairness`, ``fit_equitable_survival_estimate`` (at
the top level of the package) returns both a causal estimate and a
subgroup fairness report from one call.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from lifelines import CoxPHFitter

from equitable_rwe.causal.iptw import compute_iptw_weights, effective_sample_size
from equitable_rwe.survival.cox import fit_cox_ph


@dataclass
class DoublyRobustResult:
    """A fitted doubly-robust (IPTW-weighted, covariate-adjusted) Cox model.

    Attributes
    ----------
    fitter : lifelines.CoxPHFitter
        The underlying fitted weighted Cox model.
    treatment_col : str
        The treatment column whose coefficient is the causal estimate of
        interest.
    hazard_ratio : float
        ``exp(coef)`` for the treatment effect.
    ci_lower, ci_upper : float
        95% confidence interval for the hazard ratio.
    p_value : float
        p-value for the treatment coefficient.
    effective_sample_size : float
        Kish's effective sample size of the IPTW weights (see
        :func:`equitable_rwe.causal.iptw.effective_sample_size`) — a
        large gap versus the raw sample size signals a small number of
        extreme weights are dominating the estimate.
    """

    fitter: CoxPHFitter
    treatment_col: str
    hazard_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float
    effective_sample_size: float


def fit_doubly_robust_survival(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    treatment_col: str,
    covariates: list[str],
    stabilized: bool = True,
    truncate_at: float | None = 0.99,
) -> DoublyRobustResult:
    """Fit a doubly-robust IPTW-weighted, covariate-adjusted Cox model.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per subject.
    duration_col, event_col : str
        Follow-up time and event indicator columns.
    treatment_col : str
        Binary (0/1) treatment indicator; its coefficient is the causal
        estimate returned.
    covariates : list[str]
        Covariates used both for the propensity model (IPTW weights) and
        as adjustment covariates in the outcome (Cox) model — the source
        of this estimator's double robustness.
    stabilized, truncate_at
        Passed through to :func:`equitable_rwe.causal.iptw.compute_iptw_weights`.

    Returns
    -------
    DoublyRobustResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 600
    >>> x1 = rng.normal(size=n)
    >>> treat = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x1)))
    >>> duration = rng.exponential(50 * np.exp(-0.3 * treat + 0.3 * x1))
    >>> df = pd.DataFrame(
    ...     {"T": duration, "E": rng.binomial(1, 0.9, n), "treat": treat, "x1": x1}
    ... )
    >>> result = fit_doubly_robust_survival(df, "T", "E", "treat", covariates=["x1"])
    >>> result.hazard_ratio > 0
    True
    """
    df = df.copy()
    weights = compute_iptw_weights(
        df, treatment_col, covariates, stabilized=stabilized, truncate_at=truncate_at
    )
    df["_iptw_weight"] = weights

    cph = fit_cox_ph(
        df,
        duration_col,
        event_col,
        covariates=[treatment_col, *covariates],
        weights_col="_iptw_weight",
        robust=True,
    )
    summary = cph.summary.loc[treatment_col]

    return DoublyRobustResult(
        fitter=cph,
        treatment_col=treatment_col,
        hazard_ratio=float(summary["exp(coef)"]),
        ci_lower=float(summary["exp(coef) lower 95%"]),
        ci_upper=float(summary["exp(coef) upper 95%"]),
        p_value=float(summary["p"]),
        effective_sample_size=effective_sample_size(weights),
    )
