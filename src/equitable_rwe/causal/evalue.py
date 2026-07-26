"""E-value sensitivity analysis for unmeasured confounding.

The E-value (VanderWeele & Ding, 2017, *Annals of Internal Medicine*) is
the minimum strength of association, on the risk-ratio scale, that an
unmeasured confounder would need to have with both treatment and outcome
to fully explain away an observed association, given the measured
covariates. It answers a question every one of this project's causal
estimators leaves open: *how robust is this estimate to confounding we
didn't measure?* — a robustness check most tutorial-level causal-inference
code skips entirely.

This module computes the E-value for a risk ratio (RR) or odds ratio
approximately equal to RR (rare outcome). For a **hazard ratio**, the
standard practice this module follows is to treat HR as an approximation
of RR, which is only appropriate when cumulative outcome incidence is low
(roughly ≲15%) in both compared groups (VanderWeele & Ding, 2017); for
common outcomes, HR should be converted to RR via an appropriate
rare-outcome-correction transform before calling this function, which
this module does not implement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EValueResult:
    """E-value for a point estimate and for its confidence interval.

    Attributes
    ----------
    estimate_evalue : float
        E-value for the point estimate.
    ci_evalue : float
        E-value for the confidence interval bound closest to the null
        (1.0) — the more conservative, and more commonly reported,
        quantity: how strong unmeasured confounding would need to be to
        move the *confidence interval* to include the null, not just the
        point estimate.
    ratio_measure : float
        The risk-ratio-scale estimate this was computed from (after
        inverting ratios below 1, per the standard E-value convention).
    """

    estimate_evalue: float
    ci_evalue: float
    ratio_measure: float


def _evalue_for_ratio(ratio: float) -> float:
    if ratio < 1:
        ratio = 1 / ratio
    return ratio + (ratio * (ratio - 1)) ** 0.5


def evalue(
    estimate: float, ci_lower: float | None = None, ci_upper: float | None = None
) -> EValueResult:
    """Compute the E-value for a risk-ratio-scale (or HR, rare-outcome) estimate.

    Parameters
    ----------
    estimate : float
        The point estimate, on the risk ratio / hazard ratio scale.
    ci_lower, ci_upper : float, optional
        The confidence interval. If provided, the E-value is also
        computed for whichever bound is closer to the null (1.0) — the
        standard VanderWeele & Ding recommendation, since a CI including
        the null already gives an E-value of 1 for that bound.

    Returns
    -------
    EValueResult

    Examples
    --------
    >>> result = evalue(0.65, ci_lower=0.55, ci_upper=0.77)
    >>> round(result.estimate_evalue, 2)
    2.45
    >>> round(result.ci_evalue, 2)
    1.92
    """
    estimate_evalue = _evalue_for_ratio(estimate)

    ci_evalue = 1.0
    if ci_lower is not None and ci_upper is not None:
        bounds = [ci_lower, ci_upper]
        bound_closest_to_null = min(bounds, key=lambda b: abs(b - 1))
        if (estimate >= 1 and bound_closest_to_null <= 1) or (
            estimate < 1 and bound_closest_to_null >= 1
        ):
            ci_evalue = 1.0
        else:
            ci_evalue = _evalue_for_ratio(bound_closest_to_null)

    return EValueResult(
        estimate_evalue=estimate_evalue, ci_evalue=ci_evalue, ratio_measure=estimate
    )
