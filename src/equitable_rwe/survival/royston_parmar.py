"""Flexible parametric survival regression (Royston-Parmar family).

Royston & Parmar (2002) model the baseline log cumulative hazard with
restricted cubic splines instead of assuming a specific parametric family
(exponential, Weibull, ...) or relying on the semi-parametric Cox partial
likelihood — useful when the proportional hazards assumption
(:mod:`equitable_rwe.survival.cox`) is doubtful but a fully nonparametric
model (:mod:`equitable_rwe.survival.kaplan_meier`) is more than the
question calls for.

This wraps :class:`lifelines.CRCSplineFitter`, which implements Crowther,
Royston & Clements' (2014) accelerated-failure-time (AFT) generalization
of the original Royston-Parmar model, rather than hand-rolling a custom
proportional-hazards-scale spline model. See
``docs/architecture/adr-0002-survival-model-choices.md`` for why.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from lifelines import CRCSplineFitter


@dataclass
class RoystonParmarResult:
    """A fitted flexible parametric (AFT spline) survival model.

    Attributes
    ----------
    fitter : lifelines.CRCSplineFitter
        The underlying fitted model.
    summary : pandas.DataFrame
        Coefficient table, including the ``beta_`` (covariate) and
        ``gamma*_`` (baseline spline) parameters.
    n_baseline_knots : int
        Number of knots used for the baseline spline.
    """

    fitter: CRCSplineFitter
    summary: pd.DataFrame
    n_baseline_knots: int

    def median_survival_time(self) -> float:
        """Median survival time implied by the fitted baseline (covariates at reference)."""
        return float(self.fitter.median_survival_time_)


def fit_royston_parmar(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    n_baseline_knots: int = 3,
) -> RoystonParmarResult:
    """Fit a flexible parametric (Royston-Parmar-family) survival model.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per subject.
    duration_col, event_col : str
        Follow-up time and event indicator columns.
    covariates : list[str]
        Covariates entering the accelerated-failure-time linear predictor
        (``beta_``).
    n_baseline_knots : int, default 3
        Number of interior knots for the baseline log cumulative hazard
        spline. More knots allow a more flexible baseline at the cost of
        numerical stability; if you see a ``StatisticalWarning`` about a
        negative variance diagonal, try reducing this.

    Returns
    -------
    RoystonParmarResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> df = pd.DataFrame(
    ...     {
    ...         "T": rng.weibull(1.3, n) * 50 * np.exp(0.4 * x1),
    ...         "E": rng.binomial(1, 0.85, n),
    ...         "x1": x1,
    ...     }
    ... )
    >>> result = fit_royston_parmar(df, "T", "E", covariates=["x1"])
    >>> result.n_baseline_knots
    3
    """
    regressors = {"beta_": " + ".join(covariates) if covariates else "1"}
    regressors.update({f"gamma{i}_": "1" for i in range(n_baseline_knots)})

    crc = CRCSplineFitter(n_baseline_knots=n_baseline_knots)
    crc.fit(df, duration_col=duration_col, event_col=event_col, regressors=regressors)

    return RoystonParmarResult(fitter=crc, summary=crc.summary, n_baseline_knots=n_baseline_knots)
