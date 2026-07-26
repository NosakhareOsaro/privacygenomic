"""Cox proportional hazards regression with a Schoenfeld-residual PH test.

lifelines' :meth:`CoxPHFitter.check_assumptions` prints its diagnostic
rather than returning a structured, testable result. This module
implements the same underlying idea — correlating scaled Schoenfeld
residuals with (rank-transformed) event time, per Grambsch & Therneau
(1994) — as a plain :class:`ProportionalHazardsTestResult`, so calling code
can assert on it and CI can fail a build on a genuine PH violation.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter
from matplotlib.axes import Axes
from scipy import stats


@dataclass
class ProportionalHazardsTestResult:
    """Per-covariate correlation test of scaled Schoenfeld residuals vs. time.

    A significant correlation (small ``p_values[covariate]``) is evidence
    that covariate's hazard ratio is not constant over time, i.e. the
    proportional hazards assumption is violated for that covariate.

    Attributes
    ----------
    correlations : dict[str, float]
        Pearson correlation between each covariate's scaled Schoenfeld
        residuals and rank-transformed event time.
    p_values : dict[str, float]
        Two-sided p-value for each correlation.
    """

    correlations: dict[str, float]
    p_values: dict[str, float]

    @property
    def any_violated(self) -> bool:
        """``True`` if any covariate's p-value is below 0.05."""
        return any(p < 0.05 for p in self.p_values.values())


def fit_cox_ph(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str] | None = None,
    strata: list[str] | None = None,
    weights_col: str | None = None,
    robust: bool = False,
) -> CoxPHFitter:
    """Fit a Cox proportional hazards model.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per subject.
    duration_col, event_col : str
        Column names for follow-up time and the event indicator (1=event,
        0=censored).
    covariates : list[str], optional
        Columns to include as covariates. Defaults to every column other
        than ``duration_col``, ``event_col``, ``strata``, and
        ``weights_col``.
    strata : list[str], optional
        Columns to stratify on (a separate baseline hazard per stratum,
        shared coefficients).
    weights_col : str, optional
        Column of per-row weights (e.g. IPTW weights).
    robust : bool, default False
        Use a robust (sandwich) variance estimator; required whenever
        ``weights_col`` reflects anything other than integer frequency
        weights (e.g. inverse-probability weights).

    Returns
    -------
    lifelines.CoxPHFitter
        Fitted model.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> df = pd.DataFrame(
    ...     {
    ...         "T": rng.exponential(50, n),
    ...         "E": rng.binomial(1, 0.8, n),
    ...         "x1": rng.normal(size=n),
    ...     }
    ... )
    >>> cph = fit_cox_ph(df, "T", "E", covariates=["x1"])
    >>> "x1" in cph.params_.index
    True
    """
    keep = (
        list(covariates)
        if covariates is not None
        else [
            c
            for c in df.columns
            if c not in {duration_col, event_col, weights_col, *(strata or [])}
        ]
    )
    fit_df = df[[duration_col, event_col, *keep, *(strata or [])]].copy()
    if weights_col is not None:
        fit_df[weights_col] = df[weights_col]

    cph = CoxPHFitter()
    cph.fit(
        fit_df,
        duration_col=duration_col,
        event_col=event_col,
        strata=strata,
        weights_col=weights_col,
        robust=robust,
    )
    return cph


def check_proportional_hazards(
    cph: CoxPHFitter, df: pd.DataFrame, duration_col: str
) -> ProportionalHazardsTestResult:
    """Schoenfeld-residual test of the proportional hazards assumption.

    Parameters
    ----------
    cph : lifelines.CoxPHFitter
        A fitted model (see :func:`fit_cox_ph`).
    df : pandas.DataFrame
        The same data frame used to fit ``cph``.
    duration_col : str
        Follow-up time column, used to build the rank-transformed time
        axis the residuals are correlated against.

    Returns
    -------
    ProportionalHazardsTestResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> df = pd.DataFrame(
    ...     {
    ...         "T": rng.exponential(50, n),
    ...         "E": rng.binomial(1, 0.8, n),
    ...         "x1": rng.normal(size=n),
    ...     }
    ... )
    >>> cph = fit_cox_ph(df, "T", "E", covariates=["x1"])
    >>> result = check_proportional_hazards(cph, df, "T")
    >>> "x1" in result.p_values
    True
    """
    residuals = cph.compute_residuals(df, kind="scaled_schoenfeld")
    event_times = df.loc[residuals.index, duration_col]
    rank_time = event_times.rank()

    correlations: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for covariate in residuals.columns:
        r, p = stats.pearsonr(residuals[covariate], rank_time)
        correlations[covariate] = float(r)
        p_values[covariate] = float(p)

    return ProportionalHazardsTestResult(correlations=correlations, p_values=p_values)


def plot_schoenfeld(
    cph: CoxPHFitter, df: pd.DataFrame, duration_col: str, covariate: str, ax: Axes | None = None
) -> Axes:
    """Scatter plot of a covariate's scaled Schoenfeld residuals vs. time.

    A flat, patternless scatter is consistent with proportional hazards
    for that covariate; a visible trend suggests a time-varying effect.
    """
    if ax is None:
        _, ax = plt.subplots()
    residuals = cph.compute_residuals(df, kind="scaled_schoenfeld")
    event_times = df.loc[residuals.index, duration_col]
    ax.scatter(event_times, residuals[covariate], alpha=0.6)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Scaled Schoenfeld residual ({covariate})")
    return ax
