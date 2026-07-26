"""Kaplan-Meier survival curve estimation.

Thin, typed wrapper around :class:`lifelines.KaplanMeierFitter` returning
plain-data summaries and a consistent plotting convention used throughout
this package's survival modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib.axes import Axes


@dataclass
class KaplanMeierResult:
    """A fitted Kaplan-Meier curve plus convenience summaries.

    Attributes
    ----------
    fitter : lifelines.KaplanMeierFitter
        The underlying fitted model, for direct access to any lifelines
        functionality not wrapped here.
    label : str
        Group label used in plots and summaries.
    median_survival_days : float
        Estimated median survival time. ``inf`` if the median was not
        reached within the observed follow-up.
    n : int
        Number of subjects in this group.
    n_events : int
        Number of observed events (non-censored) in this group.
    """

    fitter: KaplanMeierFitter
    label: str
    median_survival_days: float
    n: int
    n_events: int

    def survival_at(self, times: list[float]) -> pd.Series:
        """Survival probability estimate at each requested time.

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> result = fit_km(rng.exponential(100, 50), np.ones(50), label="all")
        >>> len(result.survival_at([0, 50, 100])) == 3
        True
        """
        return self.fitter.survival_function_at_times(times)


def fit_km(durations, event_observed, label: str = "group") -> KaplanMeierResult:
    """Fit a Kaplan-Meier curve.

    Parameters
    ----------
    durations : array-like
        Observed follow-up time (event or censoring) per subject.
    event_observed : array-like
        1 if the event occurred, 0 if censored.
    label : str, default "group"
        Label for this curve, used in plots and stratified comparisons.

    Returns
    -------
    KaplanMeierResult

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> durations = rng.exponential(100, 200)
    >>> events = rng.binomial(1, 0.7, 200)
    >>> result = fit_km(durations, events, label="cohort")
    >>> result.n
    200
    """
    kmf = KaplanMeierFitter(label=label)
    kmf.fit(durations, event_observed=event_observed, label=label)
    return KaplanMeierResult(
        fitter=kmf,
        label=label,
        median_survival_days=float(kmf.median_survival_time_),
        n=len(durations),
        n_events=int(pd.Series(event_observed).sum()),
    )


def plot_km(results: list[KaplanMeierResult], ax: Axes | None = None, ci_show: bool = True) -> Axes:
    """Plot one or more Kaplan-Meier curves on shared axes.

    Parameters
    ----------
    results : list[KaplanMeierResult]
        Curves to overlay, e.g. one per treatment arm or subgroup.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    ci_show : bool, default True
        Whether to draw the pointwise confidence band for each curve.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> a = fit_km(rng.exponential(100, 50), np.ones(50), label="A")
    >>> b = fit_km(rng.exponential(80, 50), np.ones(50), label="B")
    >>> ax = plot_km([a, b])
    >>> ax.get_legend() is not None
    True
    """
    if ax is None:
        _, ax = plt.subplots()
    for result in results:
        result.fitter.plot_survival_function(ax=ax, ci_show=ci_show)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    return ax
