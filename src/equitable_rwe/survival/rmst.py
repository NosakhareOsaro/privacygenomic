"""Restricted mean survival time (RMST) and two-group RMST differences.

RMST — the area under the survival curve up to a fixed horizon ``tau`` —
is a hazard-model-free summary of survival, useful precisely when the
proportional hazards assumption (tested in
:mod:`equitable_rwe.survival.cox`) does not hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time


@dataclass
class RMSTDifference:
    """Bootstrap comparison of RMST between two groups at horizon ``tau``.

    Attributes
    ----------
    rmst_a, rmst_b : float
        Restricted mean survival time in each group.
    difference : float
        ``rmst_a - rmst_b``.
    ci_lower, ci_upper : float
        Percentile bootstrap confidence interval for ``difference``.
    p_value : float
        Two-sided bootstrap p-value (fraction of bootstrap differences at
        least as extreme as zero, in the direction away from the
        observed difference).
    tau : float
        The truncation horizon used.
    """

    rmst_a: float
    rmst_b: float
    difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    tau: float


def compute_rmst(durations, event_observed, tau: float) -> float:
    """Restricted mean survival time up to ``tau`` for a single group.

    Parameters
    ----------
    durations : array-like
        Observed follow-up times.
    event_observed : array-like
        1 if the event occurred, 0 if censored.
    tau : float
        Truncation horizon. Must not exceed the largest observed time.

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> durations = rng.exponential(200, 300)
    >>> events = np.ones(300)
    >>> rmst = compute_rmst(durations, events, tau=100)
    >>> 0 < rmst <= 100
    True
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=event_observed)
    return float(restricted_mean_survival_time(kmf, t=tau))


def rmst_difference(
    durations_a,
    event_a,
    durations_b,
    event_b,
    tau: float,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> RMSTDifference:
    """Bootstrap the difference in RMST between two groups.

    Parameters
    ----------
    durations_a, event_a : array-like
        Follow-up times and event indicators for group A.
    durations_b, event_b : array-like
        Follow-up times and event indicators for group B.
    tau : float
        Truncation horizon, shared across both groups.
    n_bootstrap : int, default 1000
        Number of bootstrap resamples (within each group, independently).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    RMSTDifference

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> a_t = rng.exponential(150, 200)
    >>> b_t = rng.exponential(100, 200)
    >>> events = np.ones(200)
    >>> result = rmst_difference(a_t, events, b_t, events, tau=100, n_bootstrap=200, seed=0)
    >>> result.difference > 0
    True
    """
    rng = np.random.default_rng(seed)
    durations_a = np.asarray(durations_a)
    event_a = np.asarray(event_a)
    durations_b = np.asarray(durations_b)
    event_b = np.asarray(event_b)

    rmst_a = compute_rmst(durations_a, event_a, tau)
    rmst_b = compute_rmst(durations_b, event_b, tau)
    observed_diff = rmst_a - rmst_b

    boot_diffs = np.empty(n_bootstrap)
    n_a, n_b = len(durations_a), len(durations_b)
    for i in range(n_bootstrap):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)
        boot_a = compute_rmst(durations_a[idx_a], event_a[idx_a], tau)
        boot_b = compute_rmst(durations_b[idx_b], event_b[idx_b], tau)
        boot_diffs[i] = boot_a - boot_b

    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
    # Two-sided bootstrap test of H0: difference = 0, via the fraction of
    # the bootstrap distribution of the difference that crosses zero.
    frac_above = float(np.mean(boot_diffs >= 0))
    frac_below = float(np.mean(boot_diffs <= 0))
    p_value = min(1.0, 2 * min(frac_above, frac_below))

    return RMSTDifference(
        rmst_a=rmst_a,
        rmst_b=rmst_b,
        difference=float(observed_diff),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=p_value,
        tau=tau,
    )
